# app.py
import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import procrustes
import os

# CONFIG
st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("🦴 OsteoID.ai")
st.markdown("**Primate Shoulder Bone Classifier** — Kevin P. Klier")
st.markdown("*Upload any raw `.ply` file — no landmarking required · Auto-landmarking via ICP*")

# CACHING
@st.cache_resource
def load_bone_models(bone_type):
    base_path = f"models/{bone_type}"
    if not os.path.exists(base_path):
        st.error(f"Models for {bone_type} not found. Check your repo structure.")
        st.stop()
    
    with open(f"{base_path}/mean_shape_{bone_type}.pkl", "rb") as f:
        mean_shape = pickle.load(f)
    with open(f"{base_path}/model_sex_{bone_type}.pkl", "rb") as f:
        model_sex = pickle.load(f)
    with open(f"{base_path}/model_side_{bone_type}.pkl", "rb") as f:
        model_side = pickle.load(f)
    with open(f"{base_path}/model_species_{bone_type}.pkl", "rb") as f:
        model_species = pickle.load(f)
    with open(f"{base_path}/le_species_{bone_type}.pkl", "rb") as f:
        le_species = pickle.load(f)
    with open(f"{base_path}/pca_{bone_type}.pkl", "rb") as f:
        pca = pickle.load(f)
    
    return mean_shape, model_sex, model_side, model_species, le_species, pca

# UI
bone = st.selectbox("Bone type (or Auto-detect)", ["Auto", "clavicle", "scapula", "humerus"])
uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file is not None:
    try:
        mesh = trimesh.load(uploaded_file, file_type="ply", force="mesh")
    except Exception as e:
        st.error(f"Failed to load PLY file. Error: {e}")
        st.stop()

    verts = np.asarray(mesh.vertices)
    st.write(f"Mesh loaded · {len(verts):,} vertices")

    # Auto-detect bone
    if bone == "Auto":
        if len(verts) < 2_000:
            bone = "clavicle"
        elif len(verts) < 6_000:
            bone = "scapula"
        else:
            bone = "humerus"
        st.info(f"Auto-detected as **{bone.capitalize()}**")
        if (1800 < len(verts) < 2200) or (5500 < len(verts) < 6500):
            st.warning("Borderline vertex count—select manually if predictions seem off.")

    # Load all models for this bone (cached!)
    mean_shape, model_sex, model_side, model_species, le_species, pca = load_bone_models(bone)

    # ICP + ALIGNMENT
    @st.cache_data
    def auto_landmark_and_predict(_verts, _mean_shape, _pca):
        # Farthest Point Sampling (FPS) for subsampling
        def fps(points, n_samples):
            if len(points) <= n_samples:
                return points
            dist_mat = squareform(pdist(points))
            idx = [np.random.randint(len(points))]
            for _ in range(1, n_samples):
                dists = np.min(dist_mat[idx], axis=0)
                idx.append(np.argmax(dists))
            return points[np.array(idx)]

        sample_size = min(1000, len(_verts))
        sample = fps(_verts, sample_size)

        # Initial alignment: center and align principal axes of sample to mean
        sample -= np.mean(sample, axis=0)
        mean_centered = _mean_shape - np.mean(_mean_shape, axis=0)
        U_s, _, Vt_s = np.linalg.svd(sample.T @ sample)
        U_m, _, Vt_m = np.linalg.svd(mean_centered.T @ mean_centered)
        R = Vt_m.T @ U_m.T @ U_s @ Vt_s.T  # Approximate rotation
        if np.linalg.det(R) < 0:
            Vt_s[2, :] *= -1
            R = Vt_m.T @ U_m.T @ U_s @ Vt_s.T
        sample = sample @ R

        # Fixed ICP: align src (mean landmarks) to tgt (sample) to estimate landmarks
        def simple_icp(src, tgt, iters=50, threshold=1e-5):
            s = src.copy()
            prev_disp = float('inf')
            for _ in range(iters):
                dists = cdist(s, tgt)
                correspondences = tgt[np.argmin(dists, axis=1)]
                disp, s, _ = procrustes(s, correspondences)  # Align src to correspondences
                if abs(prev_disp - disp) < threshold:
                    break
                prev_disp = disp
            return s

        aligned = simple_icp(_mean_shape, sample)

        # Final GPA to match training
        _, final_landmarks, _ = procrustes(_mean_shape, aligned)
        features = _pca.transform(final_landmarks.flatten().reshape(1, -1))

        # Predictions
        species_pred = le_species.inverse_transform(model_species.predict(features))[0]
        sex_raw = model_sex.predict(features)[0]
        sex_pred = "Male" if sex_raw == 'M' else "Female"
        side_raw = model_side.predict(features)[0]
        side_pred = "Left" if side_raw == 'L' else "Right"

        conf_species = np.max(model_species.predict_proba(features)) * 100
        conf_sex = np.max(model_sex.predict_proba(features)) * 100
        conf_side = np.max(model_side.predict_proba(features)) * 100

        if conf_species < 70:
            st.warning("Low confidence—ensure .ply is clean/oriented like training data.")

        return final_landmarks, species_pred, sex_pred, side_pred, conf_species, conf_sex, conf_side

    with st.spinner("Running auto-landmarking + classification..."):
        landmarks, species, sex, side, c_sp, c_sex, c_side = auto_landmark_and_predict(verts, mean_shape, pca)

    # RESULTS
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Species", f"{species}", f"{c_sp:.1f}% confidence")
        st.metric("Sex", sex, f"{c_sex:.1f}% confidence")
    with col2:
        st.metric("Side", side, f"{c_side:.1f}% confidence")
        st.metric("Bone", bone.capitalize())

    # 3D VISUALIZATION
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[::10, 0], y=verts[::10, 1], z=verts[::10, 2],  # decimate for speed
            color='lightgray', opacity=0.4, name='Specimen'
        ),
        go.Scatter3d(
            x=landmarks[:, 0], y=landmarks[:, 1], z=landmarks[:, 2],
            mode='markers+text', marker=dict(size=6, color='red'), name='Auto-landmarks'
        )
    ])
    fig.update_layout(
        scene_aspectmode='data',
        height=600,
        title="Uploaded Mesh + Detected Landmarks"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a raw `.ply` file to begin identification")
    st.markdown(
        """
        **Supported bones**: Clavicle · Scapula · Humerus  
        **Taxa included**: Cercopithecus ascanius, Trachypithecus cristatus, Gorilla gorilla, Hylobates lar, Macaca mulatta, Pongo pygmaeus, Pan troglodytes  
        Fully automated ICP alignment
        """
    )

st.markdown("---")
st.markdown("© 2023–2025 Kevin P. Klier | Based on M.A. research at University at Buffalo BHEML under Dr. Noreen von Cramon-Taubadel")
