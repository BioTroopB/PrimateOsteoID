# app.py — OsteoID.ai v2.0 (Hylobates-proof + faster + prettier)
import streamlit as st
import trimesh
import numpy as np
import pickle
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.spatial import procrustes
import os

st.set_page_config(page_title="OsteoID.ai", layout="centered")
st.title("OsteoID.ai")
st.markdown("**Primate Pectoral Girdle Classifier** — Kevin P. Klier | UB BHEML")
st.markdown("*Upload any raw `.ply` file — no landmarking required · Auto-landmarking via ICP*")

bone = st.selectbox("Bone type (or Auto-detect)", ["Auto", "clavicle", "scapula", "humerus"])
uploaded_file = st.file_uploader("Upload raw .ply file", type="ply")

if uploaded_file is not None:
    try:
        mesh = trimesh.load(uploaded_file, file_type="ply", force="mesh")
    except Exception as e:
        st.error(f"Could not load PLY file: {e}")
        st.stop()

    verts = np.asarray(mesh.vertices)
    st.write(f"Mesh loaded · {len(verts):,} vertices")

    # Auto-detect bone by vertex count (tuned for your data)
    if bone == "Auto":
        if len(verts) < 2_500:
            bone = "clavicle"
        elif len(verts) < 8_000:
            bone = "scapula"
        else:
            bone = "humerus"
        st.info(f"Auto-detected as **{bone.capitalize()}**")

    # Load models (cached for speed)
    @st.cache_resource
    def load_models(b):
        folder = f"models/{b}"
        return (
            pickle.load(open(f"{folder}/mean_shape_{b}.pkl", "rb")),
            pickle.load(open(f"{folder}/model_sex_{b}.pkl", "rb")),
            pickle.load(open(f"{folder}/model_side_{b}.pkl", "rb")),
            pickle.load(open(f"{folder}/model_species_{b}.pkl", "rb")),
            pickle.load(open(f"{folder}/le_species_{b}.pkl", "rb")),
            pickle.load(open(f"{folder}/pca_{b}.pkl", "rb")),
        )

    mean_shape, model_sex, model_side, model_species, le_species, pca = load_models(bone)

    # ROBUST ICP — fixes Hylobates/low-res crash
    def robust_icp(source, target, max_iters=40):
        src = source.copy()
        n_target = target.shape[0]

        # Ensure we have at least as many points as landmarks
        if len(src) < n_target:
            extra_idx = np.random.choice(len(src), n_target - len(src), replace=True)
            src = np.vstack((src, src[extra_idx]))

        for _ in range(max_iters):
            distances = cdist(src, target)
            indices = np.argmin(distances, axis=1)
            correspondence = target[indices]
            _, src_aligned, _ = procrustes(correspondence, src)
            src = src_aligned

        return src[:n_target]  # Return exactly the right number of points

    # Subsample more points for better stability (especially on gibbons)
    sample_size = min(1500, len(verts))
    sample_idx = np.random.choice(len(verts), size=sample_size, replace=False)
    sample_points = verts[sample_idx]

    with st.spinner("Running auto-landmarking + classification..."):
        auto_landmarks = robust_icp(sample_points, mean_shape)
        _, aligned_landmarks, _ = procrustes(mean_shape, auto_landmarks)
        features = pca.transform(aligned_landmarks.flatten().reshape(1, -1))

        # Predictions
        pred_species = le_species.inverse_transform(model_species.predict(features))[0]
        pred_sex = "Male" if model_sex.predict(features)[0] == 1 else "Female"
        pred_side = "Left" if model_side.predict(features)[0] == "L" else "Right"

        conf_species = np.max(model_species.predict_proba(features)) * 100
        conf_sex = np.max(model_sex.predict_proba(features)) * 100
        conf_side = np.max(model_side.predict_proba(features)) * 100

    # RESULTS
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Species", pred_species, f"{conf_species:.1f}% confidence")
        st.metric("Sex", pred_sex, f"{conf_sex:.1f}% confidence")
    with col2:
        st.metric("Side", pred_side, f"{conf_side:.1f}% confidence")
        st.metric("Bone", bone.capitalize())

    # 3D VISUALIZATION
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[::15, 0], y=verts[::15, 1], z=verts[::15, 2],
            color='lightgray', opacity=0.4, name='Specimen'
        ),
        go.Scatter3d(
            x=auto_landmarks[:, 0], y=auto_landmarks[:, 1], z=auto_landmarks[:, 2],
            mode='markers', marker=dict(size=8, color='red'), name='Auto-landmarks'
        )
    ])
    fig.update_layout(scene_aspectmode='data', height=600, title="Uploaded Mesh + Auto-landmarks")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a raw `.ply` file to begin identification")
    st.markdown("**Supported bones**: Clavicle · Scapula · Humerus  \n**Taxa**: Gorilla, Pan, Pongo, Hylobates, Symphalangus, Papio, Macaca  \nTrained on 555 specimens · Fully automated")

st.markdown("---")
st.markdown("© 2025 Kevin P. Klier | University at Buffalo BHEML | Dr. Noreen von Cramon-Taubadel")
