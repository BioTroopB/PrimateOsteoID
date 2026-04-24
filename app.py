import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
import pandas as pd
from pathlib import Path
from scipy.spatial import procrustes
import base64

# ==================== MUST BE FIRST ====================
st.set_page_config(page_title="PrimateOsteoID V3", page_icon="🦴", layout="centered")

# ==================== CENTERED LOGO + TITLE ====================
def get_base64_image(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

img_b64 = get_base64_image("lab-logo.jpg")
if img_b64:
    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 0px;'>
            <img src='data:image/jpeg;base64,{img_b64}' width='180' style='display: block; margin: 20px auto 10px auto;'/>
            <h1 style='margin-top: 0;'>PrimateOsteoID V3</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown("<h1 style='text-align: center;'>🦴 PrimateOsteoID V3</h1>", unsafe_allow_html=True)

st.markdown("<p style='text-align: center; font-size: 18px;'>Non-Human Primate Shoulder Bone Classifier — Landmark-Based</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 14px; color: #666;'>Predicts species, sex, and side from 3D landmark coordinates</p>", unsafe_allow_html=True)

st.markdown("---")

# ==================== FILE UPLOADER ====================
uploaded_file = st.file_uploader(
    "Upload landmark file (.txt, .csv, or .dta)",
    type=["txt", "csv", "dta"],
    help="Clavicle: 7 landmarks | Scapula: 13 landmarks | Humerus: 16 landmarks"
)

if uploaded_file is not None:
    with st.spinner("Processing landmarks..."):
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()

            if file_ext == "dta":
                df = pd.read_stata(uploaded_file)
            elif file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, on_bad_lines='skip')

            coord_cols = df.filter(regex='^(x|X|y|Y|z|Z)').columns
            landmarks = df[coord_cols[:3]].values if len(coord_cols) >= 3 else df.iloc[:, :3].values
            landmarks = np.array(landmarks, dtype=np.float64)

            if landmarks.shape[1] != 3:
                st.error("❌ File must have exactly 3 coordinate columns (X, Y, Z).")
                st.stop()

            landmark_count = landmarks.shape[0]
            bone_map = {7: "clavicle", 13: "scapula", 16: "humerus"}

            if landmark_count not in bone_map:
                st.error(f"❌ Invalid landmark count: {landmark_count}. Expected 7 (clavicle), 13 (scapula), or 16 (humerus).")
                st.stop()

            bone = bone_map[landmark_count]

            # ==================== LOAD MODELS ====================
            base_path = Path("models_onnx") / bone
            pca         = pickle.load(open(base_path / f"pca_{bone}.pkl", "rb"))
            le_species  = pickle.load(open(base_path / f"le_species_{bone}.pkl", "rb"))
            le_sex      = pickle.load(open(base_path / f"le_sex_{bone}.pkl", "rb"))
            le_side     = pickle.load(open(base_path / f"le_side_{bone}.pkl", "rb"))
            cs_scaler   = pickle.load(open(base_path / f"cs_scaler_{bone}.pkl", "rb"))
            mean_shape  = pickle.load(open(base_path / f"mean_shape_{bone}.pkl", "rb"))

            sess_species = ort.InferenceSession(str(base_path / f"model_species_{bone}.onnx"))
            sess_sex     = ort.InferenceSession(str(base_path / f"model_sex_{bone}.onnx"))
            sess_side    = ort.InferenceSession(str(base_path / f"model_side_{bone}.onnx"))

            # ==================== PROCESS ====================
            original_cs = np.sqrt(np.sum((landmarks - np.mean(landmarks, axis=0))**2))
            cs_norm = cs_scaler.transform([[original_cs]])[0][0] if original_cs > 0 else 0

            _, aligned, _ = procrustes(mean_shape, landmarks)
            flat = aligned.flatten()
            pca_feats = pca.transform([flat])
            features = np.hstack([pca_feats[0], cs_norm]).astype(np.float32).reshape(1, -1)

            # ==================== PREDICTIONS ====================
            prob_species = sess_species.run(None, {"float_input": features})[1][0]
            pred_species = le_species.inverse_transform([np.argmax(prob_species)])[0]
            conf_species = prob_species.max()

            prob_sex = sess_sex.run(None, {"float_input": features})[1][0]
            pred_sex = le_sex.inverse_transform([np.argmax(prob_sex)])[0]
            conf_sex = prob_sex.max()

            prob_side = sess_side.run(None, {"float_input": features})[1][0]
            pred_side = "Left" if le_side.inverse_transform([np.argmax(prob_side)])[0] == "L" else "Right"
            conf_side = prob_side.max()

            LOW_CONF_THRESHOLD = 0.70

            # ==================== RESULTS ====================
            st.markdown("---")
            bone_emoji = {"clavicle": "🦴", "scapula": "🦴", "humerus": "🦴"}
            st.info(f"**Detected bone:** {bone.capitalize()} ({landmark_count} landmarks)")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="Species", value=pred_species, delta=f"{conf_species:.1%} confidence")
                if conf_species < LOW_CONF_THRESHOLD:
                    st.warning("⚠️ Low confidence — interpret with caution")

            with col2:
                st.metric(label="Sex", value=pred_sex, delta=f"{conf_sex:.1%} confidence")
                if conf_sex < LOW_CONF_THRESHOLD:
                    st.warning("⚠️ Low confidence — interpret with caution")

            with col3:
                st.metric(label="Side", value=pred_side, delta=f"{conf_side:.1%} confidence")
                if conf_side < LOW_CONF_THRESHOLD:
                    st.warning("⚠️ Low confidence — interpret with caution")

            # ==================== SPECIES PROBABILITY BREAKDOWN ====================
            with st.expander("📊 Full species probability breakdown"):
                species_probs = {
                    le_species.inverse_transform([i])[0]: float(prob_species[i])
                    for i in range(len(prob_species))
                }
                species_df = pd.DataFrame(
                    list(species_probs.items()),
                    columns=["Species", "Probability"]
                ).sort_values("Probability", ascending=False).reset_index(drop=True)
                species_df["Probability"] = species_df["Probability"].map(lambda x: f"{x:.1%}")
                st.dataframe(species_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your file format and landmark count.")

else:
    st.markdown("""
    <div style='text-align: center; color: #888; padding: 2rem 0;'>
        <p>Upload a landmark coordinate file to begin classification.</p>
        <p style='font-size: 13px;'>Supported formats: .txt (space-delimited), .csv, .dta<br>
        Clavicle: 7 landmarks &nbsp;|&nbsp; Scapula: 13 landmarks &nbsp;|&nbsp; Humerus: 16 landmarks</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("PrimateOsteoID V3 • Buffalo Human Evolutionary Morphology Lab • Kevin P. Klier, M.A.")
