import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
import pandas as pd
from pathlib import Path
from scipy.spatial import procrustes

# ==================== MUST BE FIRST ====================
st.set_page_config(page_title="PrimateOsteoID V3", page_icon="🦴", layout="centered")

st.title("🦴 PrimateOsteoID V3")
st.markdown("**Non-Human Primate Shoulder Bone Classifier (Landmark-Based)**")

st.info("Upload a landmark coordinate file (TXT, CSV, or DTA). See the README for landmark protocols.")

uploaded_file = st.file_uploader("Upload landmark file (TXT, CSV, or DTA)", type=["txt", "csv", "dta"])

if uploaded_file is not None:
    with st.spinner("Processing..."):
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == "dta":
                df = pd.read_stata(uploaded_file)
            elif file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, on_bad_lines='skip')
            
            coord_cols = df.filter(regex='^(x|X|y|Y|z|Z)').columns
            landmarks = df[coord_cols[:3]].values if len(coord_cols) >= 3 else df.iloc[:, :3].values
            landmarks = np.array(landmarks, dtype=np.float64)
            
            if landmarks.shape[1] != 3:
                st.error("❌ File must have exactly 3 coordinate columns (X, Y, Z).")
                st.stop()
            
            landmark_count = landmarks.shape[0]
            bone_map = {7: "clavicle", 13: "scapula", 16: "humerus"}
            
            if landmark_count not in bone_map:
                st.error(f"❌ Invalid landmark count: {landmark_count}")
                st.stop()
            
            bone = bone_map[landmark_count]
            
            # Load models
            base_path = Path("models_onnx") / bone
            pca = pickle.load(open(base_path / f"pca_{bone}.pkl", "rb"))
            le_species = pickle.load(open(base_path / f"le_species_{bone}.pkl", "rb"))
            le_sex = pickle.load(open(base_path / f"le_sex_{bone}.pkl", "rb"))
            le_side = pickle.load(open(base_path / f"le_side_{bone}.pkl", "rb"))
            cs_scaler = pickle.load(open(base_path / f"cs_scaler_{bone}.pkl", "rb"))
            mean_shape = pickle.load(open(base_path / f"mean_shape_{bone}.pkl", "rb"))
            
            sess_species = ort.InferenceSession(str(base_path / f"model_species_{bone}.onnx"))
            sess_sex = ort.InferenceSession(str(base_path / f"model_sex_{bone}.onnx"))
            sess_side = ort.InferenceSession(str(base_path / f"model_side_{bone}.onnx"))
            
            # Process
            original_cs = np.sqrt(np.sum((landmarks - np.mean(landmarks, axis=0))**2))
            cs_norm = cs_scaler.transform([[original_cs]])[0][0] if original_cs > 0 else 0
            
            _, aligned, _ = procrustes(mean_shape, landmarks)
            flat = aligned.flatten()
            pca_feats = pca.transform([flat])
            features = np.hstack([pca_feats[0], cs_norm]).astype(np.float32).reshape(1, -1)
            
            # Predictions
            prob_species = sess_species.run(None, {"float_input": features})[1][0]
            pred_species = le_species.inverse_transform([np.argmax(prob_species)])[0]
            conf_species = prob_species.max()
            
            prob_sex = sess_sex.run(None, {"float_input": features})[1][0]
            pred_sex = le_sex.inverse_transform([np.argmax(prob_sex)])[0]
            conf_sex = prob_sex.max()
            
            prob_side = sess_side.run(None, {"float_input": features})[1][0]
            pred_side = "Left" if le_side.inverse_transform([np.argmax(prob_side)])[0] == "L" else "Right"
            conf_side = prob_side.max()
            
            # Results
            st.success(f"**Species**: {pred_species} ({conf_species:.1%})")
            st.success(f"**Sex**: {pred_sex} ({conf_sex:.1%})")
            st.success(f"**Side**: {pred_side} ({conf_side:.1%})")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your file format and landmark count.")

else:
    st.info("Upload a landmark file to begin.")
