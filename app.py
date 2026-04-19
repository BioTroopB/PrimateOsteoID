import streamlit as st
import numpy as np
import onnxruntime as ort
import pickle
import pandas as pd
from pathlib import Path
from scipy.spatial import procrustes
from itertools import permutations, product

@st.cache_resource
def load_bone_resources(bone):
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
    
    return {
        "pca": pca,
        "le_species": le_species,
        "le_sex": le_sex,
        "le_side": le_side,
        "cs_scaler": cs_scaler,
        "mean_shape": mean_shape,
        "sess_species": sess_species,
        "sess_sex": sess_sex,
        "sess_side": sess_side
    }

resources = {
    "clavicle": load_bone_resources("clavicle"),
    "scapula": load_bone_resources("scapula"),
    "humerus": load_bone_resources("humerus")
}

LANDMARK_COUNTS = {
    7: "clavicle",
    13: "scapula",
    16: "humerus"
}

st.set_page_config(page_title="PrimateOsteoID v3", page_icon="🦴", layout="centered")

st.title("🦴 PrimateOsteoID v3")
st.markdown("**Non-Human Primate Shoulder Bone Classifier**")
st.markdown("Trained on 555 manually landmarked specimens using classical geometric morphometrics (PCA + Centroid Size + Random Forest)")

st.info("Upload a **landmark coordinate file** (TXT, CSV, or DTA) with exactly the correct number of landmarks for the bone type. "
        "This tool is designed for verification and classification of already-landmarked data.")

uploaded_file = st.file_uploader(
    "Upload landmark file (TXT, CSV, or DTA)", 
    type=["txt", "csv", "dta"]
)

if uploaded_file is not None:
    with st.spinner("Processing landmarks and predicting..."):
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            if file_ext == "dta":
                df = pd.read_stata(uploaded_file)
            elif file_ext == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, delim_whitespace=True, header=None, on_bad_lines='skip')
            
            coord_cols = df.filter(regex='^(x|X|y|Y|z|Z)').columns
            if len(coord_cols) >= 3:
                landmarks = df[coord_cols[:3]].values
            else:
                landmarks = df.iloc[:, :3].values
            
            landmarks = np.array(landmarks, dtype=np.float64)
            
            if landmarks.shape[1] != 3:
                st.error("File must have exactly 3 coordinate columns (X, Y, Z).")
                st.stop()
            
            landmark_count = landmarks.shape[0]
            if landmark_count not in LANDMARK_COUNTS:
                st.error(f"Invalid landmark count: {landmark_count}. Expected 7 (clavicle), 13 (scapula), or 16 (humerus).")
                st.stop()
            
            bone = LANDMARK_COUNTS[landmark_count]
            st.success(f"Detected bone type: {bone.capitalize()} ({landmark_count} landmarks)")

            res = resources[bone]
            pca = res["pca"]
            cs_scaler = res["cs_scaler"]
            le_species = res["le_species"]
            le_sex = res["le_sex"]
            le_side = res["le_side"]
            sess_species = res["sess_species"]
            sess_sex = res["sess_sex"]
            sess_side = res["sess_side"]
            mean_shape = res["mean_shape"]

            # Compute ORIGINAL CS
            original_cs = np.sqrt(np.sum((landmarks - np.mean(landmarks, axis=0))**2))
            cs_norm = cs_scaler.transform([[original_cs]])[0][0] if original_cs > 0 else 0

            # Auto-detect best orientation by trying both mirroring options and all perms/signs
            min_disparity = np.inf
            best_aligned = None
            best_mirror = None
            for mirror in [False, True]:
                lm_temp = landmarks.copy()
                if mirror:
                    lm_temp[:, 0] *= -1
                for perm in permutations(range(3)):
                    lm_perm = lm_temp[:, list(perm)]
                    for signs in product([-1, 1], repeat=3):
                        lm_sign = lm_perm * np.array(signs)
                        _, aligned, disparity = procrustes(mean_shape, lm_sign)
                        if disparity < min_disparity:
                            min_disparity = disparity
                            best_aligned = aligned
                            best_mirror = mirror

            # Use best alignment
            flat = best_aligned.reshape(1, -1)
            pca_feats = pca.transform(flat)
            features = np.hstack([pca_feats[0], cs_norm]).astype(np.float32)
            features = features.reshape(1, -1)

            # Species
            outputs_species = sess_species.run(None, {"float_input": features})
            pred_species_probs = outputs_species[1][0]
            sum_probs = np.sum(pred_species_probs)
            probs = pred_species_probs / sum_probs if sum_probs > 0 else np.ones_like(pred_species_probs) / len(pred_species_probs)
            pred_species_idx = np.argmax(probs)
            pred_species_label = le_species.inverse_transform([pred_species_idx])[0]
            conf_species = probs[pred_species_idx]

            # Sex
            outputs_sex = sess_sex.run(None, {"float_input": features})
            pred_sex_probs = outputs_sex[1][0]
            pred_sex_idx = np.argmax(pred_sex_probs)
            pred_sex_label = le_sex.inverse_transform([pred_sex_idx])[0]
            conf_sex = pred_sex_probs[pred_sex_idx]

            # Side
            outputs_side = sess_side.run(None, {"float_input": features})
            pred_side_probs = outputs_side[1][0]
            pred_side_idx = np.argmax(pred_side_probs)
            pred_side_label_short = le_side.inverse_transform([pred_side_idx])[0]
            pred_side_label = "Left" if pred_side_label_short == "L" else "Right"
            conf_side = pred_side_probs[pred_side_idx]

            st.success(f"**Species**: {pred_species_label} ({conf_species:.1%} confidence)")
            st.success(f"**Sex**: {pred_sex_label} ({conf_sex:.1%} confidence)")
            st.success(f"**Side**: {pred_side_label} ({conf_side:.1%} confidence)")

            species_probs_df = pd.DataFrame({
                "Species": le_species.classes_,
                "Probability": probs
            }).sort_values("Probability", ascending=False)
            st.bar_chart(species_probs_df.set_index("Species"))

            if conf_species < 0.5 or conf_sex < 0.5 or conf_side < 0.5:
                st.warning("Low confidence in one or more predictions—check landmark file, orientation, or bone type match.")

            st.info("Prediction based on raw landmarks + PCA + Centroid Size + Random Forest. "
                    "Model trained on identical landmark configurations. For citation and details, see README.")

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("File must contain exactly 3 columns of X Y Z coordinates, one landmark per row.")
else:
    st.info("Upload a landmark file to begin. The bone type will be auto-detected based on landmark count.")
    st.markdown("**Clavicle**: 7 landmarks · **Scapula**: 13 landmarks · **Humerus**: 16 landmarks")

st.markdown("---")
st.caption("See README for full citation and project information.")