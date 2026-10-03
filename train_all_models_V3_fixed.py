"""
PrimateOsteoID V3 — full retrain + ONNX export (app-compatible)

Matches BioTroopB/PrimateOsteoID app.py artifact expectations:
  models_onnx/<bone>/
    mean_shape_<bone>.pkl
    pca_<bone>.pkl
    cs_scaler_<bone>.pkl
    le_species_<bone>.pkl
    le_sex_<bone>.pkl      # FIXED: was missing from prior train scripts
    le_side_<bone>.pkl     # FIXED: was missing from prior train scripts
    model_species_<bone>.onnx
    model_sex_<bone>.onnx
    model_side_<bone>.onnx

Run from a directory that contains the three MorphoFile*_CLEAN.txt files
(or set MORPHO_DIR). Requires: numpy, scikit-learn, scipy, imbalanced-learn,
skl2onnx, onnx. Pin scikit-learn==1.5.2 to match the V3 Space pickles.

WARNING: This overwrites models_onnx/. Back up first if you need the shipped weights.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import pickle
from imblearn.over_sampling import SMOTE
from scipy.spatial import procrustes
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

MORPHO_DIR = Path(os.environ.get("MORPHO_DIR", "."))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "models_onnx"))

BONES = {
    "clavicle": MORPHO_DIR / "MorphoFileClavicle_CLEAN.txt",
    "scapula": MORPHO_DIR / "MorphoFileScapula_CLEAN.txt",
    "humerus": MORPHO_DIR / "MorphoFileHumerus_CLEAN.txt",
}


def load_morphofile(filepath: Path):
    names, landmarks = [], []
    current_name = None
    current_lms = []
    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith(("#", "'#")):
                if current_name and current_lms:
                    landmarks.append(np.array(current_lms, dtype=np.float64))
                    names.append(current_name)
                current_name = s.lstrip("#' ").strip()
                current_lms = []
                continue
            try:
                coords = [float(x) for x in s.split() if x]
                if len(coords) == 3:
                    current_lms.append(coords)
            except ValueError:
                pass
    if current_name and current_lms:
        landmarks.append(np.array(current_lms, dtype=np.float64))
        names.append(current_name)
    if not names:
        raise ValueError(f"No specimens found in {filepath}")
    return names, np.stack(landmarks)


def parse_name(name: str):
    # e.g. C_ascanius_M_scapula_L → species, sex, side
    clean = name.strip("'")
    parts = clean.split("_")
    species = "_".join(parts[:-3])
    sex = parts[-3]
    side = parts[-1].upper()
    return species, sex, side


def gpa_align(mirrored: np.ndarray, max_iters: int = 100, tolerance: float = 1e-6):
    mean_shape = np.mean(mirrored, axis=0)
    prev_disparity = np.inf
    aligned = np.zeros_like(mirrored)
    for it in range(max_iters):
        total_disparity = 0.0
        for i in range(len(mirrored)):
            _, mtx2, disparity = procrustes(mean_shape, mirrored[i])
            aligned[i] = mtx2
            total_disparity += disparity
        new_mean = np.mean(aligned, axis=0)
        if abs(prev_disparity - total_disparity) < tolerance:
            print(f"  GPA converged after {it + 1} iterations")
            mean_shape = new_mean
            break
        prev_disparity = total_disparity
        mean_shape = new_mean
    else:
        print(f"  GPA did not converge after {max_iters} iterations")
    return mean_shape, aligned


def export_onnx(model, path: Path, n_features: int):
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=18,
        options={type(model): {"zipmap": False}},
    )
    onnx.save(onx, path)


def train_bone(bone: str, filepath: Path):
    print(f"\n=== TRAINING {bone.upper()} (ONNX EXPORT, app-compatible) ===")
    names, landmarks = load_morphofile(filepath)
    print(f"  Loaded {len(names)} specimens from {filepath.name}")

    species_list, sex_list, side_list = [], [], []
    for n in names:
        sp, sex, side = parse_name(n)
        species_list.append(sp)
        sex_list.append(sex)
        side_list.append(side)

    mirrored = landmarks.copy()
    for i, side in enumerate(side_list):
        if side == "L":
            mirrored[i][:, 0] *= -1

    mean_shape, aligned = gpa_align(mirrored)

    # Centroid size from mirrored (pre-Procrustes scale), as in V3 canonical
    centroid_sizes = np.array(
        [np.sqrt(np.sum((lm - np.mean(lm, axis=0)) ** 2)) for lm in mirrored]
    )
    cs_scaler = StandardScaler()
    cs_normalized = cs_scaler.fit_transform(centroid_sizes.reshape(-1, 1)).ravel()

    flat = aligned.reshape(len(aligned), -1)
    n_components = min(20, flat.shape[1] // 3 - 1)
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(flat)
    features = np.column_stack([features_pca, cs_normalized]).astype(np.float32)

    (
        X_train,
        X_test,
        y_sp_train,
        y_sp_test,
        y_sex_train,
        y_sex_test,
        y_side_train,
        y_side_test,
    ) = train_test_split(
        features,
        species_list,
        sex_list,
        side_list,
        test_size=0.2,
        random_state=42,
        stratify=species_list,
    )

    # --- Encoders (species + sex + side). Fit on TRAIN only; class order
    # matches sklearn RF.classes_ / ONNX label indices used by app.py. ---
    le_species = LabelEncoder()
    le_sex = LabelEncoder()
    le_side = LabelEncoder()
    y_sp_train_enc = le_species.fit_transform(y_sp_train)
    y_sex_train_enc = le_sex.fit_transform(y_sex_train)
    y_side_train_enc = le_side.fit_transform(y_side_train)

    X_res, y_res = SMOTE(random_state=42).fit_resample(X_train, y_sp_train_enc)
    model_species = RandomForestClassifier(
        n_estimators=1200, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model_species.fit(X_res, y_res)

    model_sex = RandomForestClassifier(
        n_estimators=1000, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model_sex.fit(X_train, y_sex_train_enc)

    model_side = RandomForestClassifier(
        n_estimators=800, class_weight="balanced", random_state=42, n_jobs=-1
    )
    model_side.fit(X_train, y_side_train_enc)

    sp_acc = accuracy_score(le_species.transform(y_sp_test), model_species.predict(X_test))
    sex_acc = accuracy_score(le_sex.transform(y_sex_test), model_sex.predict(X_test))
    side_acc = accuracy_score(le_side.transform(y_side_test), model_side.predict(X_test))
    print(f"  Holdout → Species: {sp_acc:.1%} | Sex: {sex_acc:.1%} | Side: {side_acc:.1%}")
    print(f"  Classes sex={list(le_sex.classes_)} side={list(le_side.classes_)}")

    out_dir = OUT_ROOT / bone
    out_dir.mkdir(parents=True, exist_ok=True)

    pickle.dump(mean_shape, open(out_dir / f"mean_shape_{bone}.pkl", "wb"))
    pickle.dump(pca, open(out_dir / f"pca_{bone}.pkl", "wb"))
    pickle.dump(cs_scaler, open(out_dir / f"cs_scaler_{bone}.pkl", "wb"))
    pickle.dump(le_species, open(out_dir / f"le_species_{bone}.pkl", "wb"))
    pickle.dump(le_sex, open(out_dir / f"le_sex_{bone}.pkl", "wb"))
    pickle.dump(le_side, open(out_dir / f"le_side_{bone}.pkl", "wb"))

    n_feat = features.shape[1]
    export_onnx(model_species, out_dir / f"model_species_{bone}.onnx", n_feat)
    export_onnx(model_sex, out_dir / f"model_sex_{bone}.onnx", n_feat)
    export_onnx(model_side, out_dir / f"model_side_{bone}.onnx", n_feat)

    print(f"  Saved → {out_dir}")


def main():
    missing = [p for p in BONES.values() if not p.is_file()]
    if missing:
        raise SystemExit(
            "Missing MorphoFiles (set MORPHO_DIR if needed):\n  "
            + "\n  ".join(str(p) for p in missing)
        )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for bone, path in BONES.items():
        train_bone(bone, path)
    print("\nALL DONE — V3-compatible artifacts in", OUT_ROOT.resolve())
    print("Includes le_sex_*.pkl and le_side_*.pkl required by app.py")


if __name__ == "__main__":
    main()
