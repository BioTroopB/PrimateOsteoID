# PrimateOsteoID — Iterative AI Non-Human Primate Shoulder Bone Classifier (V1)

**M.A. Project Extension – Kevin P. Klier**  
**STEM-Designated Program (DHS CIP 45.0201)**  
University at Buffalo • Buffalo Human Evolutionary Morphology Lab (BHEML)

**⚠️ Important Disclaimer**  
This is an **experimental research prototype** (V1). Predictions are probabilistic and may be inaccurate. Provided "as is" for demonstration and academic purposes only. Not intended for commercial, forensic, or critical applications. The author assumes no liability for use or results.

---

## Live Interactive Demos (All Public)

| Version                  | Description                                                                 | Link                                                                                   |
|--------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------------------------------------|
| **V1** (Stable baseline) | Original prototype — raw .ply scans → full geometric morphometrics pipeline | → https://huggingface.co/spaces/BioTroopB/PrimateOsteoID                               |
| **V2** (Refinement)      | Enhanced raw-scan robustness, auto-landmarking, and outlier rejection       | → https://huggingface.co/spaces/BioTroopB/PrimateOsteoID-V2                            |
| **V3** (Optimized)       | Streamlined pipeline using pre-aligned landmark coordinates (TXT/CSV/DTA) → highly reliable inference | → https://huggingface.co/spaces/BioTroopB/PrimateOsteoID-V3                             |
| **ScapulaID** (Experiment) | Specialized scapula classifier exploring PointNet on manual landmarks → valuable lessons on model choices | → https://huggingface.co/spaces/BioTroopB/ScapulaID                                     |

---

## What It Does

The system classifies nonhuman primate shoulder bones by instantly predicting:  
- **Bone type** (clavicle, humerus, scapula)  
- **Species** (7 taxa)  
- **Sex** (M/F)  
- **Side** (L/R)  

Trained exclusively on **555 real museum specimens** from 7 nonhuman primate taxa (Cercopithecus ascanius, Trachypithecus cristatus, Macaca mulatta, Pan troglodytes, Gorilla gorilla, Pongo pygmaeus, Hylobates lar).

**Input**: Raw .ply bone scan (V1/V2) or pre-aligned landmark coordinates (V3).  
**Output**: Instant predictions with confidence scores + interactive 3D visualization (V2/V3).

---

## Data Summary

| Bone     | Specimens | Males | Females | Left | Right |
|----------|-----------|-------|---------|------|-------|
| Clavicle | 185       | 92    | 93      | 69   | 116   |
| Humerus  | 185       | 91    | 94      | 66   | 119   |
| Scapula  | 185       | 91    | 94      | 88   | 97    |
| **Total** | **555**   | **274** | **281** | **223** | **332** |

---

## Holdout Test Accuracy (V1)

**Classical GM pipeline (GPA + PCA + Random Forest)**

| Bone     | Species   | Sex     | Side    |
|----------|-----------|---------|---------|
| Clavicle | 89.2%     | 67.6%   | 70.3%   |
| Scapula | **100.0%**     | 62.2%   | 94.6%   |
| Humerus | 97.3%     | 62.2%   | 83.8%   |

> **Note on sex classification**: ~60–70% accuracy reflects the known low sexual dimorphism in nonhuman primate shoulder girdles — a biological constraint, not a model shortcoming.

---

## Project Evolution & Key Lessons

- **V1**: End-to-end pipeline from raw scans (GPA → PCA → RandomForest + ONNX deployment)  
- **V2**: Focused on raw-scan robustness and outlier handling → incremental improvements  
- **V3**: Shifted to pre-aligned coordinates → eliminated processing variability and achieved highly reliable, fast inference  
- **ScapulaID**: Experimented with deep learning (PointNet) on manual landmarks → solid results but reinforced value of classical GM approaches for this dataset  

Iterative development demonstrates full ML lifecycle: experimentation, debugging, trade-off analysis, and optimization.

---

## 3D Scan Collection Credits

Scans performed by:  
- Brittany Kenyon-Flatt  
- Evan Simons  
- Marianne Cooper  
- Amandine Eriksen  
- Kevin P. Klier (*Macaca mulatta*)

## Specimen Collections

- American Museum of Natural History (AMNH)  
- Neil C. Tappen Collection, University of Minnesota (NCT)  
- Field Museum of Natural History (FMNH)  
- Harvard Museum of Comparative Zoology (MCZ)  
- University at Buffalo Primate Skeletal Collection (UBPSC)  
- Cleveland Museum of Natural History (CMNH)

---

## Committee

- **Chair**: Noreen von Cramon-Taubadel, Ph.D.  
- **Member**: Nicholas J. Holowka, Ph.D.

---

## Funding & Support

Conducted at **BHEML**, supported by the **National Science Foundation**.  
Data shared with approval from Dr. Noreen von Cramon-Taubadel.

---

## Development

- **Code, models & iterations**: Kevin P. Klier  
- **AI assistance**: Grok (xAI)

---

## License

- **Code & models**: MIT License  
- **Documentation & thesis**: CC-BY 4.0 — cite Kevin P. Klier if reused

---

*"Bridging biological anthropology and artificial intelligence through geometric morphometrics."*  
— Kevin P. Klier, M.A. Anthropology, University at Buffalo, 2023
