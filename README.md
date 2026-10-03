# 🦴 PrimateOsteoID V3

**Non-Human Primate Shoulder Bone Classifier (Landmark-Based)**

Upload a landmark coordinate file (TXT, CSV, or DTA) → instantly predicts **Species + Sex + Side**.

This is the **production-ready version** of the project. Earlier experimental versions (V1 & V2) that worked with raw 3D scans are no longer maintained.

---

## How to Use

1. Prepare a landmark coordinate file with **exactly** the correct number of landmarks in the **correct anatomical order**:

   | Bone          | Number of Landmarks | Reference in Master's Project             |
   |---------------|---------------------|-------------------------------------------|
   | **Clavicle**  | 7                   | Figure 2.1 (p. 12) and Table 2.2 (p. 12) |
   | **Scapula**   | 13                  | Figure 2.2 (p. 13) and Table 2.3 (p. 13) |
   | **Humerus**   | 16                  | Figure 2.3 (p. 14) and Table 2.4 (p. 14) |

2. **Recommended file formats**:
   - `.txt` (space-separated) ← **Best choice**
   - `.csv` (comma-separated) ← **Also excellent**
   - `.dta` (Stata format) ← Works but may have issues with newer Stata versions

3. Upload the file using the file uploader button
4. The app automatically detects the bone type from the landmark count
5. Get instant predictions with confidence scores — results flagged ⚠️ if confidence is below 70%

**Important**: The order and definitions of the landmarks must match the training data exactly. See the [master's project PDF](masters_project_pectoral_girdle.pdf) for detailed diagrams and landmark definitions.

**Example files** are included in this repository:
- `MorphoFileClavicle_CLEAN.txt`
- `MorphoFileScapula_CLEAN.txt`
- `MorphoFileHumerus_CLEAN.txt`

---

## Accuracy (Holdout Test)

| Bone       | Species        | Sex    | Side   |
|------------|----------------|--------|--------|
| Clavicle   | 89.2%          | 67.6%  | 70.3%  |
| Scapula    | **100.0%**     | 62.2%  | 94.6%  |
| Humerus    | 97.3%          | 62.2%  | 83.8%  |

> **Note on Sex Accuracy**: ~60–70% reflects the known low sexual dimorphism in nonhuman primate shoulder girdles — a biological reality, not a model limitation.

---

## Live Demo

→ [PrimateOsteoID V3 on Hugging Face](https://huggingface.co/spaces/BioTroopB/PrimateOsteoID-V3)

---

## Data Summary

- **555 specimens** from 7 nonhuman primate taxa
- **Clavicle**: 185 | **Scapula**: 185 | **Humerus**: 185
- All data fully anonymized (internal IDs only — no museum accession numbers visible)
- **Classifier trained exclusively on Morphologika-style landmark coordinate data** (3D point configurations), not on raw mesh / `.ply` scans

### Lab data constraint (why landmark-in, not `.ply`-in)

Labeled 3D surface scans (`.ply`) exist for these specimens, but **Buffalo Human Evolutionary Morphology Lab (BHEML) instruction prohibits training AI models on that mesh / scan data** (and bars use of human data). For that reason:

- All production models in V3 are trained only on **landmark coordinate tables** derived under lab protocols
- Raw scans are **not** used as training input for the classifier or for a learned auto-landmarker
- Earlier experiments that tried raw-scan / auto-landmark paths (V1, V2, ScapulaID) are **unmaintained** and are not the supported workflow
- A future “upload `.ply` → auto-landmarks → classify” front-end would require either a change in that lab rule or an approved landmarking dataset/tool that is **not** trained on the barred BHEML scans

This is a **data-use constraint**, not a claim that mesh landmarking is impossible in general.

---

## Credits

### Project Lead & Development
- **Kevin P. Klier**, M.A. Anthropology, University at Buffalo

### 3D Scan Collection
Scans performed by:
- Brittany Kenyon-Flatt
- Evan Simons
- Marianne Cooper
- Amandine Eriksen
- Kevin P. Klier (*Macaca mulatta*)

### Specimen Collections
- American Museum of Natural History (AMNH)
- Neil C. Tappen Collection, University of Minnesota (NCT)
- Field Museum of Natural History (FMNH)
- Harvard Museum of Comparative Zoology (MCZ)
- University at Buffalo Primate Skeletal Collection (UBPSC)
- Cleveland Museum of Natural History (CMNH)

### Project Committee
- **Chair**: Noreen von Cramon-Taubadel, Ph.D.
- **Member**: Nicholas J. Holowka, Ph.D.

### Funding & Support
Conducted at the **Buffalo Human Evolutionary Morphology Lab (BHEML)**, supported by the **National Science Foundation**.

---

## Development

- **Code & models**: Kevin P. Klier
- **AI pair programming assistance**: Grok (xAI), Claude (Anthropic)

---

## License

- **Code & Models**: MIT License
- **Documentation**: CC-BY 4.0 (cite Kevin P. Klier if reused)

---

*"Bridging biological anthropology and artificial intelligence through geometric morphometrics."*
— **Kevin P. Klier**, M.A. Anthropology, University at Buffalo, 2023
