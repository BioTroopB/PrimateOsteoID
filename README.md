# 🦴 PrimateOsteoID V3

**Non-Human Primate Shoulder Bone Classifier (Landmark-Based)**

Upload a landmark coordinate file (TXT, CSV, or DTA) → instantly predicts **Species + Sex + Side**.

This is the **production-ready version** of the project. Earlier experimental versions (V1 & V2) that worked with raw 3D scans are no longer maintained.

## How to Use

1. Prepare a landmark coordinate file with **exactly** the correct number of landmarks in the **correct anatomical order**:

   | Bone      | Number of Landmarks | Reference in Master's Project                  |
   |-----------|---------------------|------------------------------------------------|
   | **Clavicle**  | 7                   | Figure 2.1 (p. 12) and Table 2.2 (p. 12)      |
   | **Scapula**   | 13                  | Figure 2.2 (p. 13) and Table 2.3 (p. 13)      |
   | **Humerus**   | 16                  | Figure 2.3 (p. 14) and Table 2.4 (p. 14)      |

2. **Recommended file formats**:
   - `.txt` (space-separated) ← **Best choice**
   - `.csv` (comma-separated) ← **Also excellent**
   - `.dta` (Stata format) ← Works but may have issues with newer Stata versions

3. Upload the file to the app
4. Get instant predictions with confidence scores

**Important**: The order and definitions of the landmarks must match the training data exactly. See the [master's project PDF](masters_project_pectoral_girdle.pdf) for detailed diagrams and landmark definitions.

**Example files** are included in this repository:
- `MorphoFileClavicle_CLEAN.txt`
- `MorphoFileScapula_CLEAN.txt`
- `MorphoFileHumerus_CLEAN.txt`

## Accuracy (Holdout Test)

| Bone       | Species   | Sex    | Side   |
|------------|-----------|--------|--------|
| Clavicle   | 89.2%     | 67.6%  | 70.3%  |
| Scapula    | **100.0%**    | 62.2%  | 94.6%  |
| Humerus    | 97.3%     | 62.2%  | 83.8%  |

> **Note on Sex Accuracy**: ~60–70% reflects the known low sexual dimorphism in nonhuman primate shoulder girdles — a biological reality, not a model limitation.

## Live Demo

→ [PrimateOsteoID V3 on Hugging Face](https://huggingface.co/spaces/BioTroopB/PrimateOsteoID-V3)

## Data Summary

- **555 specimens** from 7 nonhuman primate taxa
- **Clavicle**: 185 | **Scapula**: 185 | **Humerus**: 185

## Credits

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

## Development

- **Code & models**: Kevin P. Klier
- **AI assistance**: Grok (xAI)

## License

- **Code & Models**: MIT License
- **Documentation**: CC-BY 4.0 (cite Kevin P. Klier if reused)

---

*"Bridging biological anthropology and artificial intelligence through geometric morphometrics."*  
— **Kevin P. Klier**, M.A. Anthropology, University at Buffalo, 2023
