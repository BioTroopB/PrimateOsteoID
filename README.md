# 🦴 PrimateOsteoID V3

**Non-Human Primate Shoulder Bone Classifier (Landmark-Based)**

Upload a landmark coordinate file (TXT, CSV, or DTA) → instantly predicts **Species + Sex + Side**.

This is the **production-ready version** of the project. Earlier experimental versions (V1 & V2) that worked with raw 3D scans are no longer maintained.

## Accuracy (Holdout Test)

| Bone       | Species   | Sex    | Side   |
|------------|-----------|--------|--------|
| Clavicle   | 89.2%     | 67.6%  | 70.3%  |
| Scapula    | **100.0%**    | 62.2%  | 94.6%  |
| Humerus    | 97.3%     | 62.2%  | 83.8%  |

> **Note on Sex Accuracy**: ~60–70% reflects the known low sexual dimorphism in nonhuman primate shoulder girdles — a biological reality, not a model limitation.

## How to Use

1. Prepare a landmark coordinate file with the correct number of landmarks:
   - **Clavicle**: 7 landmarks
   - **Scapula**: 13 landmarks
   - **Humerus**: 16 landmarks
2. Upload the file to the app
3. Get instant predictions with confidence scores

**Live Demo**: [PrimateOsteoID V3 on Hugging Face](https://huggingface.co/spaces/BioTroopB/PrimateOsteoID-V3)

## Data Summary

- **555 specimens** from 7 nonhuman primate taxa
- **Clavicle**: 185 | **Scapula**: 185 | **Humerus**: 185

## 3D Scan Collection Credits

Scans performed by:
- Brittany Kenyon-Flatt
- Evan Simons
- Marianne Cooper
- Amandine Eriksen
- Kevin P. Klier (*Macaca mulatta*)

## Specimen Collections

Specimens scanned at or loaned from:
- American Museum of Natural History (AMNH)
- Neil C. Tappen Collection, University of Minnesota (NCT)
- Field Museum of Natural History (FMNH)
- Harvard Museum of Comparative Zoology (MCZ)
- University at Buffalo Primate Skeletal Collection (UBPSC)
- Cleveland Museum of Natural History (CMNH)

## Committee

- **Noreen von Cramon-Taubadel, Ph.D.** (Chair)
- **Nicholas J. Holowka, Ph.D.**

## Funding & Support

Research conducted at the **Buffalo Human Evolutionary Morphology Lab (BHEML)**, supported by the **National Science Foundation**.

## Development

- **Code & models**: Kevin P. Klier
- **AI assistance**: Grok (xAI)

## License

- **Code & Models**: MIT License
- **Documentation**: CC-BY 4.0 (cite Kevin P. Klier if reused)

---

*"Bridging biological anthropology and artificial intelligence through geometric morphometrics."*  
— **Kevin P. Klier**, M.A. Anthropology, University at Buffalo, 2023
