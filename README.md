# 🦴 PrimateOsteoID V3

**Non-Human Primate Shoulder Bone Classifier (Landmark-Based)**

Upload a landmark coordinate file → instantly predicts **Species + Sex + Side**.

This is the **production-ready version**. V1 and V2 (raw scan versions) have been removed as they are no longer maintained.

## Accuracy (Holdout Test)

| Bone       | Species   | Sex    | Side   |
|------------|-----------|--------|--------|
| Clavicle   | 89.2%     | 67.6%  | 70.3%  |
| Scapula    | **100.0%**    | 62.2%  | 94.6%  |
| Humerus    | 97.3%     | 62.2%  | 83.8%  |

## How to Use

1. Prepare a landmark file with the correct number of points
2. Upload it to the app
3. Get instant predictions

**Live Demo**: [PrimateOsteoID V3](https://huggingface.co/spaces/BioTroopB/PrimateOsteoID-V3)

## Credits

M.A. research by **Kevin P. Klier**  
Buffalo Human Evolutionary Morphology Lab (BHEML) • University at Buffalo

## License

MIT License