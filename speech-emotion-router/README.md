# Speech Emotion Router

```markdown
speech-emotion-router/
├── data/
│   ├── raw/         # Raw .wav audio files
│   └── processed/   # Extracted feature matrices / DataFrames
├── docs/            # Written project documentation, assignments, and PDFs (e.g., 01_problem_scoping.pdf)
├── assets/          # Static images, diagrams, and figures used in your README or presentation
├── outputs/         # Generated runtime files (models, saved dataframes, exported performance plots)
├── notebooks/       # Jupyter notebooks for EDA and modeling
├── src/             # Reusable Python scripts (feature extraction, training pipeline)
└── README.md        # Project entry point and data dictionary
```

## Data Attribution & References

This project utilizes the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) audio-only speech files (Actors 01–24, encompassing 1,440 speech trials) focusing on North American English vocal expressions.

* **Dataset Source & Repository:** [RAVDESS Zenodo Record](https://zenodo.org/records/1188976)
* **Academic Publication Citation:**
  Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)
  