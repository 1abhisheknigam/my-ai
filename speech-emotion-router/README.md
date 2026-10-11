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

This project utilizes the **CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) for audio feature extraction and emotional state modeling.

* **Dataset Source:** [Kaggle CREMA-D Repository (ejlok1 mirror)](https://www.kaggle.com/datasets/ejlok1/cremad)
* **Original Publication:**
  Cao, H., Cooper, D. G., Keutmann, M. K., Gur,. R. C., Nenkova, A., & Verma, R. (2014). CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset. *IEEE Transactions on Affective Computing*, 5(4), 377–390.
