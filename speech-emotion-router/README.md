# Speech Emotion Router

```text
capstone-project/
├── data/
│   ├── raw/                      # Raw RAVDESS and CREMA-D audio archives
│   └── processed/                # Processed feature matrices and train/test splits
├── docs/
│   └── 01_problem_scoping.pdf    # Final official capstone scoping document
├── notebooks/
│   ├── .venv/                    # Local Python virtual environment
│   └── 01_eda_baseline.ipynb     # Active EDA, feature engineering, and baseline models
├── outputs/                      # Generated figures, evaluation plots, and artifacts
├── .gitignore                    # Git tracking exclusions
└── README.md                     # Project overview, installation, and data attribution
```

## Setup & Installation Instructions

To set up your local development environment for exploratory data analysis and baseline modeling, navigate to your project root directory and follow these steps:

1. **System Requirements & Homebrew:**
    This project targets **Python 3.11+** installed via [Homebrew](https://brew.sh/) to ensure compatibility with modern audio processing and machine learning libraries.

    Ensure Homebrew binaries and Python 3.11 are prioritized in your shell path (e.g., `~/.bash_profile`):

    ```bash
    # Prioritize Homebrew and Python 3.11 binaries
    export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:$PATH"
    export PATH="/opt/homebrew/opt/python@3.11/libexec/bin:$PATH"
    ```

    Verify the environment:

    ```bash
    python3 --version  # Should output Python 3.11.x
    ```

2. **Create the Virtual Environment:**

    ```bash
      python3 -m venv .venv
    ```

3. **Activate the Virtual Environment:**

   * On macOS / Linux:

     ```bash
     source .venv/bin/activate
     ```

   * On Windows:

     ```bash
     .venv\Scripts\activate
     ```

4. **Install Dependencies:**

    Upgrade pip and install all necessary dependencies, including `ipykernel` for VS Code Jupyter notebook execution:

    ```bash
    pip install --upgrade pip
    pip install \
      numpy \
      pandas \
      matplotlib \
      seaborn \
      scikit-learn \
      librosa \
      soundfile \
      ipykernel
    ```

5. **Launch Jupyter Lab or VS Code:**

    Open your project directory in VS Code, ensure your notebook kernel is set to your `.venv` interpreter, and execute your notebooks inside the `notebooks/` directory.

## Data Attribution & References

This project utilizes the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) audio-only speech files (Actors 01–24, encompassing 1,440 speech trials) focusing on North American English vocal expressions.

* **Dataset Source & Repository:** [RAVDESS Zenodo Record](https://zenodo.org/records/1188976)
* **Academic Publication Citation:**
  Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)
