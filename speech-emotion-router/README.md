# Speech Emotion Router

```text
capstone-project/
├── data/
│   ├── processed/
│   │   ├── 00_ravdess_metadata.csv                  # Parsed filename metadata
│   │   ├── 01_1_df_metadata_with_duration.parquet   # Metadata with audio lengths
│   │   ├── 01_2_duration_outliers_w_silence.parquet # Audited outliers with silence stats
│   │   ├── 01_3_df_cleaned.parquet                  # Cleaned, verified dataset
│   │   └── ser_features.parquet                     # Final 26-MFCC feature matrix
│   └── raw/
│       └── RAVDESS/                                 # [gitignored] Raw audio dataset
│           ├── Audio_Speech_Actors_01-24/           # 24 actor speech folders (1,440 files)
│           └── README.md                            # Dataset documentation
├── docs/
│   ├── 01_problem_scoping.pdf                       # Project scoping brief
│   └── 02_problem_statement.pdf                     # Formal problem statement
├── notebooks/
│   ├── 00_metadata_eda.ipynb                        # Metadata distribution analysis
│   └── 01_signal_eda_features.ipynb                 # Signal audit & MFCC extraction
├── outputs/                                         # Plots, figures, and charts
├── .gitignore                                       # Git tracking exclusions
├── git-prefix.sh                                    # Git commit message hook
└── README.md                                        # Project documentation
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
      # Core data manipulation & numerical computing
      numpy \
      pandas \
      # Data visualization
      matplotlib \
      seaborn \
      # Audio I/O & digital signal processing
      librosa \
      soundfile \
      # Machine learning & modeling
      scikit-learn \
      # High-performance columnar storage
      pyarrow \
      fastparquet \
      # Interactive utilities & progress bars
      ipykernel \
      tqdm
    ```

5. **Launch Jupyter Lab or VS Code:**

    Open your project directory in VS Code, ensure your notebook kernel is set to your `.venv` interpreter, and execute your notebooks inside the `notebooks/` directory.

## Data Attribution & References

This project utilizes the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) audio-only speech files (Actors 01–24, encompassing 1,440 speech trials) focusing on North American English vocal expressions.

* **Dataset Source & Repository:** [RAVDESS Zenodo Record](https://zenodo.org/records/1188976)
* **Academic Publication Citation:**
  Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)
