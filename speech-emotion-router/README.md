# Speech Emotion Router: Affective Audio Classification

## Overview

My core research objective is to investigate: **Given an audio voice sample, can a machine learning model reliably identify broad traits of the speaker, specifically gender and emotional state, similar to human perception?**

- In critical communication streams like emergency dispatch lines and customer support centers, callers frequently exhibit high-stress escalations and subtle cues of distress that go unnoticed during manual triage.
- Automated speech perception provides an operational routing mechanism to detect frustration early, enabling immediate escalation to specialized teams.
- Beyond enterprise triage, this capability supports assistive accessibility technologies: hard-of-hearing and neurodivergent individuals (such as those on the autism spectrum) often face barriers interpreting tone and affect from vocal acoustics alone.

An affective classification engine can dynamically enrich real-time captions with emotional context, translating raw sound into actionable social signals.

To answer this question without models simply memorizing individual speaker timbres, this project evaluates acoustic models under a strict **speaker-independent validation framework** (partitioning training and evaluation by speaker identity). The analysis follows a progressive modeling strategy:

- **Baseline & Classical Acoustic Models:** Evaluating linear and kernel-based estimators on standardized tabular audio features (such as Mel-Frequency Cepstral Coefficients) to establish benchmark discrimination.
- **Deep Representation Learning:** Advancing to convolutional neural architectures trained on continuous time-frequency representations to capture subtle temporal inflections, pauses, and pitch dynamics.
- **Empirical Target Milestones:** Benchmarking performance against theoretical chance baselines, aiming for >90% accuracy on binary gender separation and 50-60%+ accuracy/macro-F1 on multi-class emotion classification when evaluated on completely unseen speakers.

While the broader architectural goal is designed to generalize across North American English speech datasets (such as RAVDESS and CREMA-D), this implementation establishes its initial benchmarks, acoustic hygiene, and feature extraction pipeline on the RAVDESS corpus.

---

## Notebooks

### 1. *Metadata Distribution & Dataset Profiling*

&emsp;&emsp;[`notebooks/00_metadata_eda.ipynb`](./notebooks/00_metadata_eda.ipynb)

- Dataset structure validation across 1,440 audio speech trials (24 professional actors, balanced 12M/12F).
- Target mapping for joint gender-emotion classification (16 classes) and validation split planning.

### 2. *Acoustic Hygiene, Signal Auditing & Feature Extraction*

&emsp;&emsp;[`notebooks/01_signal_eda_features.ipynb`](./notebooks/01_signal_eda_features.ipynb)

- Lead/trail silence trimming via logarithmic decibel calibration (`top_db=25`).
- Hardware cleanliness checks: DC offset drift, digital clipping, and peak amplitude verification.
- Duration variance and outlier analysis across statements, vocal intensities, and emotional categories.
- Frame-level 13-band MFCC extraction and static tabular aggregation (mean and standard deviation).

### 3. *Baseline Modeling & Error Diagnostics*

&emsp;&emsp;[`notebooks/02_baseline_modeling.ipynb`](./notebooks/02_baseline_modeling.ipynb)

- Leak-free 5-fold `StratifiedGroupKFold` cross-validation grouped strictly by actor ID.
- Empirical heuristic baseline (`DummyClassifier` ~6.7%) and regularized linear baseline (`LogisticRegression` 38.67%).
- Confusion matrix error auditing demonstrating gender disentanglement vs. intra-gender acoustic confusion.
- Serialization of fitted baseline pipelines and out-of-fold predictions to checkpointed artifacts.

---

## Repository Structure

```text
.
├── README.md                                        # Executive overview, EDA audit, baseline benchmarks, and roadmap
├── data/
│   ├── processed/
│   │   ├── 00_ravdess_metadata.csv                  # Parsed filename metadata
│   │   ├── 01_1_df_metadata_with_duration.parquet   # Metadata with audio lengths
│   │   ├── 01_2_duration_outliers_w_silence.parquet # Audited outliers with silence stats
│   │   ├── 01_3_df_cleaned.parquet                  # Cleaned, verified dataset
│   │   ├── ser_features.parquet                     # Final 26-MFCC static tabular feature matrix
│   │   └── models/
│   │       └── 02_2_baseline_pipelines.joblib       # Serialized baseline models, OOF predictions, and CV metadata
│   └── raw/
│       └── RAVDESS/                                 # [gitignored] Raw audio dataset
│           ├── Audio_Speech_Actors_01-24/           # 24 actor speech folders (1,440 files)
│           └── README.md                            # Dataset documentation
├── docs/
│   ├── 01_problem_scoping.pdf                       # Project scoping brief
│   └── 02_problem_statement.pdf                     # Formal problem statement
├── notebooks/
│   ├── 00_metadata_eda.ipynb                        # Metadata distribution analysis
│   ├── 01_signal_eda_features.ipynb                 # Signal audit & MFCC extraction
│   └── 02_baseline_modeling.ipynb                   # Baseline pipelines & error diagnostics
├── outputs/                                         # Plots, figures, and confusion matrices
├── git-prefix.sh                                    # Git commit message hook
└── .gitignore                                       # Git tracking exclusions
```

## Data Cleaning & Acoustic Exploratory Data Analysis (EDA)

The speech corpus under initial analysis contains 1,440 recordings across 24 professional actors (12 male, 12 female) vocalizing two standardized statements under controlled recording studio conditions.

### Dataset Integrity & Cleanliness Auditing

- **Missing Values & Nulls:** Audited the full metadata index across all 1,440 records; zero missing values, null entries, or corrupted identifiers were found.
- **Duplicate Records:** Confirmed that all 1,440 audio files correspond to unique trials across actor, emotion, vocal intensity, statement, and repetition dimensions.
- **Target Distribution:** Verified balanced class representation across genders (720 male, 720 female) and emotional categories (192 trials per emotion, with 96 trials for neutral due to single-take recordings).

### Signal Hygiene & Cleanliness Checks

Before extracting acoustic features, an automated digital signal hygiene audit evaluated the physical recording quality across all 1,440 raw audio files:

- **Digital Saturation Clipping (`clipped_samples`):** Checked sample values against digital limits ($\pm 1.0$); zero clipped samples were found across the entire dataset.
- **Direct Current Bias (`dc_offset`):** Audited the mean amplitude offset to detect hardware analog-to-digital converter drift; all files showed negligible DC offset ($\vert{}\mu\vert{} < 10^{-4}$), making destructive high-pass filtering unnecessary.
- **Peak Amplitudes (`max_abs_amp`):** Checked amplitude headroom; recordings demonstrated consistent gain staging without hardware distortion.

### Silence Auditing & Trimming

Analysis of raw waveform timing revealed substantial leading and trailing studio silence:

- **Silence Proportion:** Studio buffer silence accounted for an average of 53.2% of raw file duration (roughly 1.0s of non-speech per clip).
- **Decibel Threshold Calibration:** Calibrated a -25 dB silence floor (`top_db=25`) using `librosa.effects.trim` to strip lead and trail buffers without clipping quiet consonant onsets.
- **Active Utterance Isolation:** Trimming reduced average clip duration to an active speech mean of 1.74s, removing non-informative ambient room noise from downstream feature extractions.

### Duration & Outlier Analysis

Auditing the trimmed duration distributions across dataset facets provided empirical justification for retaining all recordings:

- **Lexical & Gender Invariance:** Statement length and actor gender showed no statistically meaningful duration differences.
- **Emotional Modulation:** Duration variance was driven primarily by emotional state and vocal intensity (such as drawn-out vowels in sad and calm states versus rapid bursts in angry states).
- **Outlier Retention Verdict:** Because prolonged trials reflected genuine vocal expressions rather than recording corruption or silence trimming errors, all 1,440 trials were kept without artificial truncation or row dropping.

## Feature Extraction & Engineering

Converting raw audio waveforms into model-ready numerical inputs requires capturing spectral energy distributions without carrying background room noise or allowing identity leakage.

### Acoustic Representation: 13-Band MFCCs

Mel-Frequency Cepstral Coefficients (MFCCs) serve as the primary acoustic feature representation because they compress spectral power distributions onto a logarithmic pitch scale that approximates human auditory frequency perception:

- **Audio Standardization:** Audio signals are loaded at a standardized 22,050 Hz sampling rate, preserving frequencies up to 11 kHz to capture human vocal tract formants and pitch harmonics.

- **Short-Time Spectral Decomposition:** Applied a Short-Time Fourier Transform (STFT) with a 2,048-sample window length and a 512-sample hop length (75% frame overlap) across the trimmed active speech.

- **Mel Filterbank Projection:** Projected power spectrograms onto 13 discrete Mel-frequency bands, applying discrete cosine transformation (DCT) to decorrelate filterbank energies into 13 dynamic MFCC contours.

### Tabular Dimensionality Reduction: Mean & Standard Deviation

Because classical classification models require fixed-dimensional inputs rather than variable-length time series arrays, dynamic frame-level trajectories were condensed into static summary statistics:

- **Curse of Dimensionality Prevention:** Flattening the entire uncollapsed time-series array directly into tabular columns would cause the feature space to explode. For a typical utterance spanning ~150 to 200 time steps, flattening 13 MFCC bands yields over 2,000 to 2,600 features per sample, heavily outstripping the 1,440 trial dataset size ($p > n$) and causing severe tabular overfitting without translation invariance.

- **Central Energy Tendency (Mean):** Computed the column-wise mean across all time frames for each of the 13 coefficients, capturing global speaker timbre, baseline vocal tract resonance, and average spectral tilt.

- **Dynamic Energy Variance (Standard Deviation):** Computed the column-wise standard deviation across time frames for each coefficient, capturing prosodic pitch fluctuations and vocal stability.

- **Final Feature Matrix:** Concatenating the mean and standard deviation vectors produces a 26-dimensional static tabular representation for each audio trial, serialized to `data/processed/ser_features.parquet`.

### Methodological Decisions & Two-Phase Roadmap

Evaluating these engineered features led to two core structural modeling choices:

- **Strict Speaker-Partitioned Validation (Group K-Fold):** Random train-test splits introduce severe leakage because an individual speaker's vocal tract geometry and pitch baseline appear in both splits. To guarantee the model evaluates generalized emotion recognition rather than speaker memorization, all cross-validation folds are partitioned strictly by Actor ID, paired with an in-pipeline `StandardScaler` fit exclusively on training splits.

- **Two-Phase Modeling Strategy:**

  - *Phase 1 (Tabular Acoustic Baseline):* Benchmark classical models (Logistic Regression, Support Vector Classifiers) on the compact 26 static MFCC summary statistics to quantify linear gender separation and establish baseline emotion performance.
  - *Phase 2 (Deep Spectral Representation):* Transition to uncollapsed 2D Log-Mel spectrogram arrays paired with a 2D Convolutional Neural Network (CNN). Unlike flattened tabular models, CNN kernels leverage weight sharing and local receptive fields to capture dynamic pitch inflections, cadence, and pauses across time and frequency without exploding parameters.

## Baseline Modeling, Diagnostic Results & Two-Phase Strategy

To evaluate the predictive ceiling of static acoustic features on speaker-independent affective classification, an empirical heuristic baseline and a regularized linear model were trained and audited across identical partitions.

### Validation Scheme & Performance Metrics

- **Speaker-Independent Grouping:** Models were evaluated using 5-Fold `StratifiedGroupKFold` partitioned strictly by `actor_id`. This setup guarantees that all four to five actors in each test fold are completely unseen during training, preventing models from leveraging speaker-specific vocal tract familiarity.

- **Primary Metric (Out-of-Fold Multiclass Accuracy):** Measures exact match performance across the 16 joint gender-emotion categories against a theoretical chance level of 6.25% (or 6.67% when accounting for single neutral takes).

- **Secondary Metric (Macro-Averaged F1-Score):** Ensures an unweighted, equal evaluation across all 16 categories, penalizing models that collapse subtle affective classes into dominant high-energy categories.

### Benchmark Modeling Results

| Model Index | Model Configuration | Feature Space | Out-of-Fold Accuracy | Macro F1 | Fit Time (s) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **B1** | DummyClassifier (Stratified) | Uniform heuristic | **~6.7%** | ~0.06 | < 0.01 |
| **B2** | LogisticRegression (L2 Regularized) | 26 Scaled MFCC Summaries | **38.67%** | 0.380 | 0.18 |

### Diagnostic Findings & Error Auditing

<!-- Class-wise F1 Bar Chart across 16 Categories -->
<figure align="center">
  <img src="outputs/plots/02_baseline_class_f1_scores.png" alt="Class-wise F1 Scores: Baseline Model" width="850">
  <figcaption><em>Figure 1: Out-of-fold F1-scores across all 16 joint gender-emotion categories evaluated using 5-fold StratifiedGroupKFold on unseen speakers.</em></figcaption>
</figure>

<!-- Out-of-Fold 16-Class Confusion Matrix -->
<figure align="center">
  <img src="outputs/plots/02_baseline_confusion_matrix.png" alt="Confusion Matrix: Logistic Regression Baseline" width="850">
  <figcaption><em>Figure 2: Normalized out-of-fold confusion matrix for the L2-regularized logistic regression baseline across 1,440 speech trials.</em></figcaption>
</figure>

### Gender Disentanglement Audit

Collapsing the 16-class predictions into binary speaker gender reveals where the linear model succeeds and where it fails:

<!-- Prediction Outcome Breakdown: Exact Match vs Intra-Gender vs Cross-Gender -->
<figure align="center">
  <img src="outputs/plots/02_baseline_gender_confusion.png" alt="Baseline Prediction Outcome Breakdown" width="850">
  <figcaption><em>Figure 3: Prediction outcome decomposition illustrating high binary gender separation (~94.5%) alongside dominant intra-gender valence confusion (>55%).</em></figcaption>
</figure>

- **Linear Gender Separability:** The regularized logistic regression baseline achieves approximately 94.5% binary gender accuracy on unseen speakers. Cross-gender classification errors represent only ~5.5% of total predictions. Fundamental frequency shifts and spectral tilt captured by static MFCC means provide strong linear separability across biological sex.

- **Intra-Gender Emotional Overlap:** Over 55% of all predictions fall into the intra-gender error category, where the model correctly predicts speaker gender but selects the incorrect affective state. The linear decision boundary struggles primarily between emotions that share vocal arousal levels:
  - *Low-Arousal Overlap:* High mutual misclassification between *Calm*, *Neutral*, and *Sadness*.
  - *High-Arousal Overlap:* High mutual misclassification between *Angry*, *Fearful*, and *Disgusted*.

### Top Intra-Gender Misclassification Pairs

Auditing the highest error rates across specific off-diagonal pairs demonstrates where static spectral summaries break down:

| True Label | Predicted Label | Pair Misclassification Rate | Acoustic Failure Mechanism |
| :--- | :--- | :---: | :--- |
| **Male Neutral** | **Male Surprised** | 27.08% | Unvoiced stop bursts and introductory breath resets mistaken for transient arousal cues |
| **Male Surprised** | **Male Fearful** | 23.96% | High vocal arousal overlap; both display elevated energy variance and pitch expansion |
| **Female Sad** | **Female Disgust** | 20.83% | Low vocal energy and downward formant trajectory overlap without temporal pause context |
| **Female Neutral** | **Female Sad** | 20.83% | Flat spectral envelope and minimal prosodic variance shared across low-arousal states |
| **Male Surprised** | **Male Happy** | 19.79% | Shared high-arousal valence confusion; static MFCC averages fail to trace upward pitch contours |

- **Valence Blindness:** High-arousal pairs (such as *surprised* versus *fearful* or *happy*) exhibit near-identical mean energy and standard deviation across frequency bands. Disentangling positive valence (happy) from negative or shock valence (fear, surprise) requires tracking dynamic pitch trajectories and vowel duration over time.

- **Low-Energy Envelope Confusion:** Flat, subdued vocal deliveries (*neutral*, *sad*, and *calm*) share depressed spectral tilt and low coefficient variance, causing the linear baseline to oscillate between these classes within the same speaker sex.

### Reassessing the Two-Phase Modeling Strategy

The baseline performance ceiling of 38.67% directly verified earlier methodological hypotheses:

- **Limits of Linear Boundaries on Valence:** While vocal arousal produces clear shifts in energy and variance, emotional valence (positive vs. negative affect, such as happy vs. angry) relies on pitch contours, inflection trajectories, and temporal pause structure. Linear hyperplanes fit to static mean and standard deviation vectors cannot separate overlapping intra-gender clusters.

- **Pivot to the Two-Phase Architecture:**

- *Phase 1 Milestone (Non-Linear Tabular Ceiling):* Evaluate a kernelized Support Vector Classifier (`SVC(kernel="rbf")`) on the existing 26-dimensional static tabular set to test whether non-linear boundary warping can resolve intra-gender confusion without temporal frames.
  - *Phase 2 Milestone (Deep 2D Spectral Representation):* Transition directly to fixed-length 2D Log-Mel spectrogram arrays input into a 2D Convolutional Neural Network (CNN). Preserving time-frequency geometry enables convolutional filters to detect prosodic pacing and pitch contours, providing the representational depth needed to target the 50% to 60%+ emotion classification goal on unseen speakers.

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
    pip install -r requirements.txt
    ```

5. **Launch Jupyter Lab or VS Code:**

    Open your project directory in VS Code, ensure your notebook kernel is set to your `.venv` interpreter, and execute your notebooks inside the `notebooks/` directory.

    Notebooks are prefixed with a number `00`, `01`, `02`.. representing which order to run them in.

## Data Attribution & References

This project utilizes the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) audio-only speech corpus (Actors 01 to 24, comprising 1,440 speech trials) focused on North American English vocal expressions.

### Primary Dataset Citation & Repository

- **Academic Publication:** Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)
- **Dataset Repository:** [RAVDESS Zenodo Record (1188976)](https://zenodo.org/records/1188976)
- **License:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

### Digital Audio Fundamentals & Signal Hygiene

- [WavePad Audio Concepts Guide](https://help.nchsoftware.com/help/en/wavepad/win/concepts.html): Technical definitions for DC offset, digital clipping thresholds, sample rates, and dynamic range.
- [Listen2It Audio Cleanup Techniques](https://www.getlisten2it.com/blog/audio-cleanup-techniques-fix-noisy-recordings-like-a-pro): Practical guidelines on baseline noise evaluation, studio room tone, and non-destructive filtering.

### Feature Engineering & MFCC Implementation Guides

- [GeeksforGeeks: Mel-Frequency Cepstral Coefficients (MFCC) for Speech Recognition](https://www.geeksforgeeks.org/nlp/mel-frequency-cepstral-coefficients-mfcc-for-speech-recognition/): Mathematical breakdown of the Mel scale, filterbank projection, and DCT decorrelation.
- [Medium: The Dummy's Guide to MFCC](https://medium.com/prathena/the-dummys-guide-to-mfcc-aceab2450fd): Intuitive overview of frequency warping and cepstral analysis for acoustic feature extraction.
- [GitHub Gist: Extracting and Visualizing MFCCs with Librosa](https://gist.github.com/stevemclaugh/80f192130852353ad53e6d8b6b275983): Practical code patterns for computing frame-level spectrograms and feature arrays using `librosa`.
