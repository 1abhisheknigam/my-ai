# RAVDESS Dataset Overview & Structure

This directory is reserved for the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) audio-only speech dataset. Specifically, this project utilizes the **Audio Speech Files (Actors 01–24)** package, which contains 1,440 audio-only speech trials (60 trials per actor across 24 actors). Due to storage size and version control best practices, raw `.wav` audio files are excluded from git tracking.

## Dataset Source & Direct Download

- **Direct Download Link:** [Audio_Speech_Actors_01-24.zip (215 MB)](https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1)
- **Dataset Repository:** [RAVDESS Zenodo Record](https://zenodo.org/records/1188976)
- **Academic Publication Citation:** 
  Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. *PLoS ONE*, 13(5), e0196391. [https://doi.org/10.1371/journal.pone.0196391](https://doi.org/10.1371/journal.pone.0196391)

---

## File Naming Convention

Each of the 1,440 RAVDESS audio files has a unique filename consisting of a 7-part numerical identifier (e.g., `03-01-05-01-01-01-01.wav`) that defines the stimulus characteristics of the recording.

### Identifier Breakdown

1. **Modality:** `01` = full-AV, `02` = video-only, `03` = audio-only *(Note: Files in this directory use `03`)*
2. **Vocal Channel:** `01` = speech, `02` = song *(Note: Files in this directory use `01`)*
3. **Emotion:** `01` = neutral, `02` = calm, `03` = happy, `04` = sad, `05` = angry, `06` = fearful, `07` = disgust, `08` = surprised
4. **Emotional Intensity:** `01` = normal, `02` = strong *(Note: There is no strong intensity for the neutral emotion)*
5. **Statement:** `01` = "Kids are talking by the door", `02` = "Dogs are sitting by the door"
6. **Repetition:** `01` = 1st repetition, `02` = 2nd repetition
7. **Actor:** `01` to `24` *(Odd-numbered actors are male, even-numbered actors are female)*

### Example Breakdown (`03-01-06-01-02-01-12.wav`)

- **03:** Audio-only
- **01:** Speech
- **06:** Fearful
- **01:** Normal intensity
- **02:** Statement ("Dogs are sitting by the door")
- **01:** 1st repetition
- **12:** 12th Actor (Female, since the ID is even)
