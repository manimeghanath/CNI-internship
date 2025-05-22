# Emotion Detection with ECG using CNN

This project uses a Convolutional Neural Network (CNN) to predict emotion scores (Valence, Arousal, Dominance) from ECG data in the DREAMER dataset. The code preprocesses two-channel ECG signals, standardizes trial lengths, and trains a CNN for regression-based emotion prediction.

## Project Overview

- **Dataset**: DREAMER, containing ECG data for 23 subjects, with 18 baseline trials (61 seconds, 15616 samples at 256 Hz) and 18 stimuli trials (variable lengths) per subject, plus Valence, Arousal, and Dominance scores (1–9 scale).
- **Objective**: Predict continuous emotion scores using preprocessed ECG signals.
- **Model**: 1D CNN with two convolutional layers, max-pooling, and dense layers.
- **Preprocessing**: High-pass, notch, and band-pass filtering, followed by z-score normalization.
- **Output**: Mean Squared Error (MSE), Mean Absolute Error (MAE), and Pearson correlations for test predictions.

## Prerequisites

- **Python**: 3.8+
- **Dependencies**:
  ```bash
  pip install mne neurokit2 biosppy pywavelets numpy pandas scikit-learn matplotlib tensorflow
