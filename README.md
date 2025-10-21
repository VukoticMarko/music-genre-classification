### Neural Networks – Mel-Spectrogram Research

This project investigates automatic music genre classification using Convolutional Neural Networks (CNNs) trained on mel-spectrograms generated from MP3 audio files.
It compares custom CNN, ResNet50V2, and EfficientNetB0 models trained on the Free Music Archive (FMA) dataset under both from-scratch and transfer-learning regimes.

---

### Overview

- Converts MP3 tracks into mel-spectrogram images using Librosa.

- Evaluates model performance across different resolutions, sampling strategies, and class balances.

- Implements data augmentation, mixed precision, and fine-tuning for better generalization.

- Provides an inference pipeline for predicting genres of unseen MP3s.

### Installation

Python 3.8.20+ is used.

There are two requirements.txt files for each part of the project.

- For preprocessing folder:
``` pip
pip install -r requirements-preprocessing.txt
```

- For classification folder:
``` pip
pip install -r requirements-classification.txt
```

### Usage

After downloading FMA dataset, place the tracks.csv file into datasets/fma_metadata and then run one of the preprocessing scripts.

After dataset is generated, in the classification script adjust path to that new processed folder with mel-spectogram images and then run script with a wanted model.
