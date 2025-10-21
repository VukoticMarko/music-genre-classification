import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving
import matplotlib.pyplot as plt
import os
import json
import librosa
from tqdm import tqdm
from pathlib import Path

'''
Old script that takes mp3 files and makes them into spectogram images.
'''


# Audio and dataset settings
AUDIO_ROOT = "Downloads/fma_large/fma_large" # NOTE: Set this up to your downloaded dataset path
TRACKS_CSV = "datasets/fma_metadata/tracks.csv"
DURATION = 30       # seconds per track
SR = 22050          # sampling rate
NUM_SEGMENTS = 4    # split each track into x segments
N_MELS = 128        # number of mel bands
N_FFT = 2048        # FFT window size = how many samples per short-time transform
HOP_LENGTH = 512    # lower HOP_LENGTH = more frames = higher time resolution and vice versa

# Create directory for mel-spectrogram images
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

# 1) MEL-SPECTROGRAM EXTRACTOR

def extract_mel_spectrogram(y, sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length)
    # Convert to dB scale
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB  # shape = (n_mels, t_frames)

# 2) MAIN LOADER + IMAGE SAVER

def process_and_save_mel_images(
        audio_root=AUDIO_ROOT, metadata_csv=TRACKS_CSV,
        sr=SR, duration=DURATION, n_segments=NUM_SEGMENTS,
        n_mels=N_MELS):
    # Load metadata
    tracks = pd.read_csv(metadata_csv, header=[0,1], index_col=0)
    subset = tracks[('set','subset')]
    genre_for = tracks[('track','genre_top')].to_dict()
    #small_ids = subset[subset=='small'].index # This is Small Dataset usage
    small_ids = subset[subset=='large'].index # SWITCH TO LARGE DATASET !

    samples_per_track = sr * duration
    samples_per_segment = samples_per_track // n_segments

    IMG_ROOT = 'data/large_mel_images'
    ensure_dir(IMG_ROOT)

    for tid in tqdm(small_ids, desc="Processing mel spectrograms"):
        genre = genre_for.get(tid)
        if not isinstance(genre, str):
            continue

        tid_str = f"{tid:06d}"
        mp3_path = Path(audio_root) / tid_str[:3] / f"{tid_str}.mp3"
        if not mp3_path.is_file():
            continue

        # Load full track
        try:
            y_full, _ = librosa.load(str(mp3_path), sr=sr, duration=duration,
                                     res_type='kaiser_fast')
        except Exception:
            continue
        if len(y_full) < samples_per_track:
            y_full = np.pad(y_full, (0, samples_per_track-len(y_full)))

        # Save images per segment
        for seg in range(n_segments):
            start = seg * samples_per_segment
            end = start + samples_per_segment
            
            y_seg = y_full[start:end]

            # Skip near-silent segments
            if np.max(np.abs(y_seg)) < 1e-4:
                continue

            # Extract mel spectrogram
            S_dB = extract_mel_spectrogram(y_seg, sr, n_mels, N_FFT, HOP_LENGTH)

            # Filter 1: Skip low dynamic range
            if np.max(S_dB) - np.min(S_dB) < 45:
                continue

            # Filter 2: Skip if mean energy is very low (mostly silence)
            if np.mean(S_dB) < -55:  # dB scale, -50 is very quiet
                continue

            # Plot and save
            genre_dir = os.path.join(IMG_ROOT, genre.lower())
            ensure_dir(genre_dir)

            plt.figure(figsize=(2.24, 2.24), dpi=100)  # 224x224 px for ResNet but images are saved 173x172 then they don't have white margin
            librosa.display.specshow(S_dB, sr=sr,
                                     x_axis='time', y_axis='mel', cmap='viridis')
            plt.axis('off')
            
            out_path = os.path.join(genre_dir, f"{tid_str}_{seg:02d}.png")
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    print("Mel-spectrogram images saved to data/mel_images")

# Execute processing
if __name__ == '__main__':
    process_and_save_mel_images()