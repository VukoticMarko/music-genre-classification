import librosa
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import logging
from matplotlib import cm

'''
Script that takes mp3 files and converts them to RGB mel-spectogram images.
'''

# Set up logging to capture and print errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

AUDIO_ROOT = "Downloads/fma_large/fma_large" # NOTE: Set this up to your downloaded dataset path
TRACKS_CSV = "datasets/fma_metadata/tracks.csv"
SR = 22050
DURATION = 30  # Seconds
NUM_SEGMENTS = 4

# STFT settings
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = (224, 224)

# Load metadata & filter for single-label large-subset tracks
tracks = pd.read_csv(TRACKS_CSV, header=[0, 1], index_col=0)
subset = tracks[('set', 'subset')]
genre = tracks[('track', 'genre_top')]
large_ids = subset[subset == 'large'].index
single_label = genre.loc[large_ids].dropna()
track_ids = single_label.index

samples_per_track = SR * DURATION
samples_per_segment = samples_per_track // NUM_SEGMENTS

# Function to save mel-spectrogram images in RGB
def save_mel_rgb_image(y, sr, out_path, colormap='viridis'):
    try:
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Normalize to [0, 1]
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

        # Apply colormap (returns MxNx4 RGBA)
        cmap = cm.get_cmap(colormap)
        S_colored = cmap(S_norm)

        # Remove alpha channel and scale to [0,255]
        S_rgb = (S_colored[:, :, :3] * 255).astype(np.uint8)

        # Create and save the image
        img = Image.fromarray(S_rgb)
        img = img.resize(IMG_SIZE, Image.BICUBIC)
        img.save(out_path)
    except Exception as e:
        logger.error(f"Error saving RGB mel image for {out_path}: {e}")

# Process tracks and save RGB mel-spectrogram images
for tid in tqdm(track_ids):
    genre_label = genre.loc[tid].lower()
    tid = int(tid)
    tid_str = f"{tid:06d}"
    path = Path(AUDIO_ROOT) / tid_str[:3] / f"{tid_str}.mp3"

    # Skip if file doesn't exist
    if not path.exists():
        logger.warning(f"Track {tid_str} missing audio file.")
        continue

    # Load the audio file
    try:
        y, _ = librosa.load(str(path), sr=SR, duration=DURATION)
    except Exception as e:
        logger.warning(f"Error loading {tid_str}: {e}")
        continue

    # Pad if audio length is less than the required duration
    if len(y) < samples_per_track:
        y = np.pad(y, (0, samples_per_track - len(y)))

    # Process segments of the track
    for seg in range(NUM_SEGMENTS):
        start = seg * samples_per_segment
        end = start + samples_per_segment
        y_seg = y[start:end]

        # Skip silent segments
        if y_seg.max() < 1e-4:
            continue

        # Ensure output directory exists
        out_dir = Path("data_opt/large_mel_images_rgb") / genre_label
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{tid_str}_{seg:02d}.png"

        # Save RGB mel-spectrogram image
        save_mel_rgb_image(y_seg, SR, str(out_file))

logger.info("RGB mel-spectrogram image extraction completed.")
