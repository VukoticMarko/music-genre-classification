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
Script that takes mp3 files and converts them to RGB mel-spectrogram images.
Modified to use entire track instead of only first 30s.
'''

# Set up logging to capture and print errors
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

AUDIO_ROOT = "downloads/archive" # NOTE: Set this up to your downloaded dataset path
TRACKS_CSV = "datasets/fma_metadata/tracks.csv"
SR = 22050
DURATION = 5      # Seconds per segment
OVERLAP = 1.5     # Seconds overlap between segments
# NUM_SEGMENTS no longer used; segments determined dynamically

# STFT settings
N_MELS = 512
N_FFT = 4096
HOP_LENGTH = 256
IMG_SIZE = (224, 224)

# Load metadata & filter for single-label large-subset tracks
tracks = pd.read_csv(TRACKS_CSV, header=[0, 1], index_col=0)
subset = tracks[('set', 'subset')]
genre = tracks[('track', 'genre_top')]
#genre = tracks[('track', 'genres_all')]
large_ids = subset[subset == 'large'].index
single_label = genre.loc[large_ids].dropna()
track_ids = single_label.index

# Function to save mel-spectrogram images in RGB
def save_mel_rgb_image(y, sr, out_path, colormap='magma'):
    try:
        # Compute mel-spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS,
            n_fft=N_FFT, hop_length=HOP_LENGTH)
        S_db = librosa.power_to_db(S, ref=np.max, top_db=80)

        # Normalize to [0, 1]
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

        # Apply colormap (returns MxNx4 RGBA)
        cmap = cm.get_cmap(colormap)
        S_colored = cmap(S_norm)

        # Remove alpha channel and scale to [0,255]
        S_rgb = (S_colored[:, :, :3] * 255).astype(np.uint8)

        # Create and save the image
        img = Image.fromarray(S_rgb)
        img = img.resize(IMG_SIZE, Image.LANCZOS)
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

    # Load the **entire audio file** (new: removed duration=)
    try:
        y, _ = librosa.load(str(path), sr=SR)
    except Exception as e:
        logger.warning(f"Error loading {tid_str}: {e}")
        continue

    # Compute segment length in samples
    segment_length = SR * DURATION

    hop_length = SR * (DURATION - OVERLAP)   # new
    num_segments = int(np.ceil((len(y) - segment_length) / hop_length)) + 1


    # Compute number of segments needed to cover entire track (new: use ceil)
    num_segments = int(np.ceil(len(y) / segment_length))

    # Ensure output directory exists
    out_dir = Path("data_opt/large_mel_images_rgb_high") / genre_label
    #out_dir = Path("data_opt/large_mel_images_rgb_MULTI") / genre_label
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loop over **all segments**, including last partial one
    for seg in range(num_segments):
        start = int(seg * (DURATION - OVERLAP) * SR)
        end = int(start + DURATION * SR)
        # start = seg * hop_length
        # end = start + segment_length
        y_seg = y[start:end]

        # Pad last segment if shorter than segment_length
        if len(y_seg) < segment_length:
            y_seg = np.pad(y_seg, (0, segment_length - len(y_seg)))

        # Skip really small segments
        if len(y_seg) < 0.5 * segment_length:
            continue

        # Skip silent segments
        if y_seg.max() < 1e-4:
            continue

        # Save image
        out_file = out_dir / f"{tid_str}_{seg:02d}.png"
        save_mel_rgb_image(y_seg, SR, str(out_file))

logger.info("RGB mel-spectrogram image extraction completed.")