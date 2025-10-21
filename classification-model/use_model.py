import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from matplotlib import cm
import matplotlib.pyplot as plt
import os

# Load trained model
model = load_model("runs-small-residual-model/run_20251014_005746_4class/full_model.keras")

# Audio and spectrogram settings
SR = 22050
DURATION = 30
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
IMG_SIZE = (224, 224)

# Class labels
class_labels = [
    'blues',
    'classical',
    'country',
    'easy listening',
    'electronic',
    'experimental',
    'folk',
    'hip-hop',
    'instrumental',
    'international',
    'jazz',
    'old-time',
    'pop',
    'rock',
    'soul-rnb',
    'spoken'
]

class_labels_4class = [
    'electronic',
    'experimental',
    'hip-hop',
    'rock',
]

def create_rgb_mel_spectrogram(audio_path, colormap='viridis'):
    """Convert MP3 to RGB mel-spectrogram image array."""
    y, sr = librosa.load(audio_path, sr=SR, duration=DURATION)

    # Pad if shorter than 30s
    if len(y) < SR * DURATION:
        y = np.pad(y, (0, SR * DURATION - len(y)))

    # Compute mel-spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)

    # Apply colormap
    cmap = cm.get_cmap(colormap)
    S_rgb = cmap(S_norm)[:, :, :3]  # Drop alpha channel
    S_rgb = (S_rgb * 255).astype(np.uint8)

    # Resize and convert to model input format
    img = Image.fromarray(S_rgb).resize(IMG_SIZE, Image.BICUBIC)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dim

    return img_array

def predict_genre(audio_path):
    """Predict genre from MP3 using trained model."""
    spectrogram = create_rgb_mel_spectrogram(audio_path)
    predictions = model.predict(spectrogram)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Get top 3 predictions
    top3_idx = np.argsort(predictions[0])[-3:][::-1]
    top3 = [(class_labels_4class[i], float(predictions[0][i])) for i in top3_idx]

    return {
        "predicted_genre": class_labels_4class[predicted_class],
        "confidence": confidence,
        "top3": top3
    }

# Example usage
if __name__ == "__main__":
    mp3_path = "Music/Kanye West - Good Morning.mp3"

    if not os.path.exists(mp3_path):
        print(f"Error: File not found at {mp3_path}")
    else:
        result = predict_genre(mp3_path)
        print("\nPrediction Results:")
        print(f"Most likely genre: {result['predicted_genre']} ({result['confidence']:.2%})")
        print("\nTop 3 predictions:")
        for genre, prob in result['top3']:
            print(f"- {genre}: {prob:.2%}")

        # Optional: visualize the RGB mel-spectrogram
        spectrogram = create_rgb_mel_spectrogram(mp3_path)[0]
        plt.imshow(spectrogram)
        plt.title(f"Predicted: {result['predicted_genre']} ({result['confidence']:.1%})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
