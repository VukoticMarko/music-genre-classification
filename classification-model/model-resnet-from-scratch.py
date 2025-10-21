# TRAIN RESNET FROM SCRATCH
import os
import gc
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, top_k_accuracy_score,
    precision_recall_fscore_support, classification_report, confusion_matrix
)

# -----------------------
# USER CONFIG
# -----------------------
DATA_DIR = '../music-preprocessing/data_opt/large_mel_images_rgb_high'
BATCH_SIZE =  2#64
INPUT_SIZE = (72, 72)
CNN_INPUT = (INPUT_SIZE[0], INPUT_SIZE[1], 3)
EPOCHS = 3
REGULARIZER = tf.keras.regularizers.L2(1e-4)
TOP_K = 3
RESNET_TRAINABLE = False  # set True to fine-tune
RUNS_DIR = "runs-resnet-fs"
# -----------------------

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Clear sessions/memory
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
gc.collect()

# Mixed precision
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Create run directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RUNS_DIR, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
print(f"\n🔹 Run directory: {run_dir}\n")

# --- Data ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb',
    shuffle=True
)

# Data augmentation
from tensorflow.keras import layers
data_augmentation = tf.keras.Sequential([
    layers.Input(shape=(CNN_INPUT)),            # Input shape for the model
    #layers.RandomFlip("horizontal"),           # Horizontal flip
    #layers.RandomRotation(0.05),               # Rotate +/- 5%
    layers.RandomZoom(0.1, 0.1),                # Zoom in/out
    layers.RandomContrast(0.2),                 # Change contrast
    layers.RandomBrightness(factor=0.2),        # Change brightness
])

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb',
    shuffle=True
)

num_classes = len(train_generator.class_indices)
print("Detected classes:", train_generator.class_indices)
print("Num classes:", num_classes)

# --- Model ---
data_augmentation = tf.keras.Sequential([
    layers.Input(shape=CNN_INPUT),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(factor=0.2),
])

def build_resnet50(input_shape=CNN_INPUT, num_classes=num_classes, trainable=RESNET_TRAINABLE):
    base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights=None, # Removed weights
        input_shape=input_shape
    )
    base.trainable = True
    inp = keras.Input(shape=input_shape)

    x = data_augmentation(inp)
    x = tf.keras.applications.resnet.preprocess_input(x)

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=REGULARIZER)(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = keras.Model(inp, out)
    return model

from tensorflow.keras.metrics import (
    CategoricalAccuracy, TopKCategoricalAccuracy, Precision, Recall, AUC)

model = build_resnet50()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=[
        CategoricalAccuracy(name='accuracy'),
        TopKCategoricalAccuracy(k=TOP_K, name=f'top_{TOP_K}_acc'),
        Precision(name='precision'),
        Recall(name='recall'),
        AUC(name='auc')
    ]
)
model.summary()

# --- Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(run_dir, "best_model.weights.h5"), monitor="val_loss",
    save_best_only=True, save_weights_only=True, verbose=0
)

# --- Train ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# --- Metrics on validation set ---
y_proba = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_proba, axis=1)
y_true = val_generator.classes

class_indices = train_generator.class_indices
inv_map = {v: k for k, v in class_indices.items()}

present_labels = np.unique(y_true)
class_names_filtered = [inv_map[i] for i in present_labels]

macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_true, y_pred)
try:
    topk_sklearn = top_k_accuracy_score(y_true, y_proba, k=TOP_K, labels=present_labels)
except Exception:
    topk_sklearn = None

print("\n=== Summary Metrics ===")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
if topk_sklearn is not None:
    print(f"Top-{TOP_K} Accuracy (sklearn): {topk_sklearn:.4f}")

train_loss = history.history['loss'][-1]
val_loss = history.history['val_loss'][-1]
train_acc = history.history['accuracy'][-1]
val_acc = history.history['val_accuracy'][-1]
print(f"\nFinal Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}")
print(f"Final Val Accuracy:   {val_acc:.4f}, Val Loss: {val_loss:.4f}\n")

# --- Save metrics summary ---
metrics_path = os.path.join(run_dir, "metrics_summary.txt")
with open(metrics_path, 'w') as f:
    f.write("=== Metrics Summary ===\n")
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"Weighted F1: {weighted_f1}\n")
    f.write(f"Balanced Accuracy: {balanced_acc}\n")

    # Train metrics
    f.write("=== Train Metrics ===\n")
    f.write(f"Train Accuracy: {train_acc:.4f}\n")
    f.write(f"Train Loss: {train_loss:.4f}\n\n")

    # Validation metrics
    f.write("=== Validation Metrics ===\n")
    f.write(f"Validation Accuracy: {val_acc:.4f}\n")
    f.write(f"Validation Loss: {val_loss:.4f}\n\n")

    if topk_sklearn is not None:
        f.write(f"Top-{TOP_K} Accuracy: {topk_sklearn}\n")
    f.write("\nPer-class P/R/F1/Support:\n")
    prec, rec, f1s, support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, zero_division=0
    )
    for i, cname in enumerate(class_names_filtered):
        f.write(f"{cname}: P={prec[i]} R={rec[i]} F1={f1s[i]} Support={support[i]}\n")

# --- Accuracy/Loss plots ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(run_dir, "metrics.png"))
plt.close()

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
plt.figure(figsize=(8, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names_filtered))
plt.xticks(tick_marks, class_names_filtered, rotation=90)
plt.yticks(tick_marks, class_names_filtered)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
plt.close()

# --- Save model ---
model.save_weights(os.path.join(run_dir, "final_model.weights.h5"))
model.save(os.path.join(run_dir, "full_model.keras"))

print(f"\nRun saved to {run_dir}\n")