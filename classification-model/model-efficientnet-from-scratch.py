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
    precision_recall_fscore_support, confusion_matrix
)

# -----------------------
# USER CONFIG
# -----------------------
DATA_DIR = '../music-preprocessing/data_opt/large_mel_images_rgb_high'
BATCH_SIZE = 400
INPUT_SIZE = (124, 124)
CNN_INPUT = (INPUT_SIZE[0], INPUT_SIZE[1], 3)
EPOCHS = 200
REGULARIZER = tf.keras.regularizers.L2(1e-4)
TOP_K = 3
RUNS_DIR = "runs-efficientnet-fs"
# -----------------------

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
gc.collect()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

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

# --- Data Augmentation ---
data_augmentation = tf.keras.Sequential([
    layers.Input(shape=CNN_INPUT),
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(factor=0.2),
])

# --- EfficientNet model ---
def build_efficientnet(input_shape=CNN_INPUT, num_classes=num_classes):
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=None,  # Training from scratch
        input_shape=input_shape
    )
    inp = keras.Input(shape=input_shape)

    x = data_augmentation(inp)
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=True)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=REGULARIZER)(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = keras.Model(inp, out)
    return model

from tensorflow.keras.metrics import (
    CategoricalAccuracy, TopKCategoricalAccuracy, Precision, Recall, AUC)

model = build_efficientnet()
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

# --- Evaluate ---
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

# --- Save metrics ---
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

print(f"\nEfficientNet training completed and saved to {run_dir}\n")
