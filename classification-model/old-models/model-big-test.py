# CPU Config
import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disables all GPUs



import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    top_k_accuracy_score
)
from datetime import datetime
# Create a unique folder for this run
run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir = os.path.join("runs", run_name)
os.makedirs(run_dir, exist_ok=True)


import os # Remove clogged warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
import gc
# Clear any existing sessions and free memory
K.clear_session()
tf.compat.v1.reset_default_graph()
gc.collect()

# Configure GPU
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_memory_growth(gpus[0], True)
#     except RuntimeError as e:
#         print(e)

# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# import os
# os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"



# Path to mel_images directory
data_dir = '../music-preprocessing/data_subset_10p'

# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,  # 85% train, 15% validation
)

TOP_K = 3
BATCH_SIZE = 16
INPUT_SHAPE = (224, 224)
CNN_INPUT = (224, 224, 1) # If gray 3rd parameter should be 1, else if RGB go with 3
EPOCHS = 3
MODEL_NAME = "model-opt-LARGE-wMetrics4070"
REGULARLIZER = tf.keras.regularizers.L2(1e-4)


# Training data
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE, 
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='grayscale'
)

# Validation data
val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='grayscale'
)

# Print class labels
print("Class labels:", train_generator.class_indices)
print(f"Found {train_generator.samples} training images")
print(f"Found {val_generator.samples} validation images")




import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense, 
    Dropout,
    BatchNormalization,
    GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# Data augmentation
from tensorflow.keras import layers
data_augmentation = tf.keras.Sequential([
    layers.Input(shape=(CNN_INPUT)),            # Input shape for the model
    #layers.RandomFlip("horizontal"),            # Horizontal flip
    #layers.RandomRotation(0.05),               # Rotate +/- 5%
    layers.RandomZoom(0.1, 0.1),                # Zoom in/out
    layers.RandomContrast(0.2),                 # Change contrast
    layers.RandomBrightness(factor=0.2),        # Change brightness
])

# Model Architecture
input_shape = CNN_INPUT  # Matches generator's target_size
num_classes = len(train_generator.class_indices)  # Automatically get number of classes


from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")


model = Sequential([
    data_augmentation,

    # Block 1
    Conv2D(32, kernel_size=3, activation="relu", input_shape=CNN_INPUT),
    Conv2D(64, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 2
    Conv2D(128, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.2),
    Conv2D(144, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.3),
    Conv2D(156, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 3
    Conv2D(172, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    Conv2D(180, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 4
    Conv2D(192, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    Conv2D(200, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    Dropout(0.2),
    MaxPooling2D(pool_size=(2, 2)),

    # Block 5
    Conv2D(224, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    Dropout(0.3),

    # Classification head
    GlobalAveragePooling2D(),
    Dense(256, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.1),
    Dense(512, activation="relu", kernel_regularizer=REGULARLIZER),
    Dense(num_classes, activation='softmax', dtype='float32')
])

from tensorflow.keras.metrics import (
    CategoricalAccuracy, TopKCategoricalAccuracy, Precision, Recall, AUC
)
model.compile(
    optimizer=Adam(learning_rate=0.0001),
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

# Callbacks
# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    min_delta=0,
    restore_best_weights=True
)
# LR Scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    verbose=0,
    min_lr=1e-6
)
# Checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(
    filepath=os.path.join(run_dir, "best_model.weights.h5"),    # or use f"{MODEL_NAME}_best.h5"
    monitor="val_loss",                 # monitor val_loss or val_accuracy
    save_best_only=True,                # only overwrite if val_loss improves
    save_weights_only=True,             # only save weights (small file)
    verbose=0
)
# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler, checkpoint,],
)

# Predict on dataset (probabilities)
y_proba = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_proba, axis=1)
y_true = val_generator.classes

# Map indices to class names in consistent order
class_indices = train_generator.class_indices
inv_map = {v: k for k, v in class_indices.items()}

# Only keep classes that are present in this validation set
present_labels = np.unique(y_true)
class_names_filtered = [inv_map[i] for i in present_labels]

# --- Sklearn metrics ---
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, top_k_accuracy_score,
    precision_recall_fscore_support, classification_report, confusion_matrix
)

macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_true, y_pred)
topk_sklearn = None
try:
    topk_sklearn = top_k_accuracy_score(y_true, y_proba, k=TOP_K, labels=present_labels)
except Exception as e:
    print("Could not compute sklearn top-k:", e)

print("\n=== Summary Metrics (sklearn) ===")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
if topk_sklearn is not None:
    print(f"Top-{TOP_K} Accuracy (sklearn): {topk_sklearn:.4f}")

# Per-class precision/recall/f1
prec, rec, f1s, support = precision_recall_fscore_support(
    y_true, y_pred, labels=present_labels, zero_division=0
)
print("\nPer-class Precision / Recall / F1 / Support:")
for i, cname in enumerate(class_names_filtered):
    print(f"{cname:20s} - P: {prec[i]:.4f}  R: {rec[i]:.4f}  F1: {f1s[i]:.4f}  Support: {support[i]}")

# Full classification report
print("\nFull classification report:\n")
print(classification_report(y_true, y_pred, labels=present_labels, target_names=class_names_filtered, zero_division=0))

# --- Save metrics summary ---
metrics_path = os.path.join(run_dir, "metrics_summary.txt")
with open(metrics_path, 'w') as f:
    f.write("=== Metrics Summary ===\n")
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"Weighted F1: {weighted_f1}\n")
    f.write(f"Balanced Accuracy: {balanced_acc}\n")
    if topk_sklearn is not None:
        f.write(f"Top-{TOP_K} Accuracy (sklearn): {topk_sklearn}\n")
    f.write("\nPer-class P / R / F1 / Support:\n")
    for i, cname in enumerate(class_names_filtered):
        f.write(f"{cname}: {prec[i]} {rec[i]} {f1s[i]} {support[i]}\n")

# --- Accuracy/Loss plots ---
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'metrics.png'))
plt.show()

# --- Confusion matrix ---
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names_filtered))
plt.xticks(tick_marks, class_names_filtered, rotation=90)
plt.yticks(tick_marks, class_names_filtered)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(os.path.join(run_dir, 'confusion_matrix.png'))
plt.show()

# --- Save model ---
model.save_weights(os.path.join(run_dir, "final_model.weights.h5"))
model.save(os.path.join(run_dir, "full_model.keras"))

print(f"Run saved to {run_dir}")
