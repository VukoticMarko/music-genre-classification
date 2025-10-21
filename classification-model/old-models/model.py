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
data_dir = '../music-preprocessing/data_opt/large_mel_images'

# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,  # 85% train, 15% validation
)

TOP_K = 3
BATCH_SIZE = 64
INPUT_SHAPE = (112, 112)
CNN_INPUT = (112, 112, 1) # If gray 3rd parameter should be 1, else if RGB go with 3
EPOCHS = 100
MODEL_NAME = "model-opt-LARGE-wMetrics"
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
    layers.RandomFlip("horizontal"),            # Horizontal flip
    #layers.RandomRotation(0.05),               # Rotate +/- 5%
    layers.RandomZoom(0.1, 0.1),                # Zoom in/out
    layers.RandomContrast(0.2),                 # Change contrast
    layers.RandomBrightness(factor=0.2),        # Change brightness
])

# Model Architecture
input_shape = CNN_INPUT  # Matches generator's target_size
num_classes = len(train_generator.class_indices)  # Automatically get number of classes

# model = Sequential([
#     data_augmentation,

#     # Block 1
#     Conv2D(8, kernel_size=3, activation="relu", input_shape=CNN_INPUT),
#     Conv2D(16, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),

#     # Block 2
#     Conv2D(16, kernel_size=3, activation="relu"),
#     Conv2D(32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
#     Dropout(0.25),

#     # Block 3
#     Conv2D(64, kernel_size=3, activation="relu", padding="same"),
#     Conv2D(82, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
#     Dropout(0.25),

#     # Block 4
#     Conv2D(64, kernel_size=3, activation="relu", padding="same"),
#     MaxPooling2D(pool_size=(2, 2)),
#     BatchNormalization(),
#     Dropout(0.3),

#     # Classifier
#     GlobalAveragePooling2D(),
#     Dense(32, activation="relu", kernel_regularizer=REGULARLIZER),
#     Dropout(0.4),
#     Dense(num_classes, activation='softmax', dtype='float32')
# ])

model = Sequential([
    data_augmentation,
    Conv2D(64, kernel_size=3, activation="relu", input_shape=CNN_INPUT),
    Conv2D(64, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(72, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.2),
    Conv2D(84, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.3),
    Conv2D(84, kernel_size=3, activation="relu", kernel_regularizer=REGULARLIZER),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(96, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    Conv2D(72, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER), 
    MaxPooling2D(pool_size=(2, 2)), ####
    Conv2D(64, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    Conv2D(52, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    Dropout(0.1),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(32, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    Dropout(0.3),
    #Flatten(),
    GlobalAveragePooling2D(),
    Dense(20, activation="relu", kernel_regularizer=REGULARLIZER),
    Dense(100, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.4),
    Dense(num_classes, activation='softmax', dtype='float32')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', TopKCategoricalAccuracy(k=TOP_K, name=f'top_{TOP_K}_acc')]
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
    verbose=1,
    min_lr=1e-6
)


# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# # Evaluation
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

# loss, accuracy = model.evaluate(test_generator)
# print(f"Test Loss: {loss:.4f}")
# print(f"Test Accuracy: {accuracy:.4f}")

# PRoveri ovaj nacin
# Predict on test set (probabilities)
y_proba = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_proba, axis=1)
y_true = test_generator.classes
class_indices = train_generator.class_indices
# Map indices to class names in consistent order
inv_map = {v:k for k,v in class_indices.items()}
class_names = [inv_map[i] for i in range(len(inv_map))]

# Basic scores
test_loss, test_acc, test_topk = model.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Top-{TOP_K} Accuracy (from compile): {test_topk:.4f}")


# Sklearn metrics
macro_f1 = f1_score(y_true, y_pred, average='macro')
weighted_f1 = f1_score(y_true, y_pred, average='weighted')
balanced_acc = balanced_accuracy_score(y_true, y_pred)
topk_sklearn = None
try:
    topk_sklearn = top_k_accuracy_score(y_true, y_proba, k=TOP_K, labels=list(range(num_classes)))
except Exception as e:
    print("Could not compute sklearn top-k:", e)

print("\n=== Summary Metrics (sklearn) ===")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
if topk_sklearn is not None:
    print(f"Top-{TOP_K} Accuracy (sklearn): {topk_sklearn:.4f}")

# Per-class precision/recall/f1
prec, rec, f1s, support = precision_recall_fscore_support(y_true, y_pred, labels=range(num_classes), zero_division=0)
print("\nPer-class Precision / Recall / F1 / Support:")
for i, cname in enumerate(class_names):
    print(f"{cname:20s} - P: {prec[i]:.4f}  R: {rec[i]:.4f}  F1: {f1s[i]:.4f}  Support: {support[i]}")

# Full classification report
print("\nFull classification report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Confusion matrix (and plot)
cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
plt.figure(figsize=(8,8))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight')
plt.show()

# Plotting function
def plot_history(history):
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
    plt.savefig('metrics.png')
    plt.show()

 

plot_history(history)

with open('metrics_summary.txt', 'w') as f:
    # TRAIN & VALIDATION metrics
    f.write("=== Train & Validation Metrics ===\n")
    for key in history.history:
        f.write(f"{key}: {history.history[key]}\n")
    f.write("\n")

    # TEST metrics
    f.write("=== Test Metrics ===\n")
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")
    if topk_sklearn is not None:
        f.write(f"Top-{TOP_K} Accuracy (sklearn): {topk_sklearn}\n")
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"Weighted F1: {weighted_f1}\n")
    f.write(f"Balanced Accuracy: {balanced_acc}\n\n")

    f.write("Per-class P / R / F1 / Support:\n")
    for i, cname in enumerate(class_names):
        f.write(f"{cname}: {prec[i]} {rec[i]} {f1s[i]} {support[i]}\n")

model.save(MODEL_NAME)