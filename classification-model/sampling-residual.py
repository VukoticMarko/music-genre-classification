# CPU Config
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Uncomment to disable GPU

import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(4)
import numpy as np
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_recall_fscore_support, top_k_accuracy_score
)
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import (
    CategoricalAccuracy, Precision, Recall, AUC
)
import matplotlib.pyplot as plt
from keras import backend as K
import gc

# Clean up any previous session
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K.clear_session()
tf.compat.v1.reset_default_graph()
gc.collect()

# === Custom sampling utilities ===
from sampling import sample_dataset, delete_sampled_dataset

# === Run setup ===
run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
run_dir = os.path.join("runs-sampled-residual-model", run_name)
os.makedirs(run_dir, exist_ok=True)

# === Dataset paths ===
data_dir = '../music-preprocessing/data_opt/large_mel_images_rgb'

# Create sampled dataset (temporary)
sampled_dir = os.path.join(run_dir, "sampled_data")
sampled_path = sample_dataset(
    source_dir=data_dir,
    dest_dir=sampled_dir,
    max_images_per_class=5000,
    val_split=0.10,
    test_split=0.10
)

# === Constants ===
TOP_K = 3
BATCH_SIZE = 16
INPUT_SHAPE = (224, 224)
CNN_INPUT = (224, 224, 3)
EPOCHS = 200
REGULARIZER = tf.keras.regularizers.L2(1e-4)
MODEL_NAME = "model-opt-SAMPLED-wMetrics1070"

# === Data generators ===
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_dir = os.path.join(sampled_path, 'train')
val_dir = os.path.join(sampled_path, 'val')
test_dir = os.path.join(sampled_path, 'test')

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=INPUT_SHAPE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb'
)

val_generator = test_datagen.flow_from_directory(
    val_dir,
    target_size=INPUT_SHAPE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb', shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=INPUT_SHAPE, batch_size=BATCH_SIZE,
    class_mode='categorical', color_mode='rgb', shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Train: {train_generator.samples}, Val: {val_generator.samples}, Test: {test_generator.samples}")
print("Class labels:", train_generator.class_indices)

# === Data augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.Input(shape=CNN_INPUT),
    layers.RandomZoom(0.1, 0.1),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(factor=0.2),
])

# === Mixed precision ===
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

# === Residual block ===
def residual_block(x, filters, kernel_size=3, dropout_rate=0.1):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding="same", activation="relu", kernel_regularizer=REGULARIZER)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, kernel_size, padding="same", activation=None, kernel_regularizer=REGULARIZER)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

# === Model definition ===
inputs = layers.Input(shape=CNN_INPUT)
x = data_augmentation(inputs)
x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = residual_block(x, 32)
x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = residual_block(x, 64)
x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu", kernel_regularizer=REGULARIZER)(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
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

# === Callbacks ===
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(run_dir, "best_model.weights.h5"),
    monitor="val_loss", save_best_only=True, save_weights_only=True
)

# === Training ===
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler, checkpoint]
)

# === Evaluation ===
y_proba = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_proba, axis=1)
y_true = val_generator.classes
class_indices = train_generator.class_indices
inv_map = {v: k for k, v in class_indices.items()}
present_labels = np.unique(y_true)
class_names_filtered = [inv_map[i] for i in present_labels]

# === Metrics ===
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
if topk_sklearn: print(f"Top-{TOP_K} Accuracy: {topk_sklearn:.4f}")

from sklearn.metrics import ConfusionMatrixDisplay

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names_filtered)

plt.figure(figsize=(10, 10))
disp.plot(cmap='Blues', values_format='d', xticks_rotation='vertical')
plt.title("Confusion Matrix (Validation Set)")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
plt.close()


# === Save results ===
metrics_path = os.path.join(run_dir, "metrics_summary.txt")
with open(metrics_path, 'w') as f:
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"Weighted F1: {weighted_f1}\n")
    f.write(f"Balanced Accuracy: {balanced_acc}\n")
    if topk_sklearn:
        f.write(f"Top-{TOP_K} Accuracy: {topk_sklearn}\n")

# === Test Evaluation ===
print("\n=== Evaluating on Test Set ===")
test_loss, test_acc, test_topk, test_precision, test_recall, test_auc = model.evaluate(test_generator, verbose=1)

# === Predict test data ===
y_test_proba = model.predict(test_generator, verbose=1)
y_test_pred = np.argmax(y_test_proba, axis=1)
y_test_true = test_generator.classes

# === Compute extra metrics ===
test_macro_f1 = f1_score(y_test_true, y_test_pred, average='macro')
test_weighted_f1 = f1_score(y_test_true, y_test_pred, average='weighted')
test_balanced_acc = balanced_accuracy_score(y_test_true, y_test_pred)

try:
    test_topk_sklearn = top_k_accuracy_score(y_test_true, y_test_proba, k=TOP_K)
except Exception:
    test_topk_sklearn = None

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_acc:.4f}")
if test_topk_sklearn:
    print(f"Test Top-{TOP_K} Accuracy: {test_topk_sklearn:.4f}")

# === Save test metrics ===
with open(metrics_path, 'a') as f:
    f.write("\n=== Test Set Metrics ===\n")
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")
    f.write(f"Test Macro F1: {test_macro_f1}\n")
    f.write(f"Test Weighted F1: {test_weighted_f1}\n")
    f.write(f"Test Balanced Accuracy: {test_balanced_acc}\n")
    if test_topk_sklearn:
        f.write(f"Test Top-{TOP_K} Accuracy: {test_topk_sklearn}\n")


# === Plots ===
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend(); plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend(); plt.title("Loss")
plt.tight_layout()
plt.savefig(os.path.join(run_dir, "metrics.png"))

# === Clean up temporary dataset ===
delete_sampled_dataset(sampled_path)

print(f"Run completed and saved to {run_dir}")
