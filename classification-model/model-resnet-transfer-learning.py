import os
import gc
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, top_k_accuracy_score,
    precision_recall_fscore_support, classification_report, confusion_matrix
)
from sklearn.model_selection import train_test_split

# -----------------------
# USER CONFIG
# -----------------------
DATA_DIR = '../music-preprocessing/data_opt/large_mel_images_rgb_high'
BATCH_SIZE = 16
INPUT_SIZE = (224, 224)
CNN_INPUT = (INPUT_SIZE[0], INPUT_SIZE[1], 3)
EPOCHS = 100
REGULARIZER = tf.keras.regularizers.L2(1e-4)
TOP_K = 3
RESNET_TRAINABLE = False
RUNS_DIR = "runs-resnet-transfer"
TEST_RATIO = 0.10
VAL_RATIO = 0.10
RANDOM_STATE = 42
# -----------------------

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Clear sessions/memory
keras.backend.clear_session()
tf.compat.v1.reset_default_graph()
gc.collect()

# Mixed precision (keep float32 if no GPU or small batches)
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('float32')
mixed_precision.set_global_policy(policy)

# Create run directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(RUNS_DIR, f"run_{timestamp}")
os.makedirs(run_dir, exist_ok=True)
print(f"\n🔹 Run directory: {run_dir}\n")

# --- Build file list and labels ---
records = []
for class_name in sorted(os.listdir(DATA_DIR)):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue
    for fname in sorted(os.listdir(class_dir)):
        # skip hidden files
        if fname.startswith('.'):
            continue
        records.append({
            'filepath': os.path.join(class_dir, fname),
            'label': class_name
        })

if len(records) == 0:
    raise RuntimeError(f"No images found under {DATA_DIR}")

df = pd.DataFrame(records)

# Stratify split: first train vs temp (80/20), then temp -> val/test (50/50 of temp)
train_df, temp_df = train_test_split(
    df, test_size=(TEST_RATIO + VAL_RATIO), stratify=df['label'], random_state=RANDOM_STATE
)
val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
val_df, test_df = train_test_split(
    temp_df, test_size=(TEST_RATIO / (VAL_RATIO + TEST_RATIO)), stratify=temp_df['label'], random_state=RANDOM_STATE
)

print(f"Samples: total={len(df)} train={len(train_df)} val={len(val_df)} test={len(test_df)}")

# --- ImageDataGenerators ---
# NOTE: rescale=1./255 is kept in generators and apply preprocess_input(x * 255.0) inside the model
train_datagen = ImageDataGenerator(
    rescale=1./255,
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label',
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='filepath',
    y_col='label',
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filepath',
    y_col='label',
    target_size=INPUT_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
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
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )
    base.trainable = trainable

    # Freeze BN layers for fine-tune later
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    inp = keras.Input(shape=input_shape)

    x = data_augmentation(inp)
    # Generators produce images in [0,1] because of rescale=1./255 -> multiply back to 0..255
    x = tf.keras.applications.resnet.preprocess_input(x * 255.0)

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
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
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
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)

# Helper: save history to txt and csv
def save_history(history, filename_txt, filename_csv):
    hist = history.history
    keys = list(hist.keys())
    df_hist = pd.DataFrame(hist)
    df_hist.index = df_hist.index + 1  # Epoch numbers starting at 1
    df_hist.index.name = 'epoch'
    df_hist.to_csv(filename_csv)

    with open(filename_txt, 'w') as f:
        f.write('epoch,' + ','.join(keys) + '\n')
        for epoch in range(len(df_hist)):
            row = df_hist.iloc[epoch]
            f.write(f"{epoch+1}," + ",".join([f"{row[k]:.6f}" for k in keys]) + "\n")

# --- Train ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler],
    verbose=1
)

# Save initial training history
save_history(history,
             os.path.join(run_dir, 'training_metrics.txt'),
             os.path.join(run_dir, 'training_metrics.csv'))

# Plot train/val accuracy & loss for initial training
def plot_history_accuracy_loss(history, out_prefix):
    hist = history.history
    epochs = range(1, len(next(iter(hist.values()))) + 1)

    # Accuracy plot
    acc_keys = [k for k in hist.keys() if 'accuracy' in k and 'val' not in k]
    val_acc_keys = [k for k in hist.keys() if 'val_accuracy' in k or 'val_' + next(iter(acc_keys), 'accuracy') in k]
    plt.figure(figsize=(8, 4))
    if 'accuracy' in hist or any('accuracy' in k for k in hist.keys()):
        train_acc = hist.get('accuracy') or hist.get(next((k for k in hist.keys() if 'accuracy' in k and not k.startswith('val_')), None))
        val_acc = hist.get('val_accuracy') or hist.get(next((k for k in hist.keys() if k.startswith('val_') and 'accuracy' in k), None))
        if train_acc is not None:
            plt.plot(epochs, train_acc, label='Train')
        if val_acc is not None:
            plt.plot(epochs, val_acc, label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + '_accuracy.png')
        plt.close()

    # Loss plot
    plt.figure(figsize=(8, 4))
    train_loss = hist.get('loss')
    val_loss = hist.get('val_loss')
    if train_loss is not None:
        plt.plot(epochs, train_loss, label='Train')
    if val_loss is not None:
        plt.plot(epochs, val_loss, label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + '_loss.png')
    plt.close()

plot_history_accuracy_loss(history, os.path.join(run_dir, 'initial_train'))

# --- Fine-tune: unfreeze some top layers of ResNet50 and train with low LR ---
# Unfreeze top N layers of the backbone
N_UNFREEZE = 50
for layer in model.layers[-N_UNFREEZE:]:
    if 'conv' in layer.name or isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = True

# BatchNorm layers remain frozen
for l in model.layers:
    if isinstance(l, tf.keras.layers.BatchNormalization):
        l.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=TOP_K)]
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=150,
    callbacks=[early_stopping, lr_scheduler]
)

# Save finetune history
save_history(history_finetune,
             os.path.join(run_dir, 'finetune_metrics.txt'),
             os.path.join(run_dir, 'finetune_metrics.csv'))

# Plot finetune history
plot_history_accuracy_loss(history_finetune, os.path.join(run_dir, 'finetune'))

# --- Metrics on test set ---
y_proba = model.predict(test_generator, verbose=1)
y_pred = np.argmax(y_proba, axis=1)
y_true = test_generator.classes

class_indices = train_generator.class_indices
inv_map = {v: k for k, v in class_indices.items()}

present_labels = np.unique(y_true)
class_names_filtered = [inv_map[i] for i in present_labels]