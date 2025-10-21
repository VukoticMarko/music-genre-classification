import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os # Remove clogged warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import backend as K
import gc
# Clear any existing sessions and free memory
K.clear_session()
tf.compat.v1.reset_default_graph()
gc.collect()

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Optional: Used to limit GPU VRAM usage so we don't run out of memory while training.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limit to 4GB (adjust based on your needs)
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4 * 1024)]  # 4GB in MB
        )
    except RuntimeError as e:
        print(e)

from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


# Path to mel_images directory
data_dir = 'data/mel_images'
data_dir = '../music_preprocessing/data_opt/large_mel_images'

# Create data generators with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.15,  # 85% train, 15% validation
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # horizontal_flip=True,
    #seed=42
)


BATCH_SIZE = 4
INPUT_SHAPE = (224, 224)
CNN_INPUT = (224, 224, 1) # If gray 3rd parameter should be 1, else if RGB go with 3
EPOCHS = 100
MODEL_NAME = "model-FAST-opt-LARGE-idgpreprop.h5"
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
    GlobalAveragePooling2D,
    SeparableConv2D
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
    # SeparableConv2D(84, kernel_size=3, activation="relu",
    # depthwise_regularizer=REGULARLIZER,  
    # pointwise_regularizer=REGULARLIZER), # This is a lighter Conv2D layer for weaker GPU's
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(96, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER),
    BatchNormalization(),
    Conv2D(112, kernel_size=3, activation="relu", padding="same", kernel_regularizer=REGULARLIZER), 
    MaxPooling2D(pool_size=(2, 2)),
    GlobalAveragePooling2D(),
    Dense(20, activation="relu", kernel_regularizer=REGULARLIZER),
    Dense(100, activation="relu", kernel_regularizer=REGULARLIZER),
    Dropout(0.4),
    Dense(num_classes, activation='softmax', dtype='float32')
])

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks:
# Early Stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
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
# Enable gradient checkpointing (reduces memory at the cost of speed)
tf.config.optimizer.set_experimental_options({'gradient_checkpointing': True})


# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=[early_stopping, lr_scheduler]
)

# Evaluation
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    data_dir,
    target_size=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    color_mode='grayscale'
)

loss, accuracy = model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

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
    plt.show()

    plt.savefig('metrics-fast.png')

plot_history(history)

model.save(MODEL_NAME)