"""
==========================================================================
PET BREED CLASSIFIER - TRAINING SCRIPT
==========================================================================
Dataset: Oxford-IIIT Pet (37 classes - 25 dog breeds + 12 cat breeds)
Model: EfficientNetV2B0 with Transfer Learning
Hardware: Optimized for RTX 4050 Laptop GPU (6GB VRAM)

HOW TO RUN:
    python 01_train_model.py

EXPECTED TIME: ~30-45 minutes on RTX 4050
EXPECTED ACCURACY: 90-95%
==========================================================================
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================================================
# STEP 1: GPU SETUP - Prevents VRAM crashes on laptop GPUs
# ==========================================================================
print("=" * 70)
print("CHECKING GPU...")
print("=" * 70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth so TF doesn't grab all VRAM at once (CRITICAL for 4050)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU detected: {gpus[0]}")
        print("✅ Memory growth enabled (prevents VRAM crashes)")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("⚠️ NO GPU DETECTED - training will be VERY slow on CPU")
    print("⚠️ Make sure CUDA + cuDNN are installed correctly")

# ==========================================================================
# STEP 2: CONFIGURATION
# ==========================================================================
IMG_SIZE = 224          # EfficientNetV2B0 expects 224x224 images
BATCH_SIZE = 16         # Small batch for 6GB VRAM (safe for 4050)
NUM_CLASSES = 37        # Oxford-IIIT Pet has 37 breeds
EPOCHS_PHASE1 = 10      # Train only the head first
EPOCHS_PHASE2 = 10      # Fine-tune entire model
LEARNING_RATE_1 = 1e-3  # Higher LR for head training
LEARNING_RATE_2 = 1e-5  # Very low LR for fine-tuning (don't destroy pretrained weights)

# Create folders for outputs
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ==========================================================================
# STEP 3: LOAD DATASET (downloads automatically first time, ~800MB)
# ==========================================================================
print("\n" + "=" * 70)
print("LOADING OXFORD-IIIT PET DATASET...")
print("(First run will download ~800MB - may take 5-10 minutes)")
print("=" * 70)

(train_ds, test_ds), info = tfds.load(
    'oxford_iiit_pet',
    split=['train', 'test'],
    as_supervised=True,  # Returns (image, label) tuples
    with_info=True,
)

class_names = info.features['label'].names
print(f"✅ Loaded {info.splits['train'].num_examples} training images")
print(f"✅ Loaded {info.splits['test'].num_examples} test images")
print(f"✅ Number of classes: {len(class_names)}")

# ==========================================================================
# STEP 4: PREPROCESSING + DATA AUGMENTATION
# ==========================================================================
def preprocess(image, label):
    """Resize images to 224x224 (EfficientNetV2B0 input size)"""
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label

# Data augmentation layers (applied only during training)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name="data_augmentation")

# Build training pipeline
train_ds = (train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

test_ds = (test_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

# ==========================================================================
# STEP 5: BUILD THE MODEL
# ==========================================================================
print("\n" + "=" * 70)
print("BUILDING MODEL: EfficientNetV2B0 + Custom Classifier Head")
print("=" * 70)

# Load pretrained EfficientNetV2B0 (without its original classifier head)
base_model = keras.applications.EfficientNetV2B0(
    include_top=False,           # Remove ImageNet classifier
    weights='imagenet',          # Use ImageNet pretrained weights
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# FREEZE the base model initially (we'll train only the head first)
base_model.trainable = False

# Build the full model
inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)                    # Data augmentation
x = base_model(x, training=False)                # Feature extraction
x = layers.GlobalAveragePooling2D()(x)           # Pooling
x = layers.Dropout(0.3)(x)                       # Regularization
x = layers.Dense(128, activation='relu')(x)      # Hidden layer
x = layers.Dropout(0.2)(x)                       # More regularization
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)  # Classifier

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_1),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================================================
# STEP 6: PHASE 1 TRAINING - Train only the classifier head
# ==========================================================================
print("\n" + "=" * 70)
print(f"PHASE 1: Training classifier head ({EPOCHS_PHASE1} epochs)")
print("Base model frozen - only new layers learn")
print("=" * 70)

# Callbacks: save best model + stop if no improvement
callbacks_phase1 = [
    keras.callbacks.ModelCheckpoint(
        'outputs/models/best_phase1.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history_phase1 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_PHASE1,
    callbacks=callbacks_phase1,
    verbose=1
)

# ==========================================================================
# STEP 7: PHASE 2 TRAINING - Fine-tune the entire model
# ==========================================================================
print("\n" + "=" * 70)
print(f"PHASE 2: Fine-tuning entire model ({EPOCHS_PHASE2} epochs)")
print("Unfreezing base model - ALL layers now train with very low LR")
print("=" * 70)

# Unfreeze the base model
base_model.trainable = True

# Recompile with MUCH lower learning rate (critical — prevents destroying pretrained weights)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE_2),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    keras.callbacks.ModelCheckpoint(
        'outputs/models/best_final.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

history_phase2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_PHASE2,
    callbacks=callbacks_phase2,
    verbose=1
)

# ==========================================================================
# STEP 8: SAVE FINAL MODEL + CLASS NAMES
# ==========================================================================
model.save('outputs/models/final_model.keras')
print("\n✅ Final model saved to: outputs/models/final_model.keras")

# Save class names for later use
with open('outputs/class_names.txt', 'w') as f:
    for name in class_names:
        f.write(name + '\n')
print("✅ Class names saved to: outputs/class_names.txt")

# ==========================================================================
# STEP 9: PLOT TRAINING HISTORY (for thesis figures)
# ==========================================================================
def plot_history(hist1, hist2, save_path):
    """Combine both training phases into one plot"""
    acc = hist1.history['accuracy'] + hist2.history['accuracy']
    val_acc = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
    loss = hist1.history['loss'] + hist2.history['loss']
    val_loss = hist1.history['val_loss'] + hist2.history['val_loss']
    phase1_end = len(hist1.history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(acc, label='Training Accuracy', linewidth=2)
    ax1.plot(val_acc, label='Validation Accuracy', linewidth=2)
    ax1.axvline(x=phase1_end - 0.5, color='red', linestyle='--', label='Fine-tuning starts')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy Over Training')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(loss, label='Training Loss', linewidth=2)
    ax2.plot(val_loss, label='Validation Loss', linewidth=2)
    ax2.axvline(x=phase1_end - 0.5, color='red', linestyle='--', label='Fine-tuning starts')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss Over Training')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to: {save_path}")

plot_history(history_phase1, history_phase2, 'outputs/figures/training_curves.png')

# ==========================================================================
# STEP 10: FINAL EVALUATION
# ==========================================================================
print("\n" + "=" * 70)
print("FINAL EVALUATION ON TEST SET")
print("=" * 70)

test_loss, test_acc = model.evaluate(test_ds, verbose=1)
print(f"\n🎯 FINAL TEST ACCURACY: {test_acc * 100:.2f}%")
print(f"🎯 FINAL TEST LOSS: {test_loss:.4f}")

# Save results summary
with open('outputs/results_summary.txt', 'w') as f:
    f.write(f"Model: EfficientNetV2B0 (Transfer Learning)\n")
    f.write(f"Dataset: Oxford-IIIT Pet (37 classes)\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Phase 1 Epochs: {EPOCHS_PHASE1}\n")
    f.write(f"Phase 2 Epochs: {EPOCHS_PHASE2}\n")
    f.write(f"\nFINAL TEST ACCURACY: {test_acc * 100:.2f}%\n")
    f.write(f"FINAL TEST LOSS: {test_loss:.4f}\n")
print("✅ Results summary saved to: outputs/results_summary.txt")

print("\n" + "=" * 70)
print("🎉 TRAINING COMPLETE!")
print("Next step: Run 02_evaluate_model.py for confusion matrix + detailed metrics")
print("=" * 70)
