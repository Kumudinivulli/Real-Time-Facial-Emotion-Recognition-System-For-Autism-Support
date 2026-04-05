"""
ASD Emotion Recognition System - CNN + BiLSTM Model Training
============================================================
This script trains a hybrid CNN-BiLSTM model on the RAF-DB dataset
for facial emotion recognition in individuals with Autism Spectrum Disorder.

Requirements:
    pip install tensorflow keras numpy pandas matplotlib scikit-learn opencv-python pillow

Dataset:
    Download RAF-DB from: http://www.whdeng.cn/raf/model1.html
    Extract to: ./RAF-DB/

Usage:
    python model_training.py --data_dir ./RAF-DB --epochs 50 --batch_size 32
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# TensorFlow / Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout,
    Flatten, Dense, Reshape, Bidirectional, LSTM, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
EMOTION_LABELS = [
    'Surprise', 'Fear', 'Disgust', 'Happiness',
    'Sadness', 'Anger', 'Neutral'
]
IMG_SIZE = 40
NUM_CLASSES = 7


def build_cnn_bilstm_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Hybrid CNN-BiLSTM architecture:
    - CNN layers extract spatial features from facial images
    - BiLSTM layers capture sequential/contextual patterns in the feature maps
    - Dense layers perform final classification
    """
    inputs = Input(shape=input_shape, name='input_image')

    # ── CNN Feature Extractor ──
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 4
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # ── Reshape for BiLSTM ──
    # Flatten spatial dims into a sequence for the BiLSTM
    shape = x.shape
    x = Reshape((shape[1] * shape[2], shape[3]), name='reshape_to_seq')(x)

    # ── BiLSTM Layers ──
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3,
                           recurrent_dropout=0.2), name='bilstm_1')(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3,
                           recurrent_dropout=0.2), name='bilstm_2')(x)
    x = GlobalAveragePooling1D(name='global_avg_pool')(x)

    # ── Classifier ──
    x = Dense(256, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax', name='emotion_output')(x)

    model = Model(inputs=inputs, outputs=outputs, name='CNN_BiLSTM_Emotion')
    return model


def create_data_generators(data_dir, batch_size=32):
    """Create training and validation data generators with augmentation."""
    train_datagen = ImageDataGenerator(
         rescale=1.0 / 255,
         rotation_range=5,
         width_shift_range=0.05,
         height_shift_range=0.05,
         zoom_range=0.05,
         horizontal_flip=True,
         validation_split=0.2
    )

    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    # If train/test split exists in dataset
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True
        )
        val_datagen = ImageDataGenerator(rescale=1.0 / 255)
        val_gen = val_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False
        )
    else:
        # Use validation_split
        train_gen = train_datagen.flow_from_directory(
            data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=True,
            subset='training'
        )
        val_gen = train_datagen.flow_from_directory(
            data_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False,
            subset='validation'
        )

    return train_gen, val_gen


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training/validation accuracy and loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(model, val_gen, save_path='confusion_matrix.png'):
    """Generate and save confusion matrix."""
    predictions = model.predict(val_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix - Emotion Recognition', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Confusion matrix saved to {save_path}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=EMOTION_LABELS))


def train(args):
    """Main training function."""
    print("=" * 60)
    print("ASD Emotion Recognition - CNN+BiLSTM Training")
    print("=" * 60)

    # Build model
    model = build_cnn_bilstm_model()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Data generators
    train_gen, val_gen = create_data_generators(args.data_dir, args.batch_size)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_emotion_model.keras',
                        monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-6, verbose=1)
    ]

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Save final model
    model.save('emotion_model_final.keras')
    print("\nFinal model saved to emotion_model_final.keras")

    # Generate plots
    plot_training_history(history)
    plot_confusion_matrix(model, val_gen)

    # Save as TFLite for lightweight deployment
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open('emotion_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved to emotion_model.tflite")

    print("\n✅ Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN-BiLSTM Emotion Model')
    parser.add_argument('--data_dir', type=str, default='./dataset',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    args = parser.parse_args()
    train(args)
