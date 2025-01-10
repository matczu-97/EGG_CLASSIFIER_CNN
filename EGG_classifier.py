import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras import Sequential
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, BatchNormalization, Flatten

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)


def load_and_preprocess_data(adhd_folder, control_folder, sequence_length=12000):
    """
    Load and preprocess EEG data from both ADHD and control folders.
    """
    data_list = []
    labels = []

    def process_file(file_path):
        df = pd.read_csv(file_path)
        data = df.values

        if len(data) < sequence_length:
            padding = np.zeros((sequence_length - len(data), data.shape[1]))
            data = np.vstack([data, padding])
        elif len(data) > sequence_length:
            data = data[:sequence_length]
        return data

    # Load ADHD data
    adhd_files = list(Path(adhd_folder).glob('*.csv'))
    print(f"Found {len(adhd_files)} ADHD files")

    for file_path in adhd_files:
        try:
            data = process_file(file_path)
            data_list.append(data)
            labels.append(1)
        except Exception as e:
            print(f"Error processing ADHD file {file_path}: {e}")
            continue

    # Load control data
    control_files = list(Path(control_folder).glob('*.csv'))
    print(f"Found {len(control_files)} control files")

    for file_path in control_files:
        try:
            data = process_file(file_path)
            data_list.append(data)
            labels.append(0)
        except Exception as e:
            print(f"Error processing control file {file_path}: {e}")
            continue

    X = np.array(data_list)
    y = np.array(labels)

    print(f"Final dataset shape: {X.shape}")
    return X, y


def create_cnn_model(input_shape):
    """
    Create a 1D CNN model for EEG classification with improved architecture.
    """
    model = Sequential([
        # First convolutional block
        Conv1D(filters=64, kernel_size=10, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=2, padding='same'),
        Dropout(0.3),

        # Second convolutional block
        Conv1D(filters=128, kernel_size=8, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=2, padding='same'),
        Dropout(0.3),

        # Third convolutional block
        Conv1D(filters=256, kernel_size=6, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=2, padding='same'),
        Dropout(0.3),

        # Fourth convolutional block
        Conv1D(filters=512, kernel_size=4, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4, strides=2, padding='same'),

        # Flatten and Dense layers
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Learning rate schedule
    initial_learning_rate = 0.001

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
    return model


def plot_confusion_matrix(cm, title, filename):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.text(-0.5, -0.5, 'Control', ha='right')
    plt.text(-0.5, 1.5, 'ADHD', ha='right')
    plt.text(0.5, 2.3, 'Control', ha='center')
    plt.text(1.5, 2.3, 'ADHD', ha='center')

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved confusion matrix to {filepath}")


def plot_metrics(metrics_dict, model_name):
    """Plot and save performance metrics."""
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy']
    values = [metrics_dict[m.lower()] for m in metrics]

    bars = plt.bar(metrics, values, color=['#9b59b6'])
    plt.ylim(0, 1.0)
    plt.title(f'{model_name} Performance Metrics')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.3f}',
                 ha='center', va='bottom')

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, f'{model_name.lower()}_metrics.png')
    plt.savefig(filepath)
    plt.close()
    print(f"Saved metrics plot to {filepath}")


def plot_training_history(history, filename):
    """Plot and save training history."""
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    filepath = os.path.join(PLOT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"Saved training history to {filepath}")


def main():
    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Get absolute paths for data folders
    adhd_folder = os.path.join(SCRIPT_DIR, 'adhd')
    control_folder = os.path.join(SCRIPT_DIR, 'control')

    print(f"\nLooking for data in:")
    print(f"ADHD folder: {adhd_folder}")
    print(f"Control folder: {control_folder}")

    # Verify folders exist
    if not os.path.exists(adhd_folder):
        raise FileNotFoundError(f"ADHD folder not found: {adhd_folder}")
    if not os.path.exists(control_folder):
        raise FileNotFoundError(f"Control folder not found: {control_folder}")

    # Load and split data
    print("\nLoading and preprocessing data...")
    X, y = load_and_preprocess_data(adhd_folder, control_folder)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    # Train and evaluate CNN
    print("\nTraining and evaluating CNN model...")
    cnn_model = create_cnn_model((X.shape[1], X.shape[2]))

    # Create validation split
    val_size = int(0.2 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_cnn = X_train[:-val_size]
    y_train_cnn = y_train[:-val_size]

    # Train CNN
    history = cnn_model.fit(
        X_train_cnn, y_train_cnn,
        epochs=100,
        batch_size=16,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Plot training history
    plot_training_history(history, 'cnn_training_history.png')

    # Evaluate CNN
    y_pred_proba = cnn_model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    # Calculate CNN metrics

    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()


    cnn_metrics = {
        'confusion_matrix': cm,
        'accuracy': accuracy
    }

    plot_confusion_matrix(cm, 'CNN Confusion Matrix', 'cnn_confusion_matrix.png')
    plot_metrics(cnn_metrics, 'CNN')

    print("\nCNN Results:")
    print(f"Accuracy: {accuracy:.4f}")



if __name__ == "__main__":
    main()