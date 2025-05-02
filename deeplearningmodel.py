# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from keras import Input, Model, layers, utils, optimizers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve,
    precision_recall_curve, auc
)

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
FILE_NORMAL = './all_sequences.pkl'
FILE_TOR    = './all_sequences.pkl'

SEQUENCE_LENGTH = 5000
NUM_CLASSES     = 2
BATCH_SIZE      = 32
EPOCHS          = 100
TEST_SIZE       = 0.30
VAL_SIZE        = 0.50
LEARNING_RATE   = 1e-4
RANDOM_STATE    = 42

# ──────────────────────────────────────────────────────────────────────────────
# Data Utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_sequences(pickle_path):
    """Load a pickle of (id, sequence) items and return list of sequences."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return [item[1] for item in data]

def create_dataset(normal_path, tor_path, seq_len):
    """Loads data, applies labels, and pads/truncates sequences."""
    normal = load_sequences(normal_path)
    tor = load_sequences(tor_path)

    X_raw = normal + tor
    y_raw = [0] * len(normal) + [1] * len(tor)

    X = []
    for seq in X_raw:
        seq = [(float(ts), float(val)) for ts, val in seq]
        padded = seq + [(0.0, 0.0)] * (seq_len - len(seq)) if len(seq) < seq_len else seq[:seq_len]
        X.append(padded)

    return np.array(X, dtype=np.float32), np.array(y_raw, dtype=np.int32)

def split_and_encode(X, y, test_size, val_size, num_classes):
    """Shuffles, splits, and one-hot encodes dataset."""
    X, y = shuffle(X, y, random_state=RANDOM_STATE)
    y_cat = utils.to_categorical(y, num_classes)

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y_cat, test_size=test_size, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=val_size, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test

def augment_data(X, noise_factor=0.01):
    """Adds Gaussian noise to training data."""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

# ──────────────────────────────────────────────────────────────────────────────
# Model Definition
# ──────────────────────────────────────────────────────────────────────────────
def build_model(seq_len, num_classes):
    inputs = Input(shape=(seq_len, 2))

    x = layers.Conv1D(32, 4, padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 4, padding='same')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 4, padding='same')(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# ──────────────────────────────────────────────────────────────────────────────
# Main Training & Evaluation Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Load and prepare data
    X, y = create_dataset(FILE_NORMAL, FILE_TOR, SEQUENCE_LENGTH)
    X_train, X_val, X_test, y_train, y_val, y_test = split_and_encode(X, y, TEST_SIZE, VAL_SIZE, NUM_CLASSES)
    X_train = augment_data(X_train)
    X_val = augment_data(X_val)

    print(f"Shapes — X_train: {X_train.shape}, Y_train: {y_train.shape}")
    print(f"           X_val:   {X_val.shape},   Y_val:   {y_val.shape}")
    print(f"           X_test:  {X_test.shape},   Y_test:  {y_test.shape}")

    # Build and train model
    model = build_model(SEQUENCE_LENGTH, NUM_CLASSES)
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val)
    )

    # Evaluation
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print(f"\nConfusion Matrix:\n{cm}")

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    fpr = FP / (FP + TN) if (FP + TN) else 0

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"Test Accuracy: {acc:.4f}")

    roc_fpr, roc_tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    pr_prec, pr_rec, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(roc_fpr, roc_tpr)
    pr_auc = auc(pr_rec, pr_prec)

    summary_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall (TPR)', 'FPR', 'ROC AUC', 'PR AUC'],
        'Value': [acc, precision, recall, fpr, roc_auc, pr_auc]
    })
    print("\nKey Evaluation Metrics:")
    print(summary_df.to_string(index=False, float_format='%.4f'))

    # ──────────────────────────────────────────────────────────────────────────
    # Plots
    # ──────────────────────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(roc_fpr, roc_tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(pr_rec, pr_prec, label=f"AUC = {pr_auc:.4f}", color='green')
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Accuracy & Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title("Accuracy Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    main()
