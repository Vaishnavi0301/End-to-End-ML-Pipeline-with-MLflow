# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle
import json
import os


def preprocess(data_path, sampling_strategy="smote"):
    """
    Load and preprocess the credit card fraud dataset.

    BUG FIXES vs original:
      1. Train/test split happens BEFORE scaling → no data leakage
      2. Two separate scalers (amount_scaler, time_scaler) → no overwrite bug
      3. Scalers fitted only on X_train → correct inference behaviour

    sampling_strategy options:
        "smote"        — oversample minority class with SMOTE
        "class_weight" — no resampling, handle imbalance in model
        "none"         — raw imbalanced data (baseline)

    Returns: X_train, X_test, y_train, y_test
    """

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.4f}%)")

    # ── Feature Setup ────────────────────────────────────────────────────────
    # V1–V28 are PCA-transformed already.
    # Amount and Time are raw — they need scaling.
    # We keep them raw here and scale AFTER splitting to avoid leakage.

    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
    X = df[feature_cols].values          # shape: (n, 30)
    y = df['Class'].values

    print(f"Features used: {len(feature_cols)}")

    # ── Train/Test Split FIRST ───────────────────────────────────────────────
    # stratify=y is critical — without it you risk 0 fraud in test set
    # Split BEFORE scaling — fitting scaler on full df was leaking test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(
        f"\nTrain size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")
    print(f"Train fraud: {sum(y_train == 1)} | Test fraud: {sum(y_test == 1)}")

    # ── Scale Amount (col 28) and Time (col 29) ──────────────────────────────
    # FIX: two separate scalers — original code used one scaler fit_transform
    # twice, which overwrites the first fit. The saved scaler only knew Time.
    # FIX: fit only on X_train, then transform X_test → no leakage

    amount_scaler = StandardScaler()
    time_scaler = StandardScaler()

    X_train[:, 28] = amount_scaler.fit_transform(
        X_train[:, 28].reshape(-1, 1)).ravel()
    X_test[:, 28] = amount_scaler.transform(
        X_test[:, 28].reshape(-1, 1)).ravel()

    X_train[:, 29] = time_scaler.fit_transform(
        X_train[:, 29].reshape(-1, 1)).ravel()
    X_test[:, 29] = time_scaler.transform(
        X_test[:, 29].reshape(-1, 1)).ravel()

    # Rename last two columns conceptually — they are now scaled
    # (callers can rely on cols 28/29 being Amount_scaled, Time_scaled)

    # ── Save Both Scalers ────────────────────────────────────────────────────
    # API loads this dict at startup for inference.
    # Original: single scaler overwritten → wrong Amount scaling at inference.
    scalers = {
        "amount": amount_scaler,
        "time":   time_scaler
    }
    scalers_path = "scalers.pkl"
    with open(scalers_path, "wb") as f:
        pickle.dump(scalers, f)
    print(f"Scalers saved to: {scalers_path}")

    # ── Imbalance Handling ───────────────────────────────────────────────────
    if sampling_strategy == "smote":
        print("\nApplying SMOTE...")
        sm = SMOTE(random_state=42, sampling_strategy=0.1)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        print(
            f"After SMOTE — Fraud: {sum(y_train == 1):,} | Legit: {sum(y_train == 0):,}")

    elif sampling_strategy == "class_weight":
        print("\nUsing class_weight balancing (no resampling)")

    elif sampling_strategy == "none":
        print("\nNo imbalance handling — baseline experiment")

    return X_train, X_test, y_train, y_test


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(
        __file__), "..", "data", "creditcard.csv")

    print("=" * 55)
    print("TEST 1: SMOTE")
    print("=" * 55)
    X_train, X_test, y_train, y_test = preprocess(
        data_path, sampling_strategy="smote")
    print(f"Final X_train shape: {X_train.shape}")
    print(
        f"Final y_train fraud ratio: {sum(y_train == 1)/len(y_train)*100:.2f}%")

    print("\n" + "=" * 55)
    print("TEST 2: No Sampling (baseline)")
    print("=" * 55)
    X_train2, X_test2, y_train2, y_test2 = preprocess(
        data_path, sampling_strategy="none")
    print(f"Final X_train shape: {X_train2.shape}")
    print(
        f"Final y_train fraud ratio: {sum(y_train2 == 1)/len(y_train2)*100:.2f}%")

    print("\n✅ preprocess.py working correctly")
