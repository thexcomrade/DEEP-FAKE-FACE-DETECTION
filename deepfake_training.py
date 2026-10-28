"""
Deepfake Training Module.

Uses canonical 4,096-pixel feature extraction to train and persist
the linear SVM model and scaler.
"""

import os
from pathlib import Path
import sys
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ml_pipeline


def train(dataset_root=None, max_samples=2000):
    candidates = [
        Path(dataset_root) if dataset_root else None,
        BASE_DIR / "Dataset",
        BACKEND_DIR / "Dataset",
    ]
    data_dir = next((c for c in candidates if c and (c / "Train").exists()), None)
    if not data_dir:
        raise FileNotFoundError("Could not find Dataset directory with 'Train' subfolder.")

    train_dir = data_dir / "Train"
    X, y = [], []

    for label_val, category in [(ml_pipeline.CLASS_REAL, "Real"), (ml_pipeline.CLASS_FAKE, "Fake")]:
        cat_dir = train_dir / category
        files = [f for f in cat_dir.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]][:max_samples]
        print(f"Loading {len(files)} {category} images...")
        for img_path in files:
            feat = ml_pipeline.extract_canonical_features(img_path)
            if feat is not None:
                X.append(feat)
                y.append(label_val)

    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print(f"Training Linear SVC on {len(X_train)} samples...")
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train, y_train)

    models_dir = BACKEND_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, models_dir / "svm_model.pkl")
    joblib.dump(scaler, models_dir / "svm_scaler.pkl")
    print(f"✅ Canonical model and scaler saved to {models_dir}")


if __name__ == "__main__":
    train()
