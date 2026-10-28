"""
Legacy Model Training Script Reference (Standardized).
Demonstrates training of the 4,096-feature Grayscale SVM model using the canonical pipeline.
"""

import os
from pathlib import Path
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    from backend import ml_pipeline
except ImportError:
    import ml_pipeline

BASE_DIR = Path(__file__).resolve().parent

def preprocess_image(image_path):
    """Preprocess image using canonical 4,096-feature grayscale pipeline."""
    return ml_pipeline.extract_canonical_features(image_path)

def load_data(dataset_path, max_per_class=1000):
    dataset_path = Path(dataset_path)
    X, y = [], []
    for label, category in enumerate(["Real", "Fake"]):
        category_path = dataset_path / category
        if not category_path.exists():
            continue
        files = [f for f in category_path.iterdir() if f.suffix.lower() in [".jpg", ".png", ".jpeg", ".webp"]][:max_per_class]
        for img_file in files:
            features = preprocess_image(img_file)
            if features is not None:
                X.append(features)
                y.append(label)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    dataset_candidates = [
        BASE_DIR.parent / "Dataset" / "Train",
        BASE_DIR / "Dataset" / "Train",
        Path("Dataset/Train"),
    ]
    dataset_path = next((p for p in dataset_candidates if p.exists()), None)
    if not dataset_path:
        print("❌ Dataset not found.")
        exit(1)

    print(f"Loading data from {dataset_path}...")
    X, y = load_data(dataset_path, max_per_class=1000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Training SVM model...")
    svm_model = SVC(kernel="linear", probability=True, random_state=42)
    svm_model.fit(X_train, y_train)

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(svm_model, models_dir / "svm_model.pkl")
    joblib.dump(scaler, models_dir / "svm_scaler.pkl")
    print(f"✅ Canonical model and scaler saved to {models_dir}")
