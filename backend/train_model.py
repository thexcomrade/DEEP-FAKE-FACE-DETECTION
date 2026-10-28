"""
Canonical Training and Evaluation Script for Deepfake Detection Model.

Trains and evaluates a Linear Support Vector Classifier (SVC) using 4,096-dimensional
grayscale facial pixel representations extracted by the canonical ML pipeline.
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Add backend directory to sys.path if needed
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import ml_pipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("train_model")


def find_dataset_dir() -> Optional[Path]:
    """Auto-discover dataset directory."""
    candidates = [
        CURRENT_DIR.parent / "Dataset",
        CURRENT_DIR / "Dataset",
        Path("Dataset"),
        Path(r"D:\DEEPFAKE\Dataset"),
    ]
    for c in candidates:
        if c.exists() and (c / "Train").exists():
            return c
    return None


def load_dataset_split(split_dir: Path, max_per_class: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load images from a split directory containing 'Real' and 'Fake' subdirectories.
    Extracts 4,096 canonical features per image.
    """
    X, y, filenames = [], [], []

    for label_val, category in [(ml_pipeline.CLASS_REAL, "Real"), (ml_pipeline.CLASS_FAKE, "Fake")]:
        cat_dir = split_dir / category
        if not cat_dir.exists():
            logger.warning("Category directory missing: %s", cat_dir)
            continue

        all_files = [f for f in cat_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]]
        if max_per_class:
            all_files = all_files[:max_per_class]

        logger.info("Extracting features from %d %s images in %s...", len(all_files), category, cat_dir.name)
        for idx, file_path in enumerate(all_files):
            feat = ml_pipeline.extract_canonical_features(file_path)
            if feat is not None:
                X.append(feat)
                y.append(label_val)
                filenames.append(str(file_path))

            if (idx + 1) % 500 == 0:
                logger.info("  Processed %d / %d %s images", idx + 1, len(all_files), category)

    return np.array(X), np.array(y), filenames


def evaluate_model(model: Any, scaler: Any, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """Compute and display rigorous classification metrics."""
    logger.info("Evaluating model on %d test images...", len(y_test))
    t0 = time.time()
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, ml_pipeline.CLASS_FAKE]
    infer_time = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 55)
    print("           MODEL EVALUATION REPORT")
    print("=" * 55)
    print(f"Total Test Samples: {len(y_test)}")
    print(f"Accuracy:           {acc:.4f} ({acc * 100:.2f}%)")
    print(f"Precision (Fake):   {prec:.4f}")
    print(f"Recall (Fake):      {rec:.4f}")
    print(f"F1-Score:           {f1:.4f}")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Inference Latency:  {infer_time / len(y_test) * 1000:.2f} ms/sample")
    print("\nConfusion Matrix:")
    print("                 Predicted Real (0)   Predicted Fake (1)")
    print(f"Actual Real (0):       {cm[0][0]:<18} {cm[0][1]}")
    print(f"Actual Fake (1):       {cm[1][0]:<18} {cm[1][1]}")
    print("\nDetailed Classification Breakdown:")
    print(classification_report(y_test, y_pred, target_names=["Real (0)", "Fake (1)"]))
    print("=" * 55 + "\n")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }


def train_pipeline(
    dataset_dir: Path,
    output_models_dir: Path,
    samples_per_class: int = 2400,
) -> Tuple[Any, Any]:
    """Train canonical linear SVM model and scaler."""
    train_dir = dataset_dir / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")

    logger.info("Loading training data (up to %d per class)...", samples_per_class)
    X, y, _ = load_dataset_split(train_dir, max_per_class=samples_per_class)

    if len(X) < 100:
        raise ValueError(f"Insufficient training samples loaded: {len(X)}")

    logger.info("Fitting StandardScaler on %d samples with %d features...", len(X), X.shape[1])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)

    logger.info("Training Linear SVC (probability=True)...")
    t0 = time.time()
    model = SVC(kernel="linear", probability=True, random_state=42)
    model.fit(X_train_scaled, y)
    logger.info("Model fitting complete in %.2f seconds.", time.time() - t0)

    output_models_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_models_dir / "svm_model.pkl"
    scaler_path = output_models_dir / "svm_scaler.pkl"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info("Saved canonical artifacts to:\n  - %s\n  - %s", model_path, scaler_path)

    return model, scaler


def main():
    parser = argparse.ArgumentParser(description="Deepfake Detection Training & Evaluation CLI")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing canonical model on Test split")
    parser.add_argument("--train", action="store_true", help="Train model on Dataset/Train")
    parser.add_argument("--samples", type=int, default=500, help="Samples per class for test evaluation or training")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Path to Dataset root directory")
    parser.add_argument("--models-dir", type=str, default=None, help="Path to models artifact directory")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_dir) if args.dataset_dir else find_dataset_dir()
    if not dataset_path or not dataset_path.exists():
        logger.error("Dataset directory not found. Please provide --dataset-dir.")
        sys.exit(1)

    models_path = Path(args.models_dir) if args.models_dir else (CURRENT_DIR / "models")

    # Default action if none specified is evaluate
    if not args.train and not args.evaluate:
        args.evaluate = True

    if args.train:
        logger.info("Starting model training pipeline...")
        model, scaler = train_pipeline(dataset_path, models_path, samples_per_class=args.samples)
    else:
        model_file = models_path / "svm_model.pkl"
        scaler_file = models_path / "svm_scaler.pkl"
        model, scaler, info = ml_pipeline.validate_and_load_model(model_file, scaler_file)
        if not info["loaded"]:
            logger.error("Cannot evaluate: %s", info["error"])
            sys.exit(1)

    if args.evaluate:
        test_dir = dataset_path / "Test"
        if not test_dir.exists():
            test_dir = dataset_path / "Validation"

        if test_dir.exists():
            X_test, y_test, _ = load_dataset_split(test_dir, max_per_class=args.samples)
            evaluate_model(model, scaler, X_test, y_test)
        else:
            logger.warning("No Test or Validation folder found at %s", dataset_path)


if __name__ == "__main__":
    main()
