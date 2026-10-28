"""
CLI Test Script for Deepfake Face Detection Model.

Loads canonical model and runs inference on an input image or test folder,
printing structured prediction, confidence, probability distribution, and heuristics.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# Ensure backend directory is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import ml_pipeline

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("test_model")


def test_image(image_path: Path, model_file: Path, scaler_file: Path):
    if not image_path.exists():
        logger.error("Test image not found: %s", image_path)
        sys.exit(1)

    model, scaler, info = ml_pipeline.validate_and_load_model(model_file, scaler_file)
    if not info["loaded"]:
        logger.warning("Running in heuristic fallback mode: %s", info["error"])

    result = ml_pipeline.predict_image(image_path, model, scaler, allow_heuristic_fallback=True)

    print("\n" + "=" * 50)
    print("        DEEPFAKE DETECTION INFERENCE RESULT")
    print("=" * 50)
    print(f"Target Image:  {image_path.name}")
    print(f"Prediction:    {result.prediction} Face")
    print(f"Confidence:    {result.confidence:.1f}%")
    print(f"Probabilities: Real: {result.probabilities.get('Real', 0.0):.4f} | Fake: {result.probabilities.get('Fake', 0.0):.4f}")
    print(f"Inference Mode:{'Trained Linear SVM' if result.model_used else 'Heuristic Fallback'}")
    print("\nDiagnostic Reasons:")
    for r in result.analysis_reasons:
        print(f"  - #{r}")
    print("\nHeuristic Metrics:")
    print(f"  - Sharpness (Laplacian Var): {result.heuristics.laplacian_variance:.2f}")
    print(f"  - ORB Keypoint Count:        {result.heuristics.keypoint_count}")
    print(f"  - Histogram Std Dev:         {result.heuristics.hist_std:.2f}")
    print(f"  - Frontal Faces Detected:    {result.heuristics.faces_detected}")
    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Test Deepfake Model on an Image")
    parser.add_argument("image", nargs="?", default=None, help="Path to image file to test")
    parser.add_argument("--models-dir", default=str(CURRENT_DIR / "models"), help="Path to models directory")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    model_file = models_dir / "svm_model.pkl"
    scaler_file = models_dir / "svm_scaler.pkl"

    image_path = None
    if args.image:
        image_path = Path(args.image)
    else:
        # Auto-pick a sample image from Dataset/Test
        sample_candidates = [
            CURRENT_DIR.parent / "Dataset" / "Test" / "Real" / "real_100.jpg",
            CURRENT_DIR.parent / "Dataset" / "Test" / "Fake" / "fake_100.jpg",
        ]
        for c in sample_candidates:
            if c.exists():
                image_path = c
                break

    if not image_path:
        logger.error("No test image specified and default test image not found.")
        sys.exit(1)

    test_image(image_path, model_file, scaler_file)


if __name__ == "__main__":
    main()
