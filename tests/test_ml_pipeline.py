"""
Unit tests for Canonical Machine Learning Pipeline (backend/ml_pipeline.py).
"""

import os
from pathlib import Path
import unittest
import numpy as np
from PIL import Image

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ml_pipeline


class TestMLPipeline(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = BASE_DIR / "tests" / "fixtures"
        self.valid_jpg = self.fixtures_dir / "valid_test.jpg"
        self.valid_png = self.fixtures_dir / "valid_test.png"
        self.corrupted = self.fixtures_dir / "corrupted.jpg"
        self.models_dir = BACKEND_DIR / "models"
        self.model_file = self.models_dir / "svm_model.pkl"
        self.scaler_file = self.models_dir / "svm_scaler.pkl"

    def test_extract_canonical_features_dimensions(self):
        """Feature extraction must yield exactly 4,096 elements of float64."""
        feats = ml_pipeline.extract_canonical_features(self.valid_jpg)
        self.assertIsNotNone(feats)
        self.assertEqual(feats.shape, (4096,))
        self.assertEqual(feats.dtype, np.float64)

    def test_extract_canonical_features_from_pil(self):
        """Feature extraction must accept PIL Image instance."""
        with Image.open(self.valid_png) as img:
            feats = ml_pipeline.extract_canonical_features(img)
            self.assertIsNotNone(feats)
            self.assertEqual(feats.shape, (4096,))

    def test_extract_canonical_features_from_numpy(self):
        """Feature extraction must accept RGB and grayscale numpy arrays."""
        rgb_arr = np.ones((80, 80, 3), dtype=np.uint8) * 128
        feats = ml_pipeline.extract_canonical_features(rgb_arr)
        self.assertIsNotNone(feats)
        self.assertEqual(feats.shape, (4096,))

        gray_arr = np.ones((100, 100), dtype=np.uint8) * 64
        feats_gray = ml_pipeline.extract_canonical_features(gray_arr)
        self.assertIsNotNone(feats_gray)
        self.assertEqual(feats_gray.shape, (4096,))

    def test_extract_canonical_features_corrupted_image(self):
        """Corrupted image bytes should return None safely."""
        feats = ml_pipeline.extract_canonical_features(self.corrupted)
        self.assertIsNone(feats)

    def test_analyze_image_heuristics(self):
        """Heuristics must produce non-negative metrics and face detection."""
        heuristics = ml_pipeline.analyze_image_heuristics(self.valid_jpg)
        self.assertIsInstance(heuristics, ml_pipeline.HeuristicMetrics)
        self.assertGreaterEqual(heuristics.laplacian_variance, 0.0)
        self.assertGreaterEqual(heuristics.keypoint_count, 0)
        self.assertGreaterEqual(heuristics.hist_std, 0.0)

    def test_validate_and_load_model_success(self):
        """Validates that svm_model.pkl and svm_scaler.pkl load with 4,096 features."""
        if not self.model_file.exists() or not self.scaler_file.exists():
            self.skipTest("Model files not present on disk.")
        model, scaler, info = ml_pipeline.validate_and_load_model(self.model_file, self.scaler_file)
        self.assertTrue(info["loaded"])
        self.assertEqual(info["n_features_in"], 4096)
        self.assertEqual(info["classes"], [0, 1])
        self.assertIsNotNone(model)
        self.assertIsNotNone(scaler)

    def test_validate_and_load_model_missing_files(self):
        """Missing model file returns graceful error in info dict."""
        model, scaler, info = ml_pipeline.validate_and_load_model("nonexistent_model.pkl", "nonexistent_scaler.pkl")
        self.assertFalse(info["loaded"])
        self.assertIn("not found", info["error"])
        self.assertIsNone(model)
        self.assertIsNone(scaler)

    def test_predict_image_with_trained_model(self):
        """End-to-end prediction with trained model returns structured result."""
        if not self.model_file.exists() or not self.scaler_file.exists():
            self.skipTest("Model files not present on disk.")
        model, scaler, _ = ml_pipeline.validate_and_load_model(self.model_file, self.scaler_file)
        result = ml_pipeline.predict_image(self.valid_jpg, model, scaler)
        self.assertIn(result.prediction, ["Real", "Fake"])
        self.assertTrue(result.model_used)
        self.assertFalse(result.is_heuristic_fallback)
        self.assertGreaterEqual(result.confidence, 50.0)
        self.assertLessEqual(result.confidence, 100.0)
        self.assertIn("Real", result.probabilities)
        self.assertIn("Fake", result.probabilities)
        self.assertAlmostEqual(result.probabilities["Real"] + result.probabilities["Fake"], 1.0, places=2)

    def test_predict_image_heuristic_fallback_when_model_absent(self):
        """When model is absent and fallback allowed, returns clearly identified heuristic output."""
        result = ml_pipeline.predict_image(self.valid_jpg, model=None, scaler=None, allow_heuristic_fallback=True)
        self.assertIn(result.prediction, ["Real", "Fake"])
        self.assertFalse(result.model_used)
        self.assertTrue(result.is_heuristic_fallback)
        self.assertIn("heuristic", result.model_name.lower())

    def test_predict_image_error_when_model_absent_and_fallback_disabled(self):
        """When model is absent and fallback disabled, returns controlled error."""
        result = ml_pipeline.predict_image(self.valid_jpg, model=None, scaler=None, allow_heuristic_fallback=False)
        self.assertEqual(result.prediction, "Error")
        self.assertIsNotNone(result.error)

    def test_generate_saliency_map(self):
        """Generates Explainable AI saliency attribution heatmap from Linear SVM weights."""
        if not self.model_file.exists() or not self.scaler_file.exists():
            self.skipTest("Model files not present on disk.")
        model, scaler, _ = ml_pipeline.validate_and_load_model(self.model_file, self.scaler_file)
        tmp_map = BASE_DIR / "tests" / "fixtures" / "tmp_saliency.jpg"
        try:
            out_path = ml_pipeline.generate_saliency_map(self.valid_jpg, model, scaler, output_path=tmp_map)
            self.assertIsNotNone(out_path)
            self.assertTrue(Path(out_path).exists())
            with Image.open(out_path) as im:
                self.assertEqual(im.format, "JPEG")
        finally:
            if tmp_map.exists():
                tmp_map.unlink()


if __name__ == "__main__":
    unittest.main()
