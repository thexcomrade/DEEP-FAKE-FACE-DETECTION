"""
Regression tests covering historical defects discovered in the audit:
1. Feature vector dimensionality (4,354 vs 4,096 mismatch)
2. Jinja template UndefinedError for 'filename'
3. Background asset path resolution
4. Missing session result in /download route
5. Model pre-inference validation & graceful fallback
"""

import io
from pathlib import Path
import unittest
import numpy as np

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ml_pipeline
from app import create_app


class TestRegressions(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = BASE_DIR / "tests" / "fixtures"
        self.tmp_uploads = BASE_DIR / "tests" / "tmp_uploads"
        self.tmp_uploads.mkdir(parents=True, exist_ok=True)
        self.app = create_app({
            "TESTING": True,
            "UPLOAD_FOLDER": str(self.tmp_uploads),
            "SECRET_KEY": "regression-secret-987",
        })
        self.client = self.app.test_client()

    def tearDown(self):
        for f in self.tmp_uploads.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            self.tmp_uploads.rmdir()
        except OSError:
            pass

    def test_regression_feature_dimension_scaler_compatibility(self):
        """
        REGRESSION TEST FOR DEFECT ID: DEF-001
        Root cause: app.py appended blur, ORB keypoints, and histogram (yielding 4,354 features)
        while StandardScaler was trained on 4,096 features.
        Verification: Ensure canonical extractor produces exactly 4,096 features, and that
        svm_scaler.pkl transforms the vector without ValueError.
        """
        scaler_file = BACKEND_DIR / "models" / "svm_scaler.pkl"
        if not scaler_file.exists():
            self.skipTest("svm_scaler.pkl missing from disk.")

        import joblib
        scaler = joblib.load(scaler_file)
        self.assertEqual(scaler.n_features_in_, 4096)

        # Extract canonical features
        sample_img = self.fixtures_dir / "valid_test.jpg"
        features = ml_pipeline.extract_canonical_features(sample_img)
        self.assertEqual(len(features), 4096)
        self.assertNotEqual(len(features), 4354)

        # Transformation MUST succeed without raising ValueError
        X_scaled = scaler.transform(features.reshape(1, -1))
        self.assertEqual(X_scaled.shape, (1, 4096))

    def test_regression_template_undefined_filename(self):
        """
        REGRESSION TEST FOR DEFECT ID: DEF-002
        Root cause: result.html previously raised jinja2.exceptions.UndefinedError: 'filename' is undefined.
        Verification: Ensure /result renders completely when 'filename' and/or 'image_filename' are in context.
        """
        # Create a dummy result file in tmp_uploads
        dummy_result = self.tmp_uploads / "result_test_dummy.jpg"
        dummy_result.write_bytes(b"DUMMY_IMAGE")

        with self.client.session_transaction() as sess:
            sess["uploaded_file"] = "upload_dummy.jpg"
            sess["result_image"] = "result_test_dummy.jpg"
            sess["prediction"] = "Real"
            sess["accuracy"] = 94.5
            sess["confidence"] = 94.5
            sess["processing_time"] = 0.12
            sess["analysis_reasons"] = ["sharp texture", "face detected"]

        response = self.client.get("/result")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"UndefinedError", response.data)
        self.assertIn(b"Deepfake Analysis Result", response.data)
        self.assertIn(b"Real Face", response.data)

    def test_regression_background_asset_exists(self):
        """
        REGRESSION TEST FOR DEFECT ID: DEF-003
        Root cause: index.html & result.html referenced static/uploads/back.jpg which was absent.
        Verification: Ensure static/images/background.jpg (and back.jpg) exists and is accessible.
        """
        bg_jpg = BACKEND_DIR / "static" / "images" / "background.jpg"
        bg_png = BACKEND_DIR / "static" / "images" / "background.png"
        back_compat = BACKEND_DIR / "static" / "uploads" / "back.jpg"

        self.assertTrue(bg_jpg.exists() or bg_png.exists())
        self.assertTrue(back_compat.exists())

        # Test route serves the background asset
        resp = self.client.get("/static/images/background.jpg")
        self.assertEqual(resp.status_code, 200)

    def test_regression_download_empty_session_safe(self):
        """
        REGRESSION TEST FOR DEFECT ID: DEF-004
        Root cause: /download with empty session previously attempted to access
        the upload folder itself or raised KeyError on session['result_image'].
        Verification: Ensure it gracefully redirects to index with flash message.
        """
        response = self.client.get("/download", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No active result available", response.data)

    def test_regression_model_validation_diagnostic(self):
        """
        REGRESSION TEST FOR DEFECT ID: DEF-005
        Root cause: Corrupted or missing model files silently dropped to heuristic without diagnostic.
        Verification: validate_and_load_model must return diagnostic info with loaded=False and error details.
        """
        model, scaler, info = ml_pipeline.validate_and_load_model("bogus_path.pkl", "bogus_scaler.pkl")
        self.assertFalse(info["loaded"])
        self.assertIsNotNone(info["error"])
        self.assertIsNone(model)
        self.assertIsNone(scaler)


if __name__ == "__main__":
    unittest.main()
