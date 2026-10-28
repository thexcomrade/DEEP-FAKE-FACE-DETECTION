"""
Integration tests for Flask application routes, session handling, upload validation,
result rendering, and error recovery (backend/app.py).
"""

import io
from pathlib import Path
import unittest

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import create_app


class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = BASE_DIR / "tests" / "fixtures"
        self.tmp_uploads = BASE_DIR / "tests" / "tmp_uploads"
        self.tmp_uploads.mkdir(parents=True, exist_ok=True)

        self.app = create_app({
            "TESTING": True,
            "UPLOAD_FOLDER": str(self.tmp_uploads),
            "SECRET_KEY": "test-secret-key-12345",
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

    def test_get_index_page(self):
        """GET / returns 200 OK and renders upload form."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Deepfake Face Detection", response.data)
        self.assertIn(b"uploadForm", response.data)

    def test_upload_missing_file(self):
        """POST /upload without file in request is rejected and redirected."""
        response = self.client.post("/upload", data={}, follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No file was selected", response.data)

    def test_upload_empty_filename(self):
        """POST /upload with empty filename is rejected."""
        data = {"file": (io.BytesIO(b""), "")}
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No file was selected", response.data)

    def test_upload_unsupported_extension(self):
        """POST /upload with .txt file is rejected."""
        data = {"file": (io.BytesIO(b"Hello world"), "test.txt")}
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid file type", response.data)

    def test_upload_corrupted_image(self):
        """POST /upload with fake corrupted image bytes is safely rejected."""
        corrupted_bytes = (self.fixtures_dir / "corrupted.jpg").read_bytes()
        data = {"file": (io.BytesIO(corrupted_bytes), "corrupt.jpg")}
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Invalid or corrupted image file", response.data)

    def test_upload_valid_image_success(self):
        """POST /upload with valid JPEG successfully analyzes image and redirects to /result."""
        img_bytes = (self.fixtures_dir / "valid_test.jpg").read_bytes()
        data = {"file": (io.BytesIO(img_bytes), "sample.jpg")}
        response = self.client.post("/upload", data=data, content_type="multipart/form-data", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Deepfake Analysis Result", response.data)
        self.assertIn(b"Model Confidence", response.data)
        # Verify result image was generated in uploads folder
        result_files = list(self.tmp_uploads.glob("result_*"))
        self.assertGreaterEqual(len(result_files), 1)

    def test_result_page_direct_access_without_session(self):
        """Navigating directly to /result without session redirects cleanly to / with flash."""
        response = self.client.get("/result", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No active analysis found", response.data)

    def test_download_without_session(self):
        """Accessing /download without session redirects cleanly to / with flash."""
        response = self.client.get("/download", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"No active result available", response.data)

    def test_download_valid_result(self):
        """GET /download with active session returns file attachment."""
        # 1. Create a dummy result file
        test_file = self.tmp_uploads / "result_test_dummy.jpg"
        test_file.write_bytes(b"DUMMY_IMAGE_BYTES")

        # 2. Establish session
        with self.client.session_transaction() as sess:
            sess["result_image"] = "result_test_dummy.jpg"

        # 3. Request download
        response = self.client.get("/download")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b"DUMMY_IMAGE_BYTES")
        self.assertIn("attachment", response.headers.get("Content-Disposition", ""))

    def test_download_directory_traversal_blocked(self):
        """Attempted path traversal in session result_image is safely blocked."""
        with self.client.session_transaction() as sess:
            sess["result_image"] = "../../../etc/passwd"

        response = self.client.get("/download", follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Requested download file was not found", response.data)

    def test_toggle_theme(self):
        """GET /toggle-theme flips session theme between dark and light."""
        # Initial is dark -> flips to light
        resp1 = self.client.get("/toggle-theme")
        self.assertEqual(resp1.status_code, 302)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess["theme"], "light")

        # Flips back to dark
        resp2 = self.client.get("/toggle-theme")
        self.assertEqual(resp2.status_code, 302)
        with self.client.session_transaction() as sess:
            self.assertEqual(sess["theme"], "dark")

    def test_health_endpoint(self):
        """GET /health returns JSON with online status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("model_loaded", data)

    def test_history_dashboard_page(self):
        """GET /history returns 200 and renders detection history dashboard."""
        response = self.client.get("/history")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Detection History", response.data)
        self.assertIn(b"Total Faces Analyzed", response.data)

    def test_api_get_and_clear_history(self):
        """GET /api/history and POST /api/history/clear work predictably."""
        # 1. Get history
        get_res = self.client.get("/api/history")
        self.assertEqual(get_res.status_code, 200)
        data = get_res.get_json()
        self.assertIn("history", data)
        self.assertIn("total_scans", data)

        # 2. Clear history
        clear_res = self.client.post("/api/history/clear")
        self.assertEqual(clear_res.status_code, 200)
        clear_data = clear_res.get_json()
        self.assertEqual(clear_data["status"], "cleared")
        self.assertEqual(clear_data["total_scans"], 0)

    def test_api_predict(self):
        """POST /api/predict returns structured prediction JSON."""
        img_bytes = (self.fixtures_dir / "valid_test.jpg").read_bytes()
        data = {"file": (io.BytesIO(img_bytes), "api_test.jpg")}
        response = self.client.post("/api/predict", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        json_data = response.get_json()
        self.assertIn("prediction", json_data)
        self.assertIn("confidence", json_data)
        self.assertIn("probabilities", json_data)


if __name__ == "__main__":
    unittest.main()
