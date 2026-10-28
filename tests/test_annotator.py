"""
Unit tests for Result Image Annotator (backend/annotator.py).
"""

from pathlib import Path
import unittest
from PIL import Image

import sys
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import annotator


class TestAnnotator(unittest.TestCase):
    def setUp(self):
        self.fixtures_dir = BASE_DIR / "tests" / "fixtures"
        self.valid_jpg = self.fixtures_dir / "valid_test.jpg"
        self.small_jpg = self.fixtures_dir / "small_test.jpg"
        self.output_dir = BASE_DIR / "tests" / "tmp_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for f in self.output_dir.iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            self.output_dir.rmdir()
        except OSError:
            pass

    def test_generate_result_image_standard(self):
        """Annotated image is generated, saved, and readable with expected label and size."""
        out_path = self.output_dir / "annot_result.jpg"
        result = annotator.generate_result_image(self.valid_jpg, "Real", 96.4, out_path)
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

        with Image.open(result) as img:
            self.assertEqual(img.size, (120, 120))
            self.assertEqual(img.format, "JPEG")

    def test_generate_result_image_small_image(self):
        """Small images (40x40) scale font dynamically without crashing."""
        out_path = self.output_dir / "small_annot.jpg"
        result = annotator.generate_result_image(self.small_jpg, "Fake", 88.0, out_path)
        self.assertIsNotNone(result)
        self.assertTrue(Path(result).exists())

    def test_generate_result_image_missing_source(self):
        """Nonexistent input image returns None safely."""
        result = annotator.generate_result_image("nonexistent_img_123.jpg", "Real", 90.0)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
