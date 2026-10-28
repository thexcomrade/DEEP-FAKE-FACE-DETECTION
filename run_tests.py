"""
Automated Test Suite Runner for Deepfake Face Detection Application.

Discovers and executes all unit, integration, and regression tests.
"""

import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def run_all_tests():
    print("=" * 65)
    print("      RUNNING DEEPFAKE FACE DETECTION AUTOMATED TEST SUITE")
    print("=" * 65)

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(BASE_DIR / "tests"), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 65)
    print(f"Tests run:     {result.testsRun}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Skipped:       {len(result.skipped)}")
    print(f"Overall State: {'PASS' if result.wasSuccessful() else 'FAIL'}")
    print("=" * 65)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
