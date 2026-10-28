"""
Root CLI wrapper for Deepfake Detection Test script.
Delegates to backend/test_model.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import test_model

if __name__ == "__main__":
    test_model.main()
