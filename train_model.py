"""
Root wrapper for Deepfake Detection Training & Evaluation CLI.
Delegates directly to backend/train_model.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import train_model

if __name__ == "__main__":
    train_model.main()
