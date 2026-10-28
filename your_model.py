"""
Root reference wrapper for your_model.py.
Delegates to backend/your_model.py.
"""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from your_model import preprocess_image, load_data

if __name__ == "__main__":
    import your_model
