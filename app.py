"""
Root entrypoint for Deepfake Detection Application.

Allows Gunicorn, WSGI runners, and local development to invoke the app
directly from repository root as:
    gunicorn app:app
or
    python app.py
"""

import os
import sys
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.app import app, create_app

if __name__ == "__main__":
    is_debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=is_debug, host="0.0.0.0", port=port)
