"""
Vercel Serverless Function entrypoint.
Imports and exposes the Flask WSGI application instance 'app' from root app.py,
with WSGI path normalization for Vercel internal rewrites.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root and backend directory to Python sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

for p in (ROOT_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from app import app

# Expose app and handler for Vercel Python serverless runtime
handler = app
