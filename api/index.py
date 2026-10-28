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


class VercelPathMiddleware:
    """
    Normalizes WSGI PATH_INFO when Vercel internal rewrites forward requests
    to the /api/index serverless handler.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        matched = environ.get("HTTP_X_MATCHED_PATH")
        if matched and not matched.startswith("/api/index"):
            environ["PATH_INFO"] = matched.split("?")[0]
        else:
            path = environ.get("PATH_INFO", "")
            if path in ("/api/index", "/api/index/"):
                environ["PATH_INFO"] = "/"
            elif path.startswith("/api/index/"):
                environ["PATH_INFO"] = path[len("/api/index"):]
            elif path.startswith("/api/") and not path.startswith("/api/predict"):
                environ["PATH_INFO"] = path[len("/api"):]
        return self.wsgi_app(environ, start_response)


# Wrap Flask's WSGI application with Vercel path normalization
app.wsgi_app = VercelPathMiddleware(app.wsgi_app)
