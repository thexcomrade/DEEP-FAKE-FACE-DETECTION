"""
Vercel Serverless Function entrypoint.
Imports and exposes the Flask WSGI application instance 'app' from root app.py,
with WSGI path normalization for Vercel internal rewrites.
"""
from __future__ import annotations

import sys
import urllib.parse
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
        qs = environ.get("QUERY_STRING", "")

        # 1. Check explicit __vercel_path passed via vercel.json rewrite
        if "__vercel_path" in qs:
            parsed_qs = urllib.parse.parse_qs(qs, keep_blank_values=True)
            if "__vercel_path" in parsed_qs:
                target_path = parsed_qs.pop("__vercel_path")[0]
                rebuilt_pairs = []
                for k, vs in parsed_qs.items():
                    for v in vs:
                        rebuilt_pairs.append((k, v))
                environ["QUERY_STRING"] = urllib.parse.urlencode(rebuilt_pairs)
                if not target_path.startswith("/"):
                    target_path = f"/{target_path}"
                environ["PATH_INFO"] = target_path or "/"
                return self.wsgi_app(environ, start_response)

        # 2. Check original URI headers set by proxy/CDN
        for hdr in ("HTTP_X_FORWARDED_URI", "HTTP_X_ORIGINAL_URL", "HTTP_X_REWRITE_URL"):
            val = environ.get(hdr)
            if val:
                path = val.split("?")[0]
                if path and not path.startswith("/api/index"):
                    environ["PATH_INFO"] = path
                    return self.wsgi_app(environ, start_response)

        # 3. Check matched path header
        matched = environ.get("HTTP_X_MATCHED_PATH")
        if matched and not matched.startswith("/api/index"):
            environ["PATH_INFO"] = matched.split("?")[0]
            return self.wsgi_app(environ, start_response)

        # 4. Standard path normalization
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
