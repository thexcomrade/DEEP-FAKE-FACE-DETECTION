"""
Deepfake Face Detection Web Application Backend.

Flask application integrating image upload validation, canonical SVM feature
extraction and inference, result image generation, session management, and
clean error handling.
"""

from __future__ import annotations

import datetime
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import secrets
import sys
import tempfile
from typing import Any, Dict, Optional

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from PIL import Image
from werkzeug.utils import secure_filename

# Local modules
try:
    from backend import annotator, ml_pipeline
except ImportError:
    import annotator
    import ml_pipeline

# Base filesystem directories
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_STATIC_DIR = BASE_DIR / "static"
DEFAULT_TEMPLATES_DIR = BASE_DIR / "templates"
DEFAULT_MODELS_DIR = BASE_DIR / "models"

# Detect serverless or read-only execution environments (Vercel, AWS Lambda, Cloud Functions)
IS_SERVERLESS = bool(
    os.environ.get("VERCEL")
    or os.environ.get("VERCEL_ENV")
    or os.environ.get("VERCEL_REGION")
    or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    or os.environ.get("LAMBDA_TASK_ROOT")
)
DEFAULT_UPLOADS_DIR = Path(tempfile.gettempdir()) if IS_SERVERLESS else (DEFAULT_STATIC_DIR / "uploads")



def forward_prediction_to_render(render_url: str, file_path: Path) -> Optional[Dict[str, Any]]:
    """Forward image to a remote Render backend /api/predict endpoint using urllib."""
    import json
    import urllib.error
    import urllib.request
    import uuid

    url = render_url.rstrip("/") + "/api/predict"
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        filename = file_path.name
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": "Deepfake-Vercel-Proxy/1.0",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=30.0) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logging.getLogger(__name__).warning("Remote Render forward failed (%s): %s", url, exc)
        return None


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """Flask application factory."""
    app = Flask(
        __name__,
        static_folder=str(DEFAULT_STATIC_DIR),
        template_folder=str(DEFAULT_TEMPLATES_DIR),
    )

    # Base configuration
    app.config["MODELS_FOLDER"] = str(DEFAULT_MODELS_DIR)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max
    app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "webp"}
    app.config["MAX_IMAGE_PIXELS"] = 25_000_000  # Decompression bomb limit

    # Ensure upload directory exists and is writable (fallback safely to /tmp if read-only)
    upload_target = Path(tempfile.gettempdir()) if IS_SERVERLESS else (DEFAULT_STATIC_DIR / "uploads")
    try:
        upload_target.mkdir(parents=True, exist_ok=True)
        # Test writability
        test_probe = upload_target / ".write_probe"
        test_probe.touch()
        test_probe.unlink(missing_ok=True)
    except (OSError, PermissionError):
        upload_target = Path(tempfile.gettempdir())
        upload_target.mkdir(parents=True, exist_ok=True)
    app.config["UPLOAD_FOLDER"] = str(upload_target)

    # Secret key configuration (stable fallback prevents crash on unconfigured dashboards)
    configured_secret = os.environ.get("SECRET_KEY")
    if configured_secret:
        app.secret_key = configured_secret
    else:
        app.secret_key = os.environ.get("SESSION_SECRET", "deepfake-detection-production-secret-982347102938")


    # Render backend integration config
    # When deployed to Vercel, connect by default to the Render backend service
    default_render_backend = "https://deep-fake-face-detection.onrender.com" if IS_SERVERLESS else ""
    app.config["RENDER_BACKEND_URL"] = os.environ.get("RENDER_BACKEND_URL", default_render_backend).rstrip("/")

    @app.after_request
    def set_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
        return response

    @app.before_request
    def handle_options_preflight():
        if request.method == "OPTIONS":
            response = app.make_default_options_response()
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
            return response

    # Apply testing or override configuration
    if test_config:
        app.config.update(test_config)

    # Logging setup: always log to stdout/stderr (essential for Vercel, Render, and containerized runtimes)
    app.logger.handlers.clear()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    app.logger.addHandler(stream_handler)
    app.logger.setLevel(logging.INFO)

    # Optional local file logging (safely ignored in read-only or serverless environments)
    if not IS_SERVERLESS:
        try:
            log_file = BASE_DIR / "app.log"
            file_handler = RotatingFileHandler(str(log_file), maxBytes=500_000, backupCount=3)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            app.logger.addHandler(file_handler)
        except (OSError, PermissionError):
            pass

    # Locate and load canonical machine learning model and scaler
    models_dir = Path(app.config.get("MODELS_FOLDER", DEFAULT_MODELS_DIR))
    if not models_dir.exists():
        # Fallback to root or alternative models directory
        alt_models_dir = BASE_DIR.parent / "models"
        if alt_models_dir.exists():
            models_dir = alt_models_dir

    model_file = models_dir / "svm_model.pkl"
    scaler_file = models_dir / "svm_scaler.pkl"

    model, scaler, ml_info = ml_pipeline.validate_and_load_model(model_file, scaler_file)
    app.config["ML_MODEL"] = model
    app.config["ML_SCALER"] = scaler
    app.config["ML_INFO"] = ml_info

    if ml_info.get("loaded"):
        app.logger.info("Canonical model loaded successfully: %s", ml_info)
    else:
        app.logger.warning("Canonical model offline: %s", ml_info.get("error"))

    def allowed_file(filename: str) -> bool:
        return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

    # ==========================================
    # ROUTES
    # ==========================================

    # When uploads are saved outside static/uploads (e.g. /tmp in serverless), serve them dynamically
    if app.config["UPLOAD_FOLDER"] != str(DEFAULT_STATIC_DIR / "uploads"):
        @app.route("/static/uploads/<path:filename>")
        def serve_serverless_upload(filename: str):
            from flask import send_from_directory
            return send_from_directory(str(app.config["UPLOAD_FOLDER"]), filename)

    @app.route("/")
    @app.route("/api/index")
    @app.route("/api/index/")
    def index():
        """Render upload home page."""
        return render_template(
            "index.html",
            theme=session.get("theme", "dark"),
            model_loaded=app.config["ML_MODEL"] is not None,
            model_info=app.config["ML_INFO"],
            render_backend_url=app.config.get("RENDER_BACKEND_URL", ""),
        )

    @app.route("/upload", methods=["POST"])
    @app.route("/api/index/upload", methods=["POST"])
    def upload_image():
        """Handle image upload, validation, inference, and result annotation."""
        if "file" not in request.files or not request.files["file"].filename:
            flash("No file was selected. Please choose an image to analyze.", "error")
            return redirect(url_for("index"))

        file = request.files["file"]
        orig_filename = secure_filename(file.filename) or "upload.jpg"

        if not allowed_file(orig_filename):
            allowed = ", ".join(ext.upper() for ext in sorted(app.config["ALLOWED_EXTENSIONS"]))
            flash(f"Invalid file type. Allowed formats: {allowed}.", "error")
            return redirect(url_for("index"))

        start_time = datetime.datetime.now()

        # Generate collision-resistant filename
        timestamp = start_time.strftime("%Y%m%d_%H%M%S")
        token = secrets.token_hex(4)
        safe_filename = f"{timestamp}_{token}_{orig_filename}"
        file_path = Path(app.config["UPLOAD_FOLDER"]) / safe_filename

        try:
            file.save(str(file_path))

            # Pillow image verification and decompression bomb check
            try:
                with Image.open(file_path) as img:
                    img_format = img.format
                    width, height = img.size
                    if (width * height) > app.config["MAX_IMAGE_PIXELS"]:
                        raise ValueError(f"Image resolution exceeds limits ({width}x{height}).")
                    if img_format not in ["PNG", "JPEG", "WEBP"]:
                        raise ValueError(f"Image format {img_format} not supported.")
                    img.verify()
            except Exception as val_exc:
                if file_path.exists():
                    file_path.unlink()
                app.logger.warning("Upload image verification failed for %s: %s", orig_filename, val_exc)
                flash("Invalid or corrupted image file. Please upload a standard JPG, PNG, or WEBP.", "error")
                return redirect(url_for("index"))

            # Run machine learning prediction
            model = app.config.get("ML_MODEL")
            scaler = app.config.get("ML_SCALER")
            prediction_result = ml_pipeline.predict_image(
                file_path,
                model=model,
                scaler=scaler,
                allow_heuristic_fallback=True,
            )

            if prediction_result.prediction == "Error":
                if file_path.exists():
                    file_path.unlink()
                flash(prediction_result.error or "Analysis failed. Please try a different image.", "error")
                return redirect(url_for("index"))

            # Generate annotated result image
            result_filename = f"result_{safe_filename}"
            result_img_path = Path(app.config["UPLOAD_FOLDER"]) / result_filename

            annotated_path = annotator.generate_result_image(
                image_path=file_path,
                result=prediction_result.prediction,
                confidence=prediction_result.confidence,
                output_path=result_img_path,
            )

            if not annotated_path or not Path(annotated_path).exists():
                if file_path.exists():
                    file_path.unlink()
                flash("Failed to generate annotated result image.", "error")
                return redirect(url_for("index"))

            # Generate Explainable AI (XAI) feature attribution heatmap if available
            saliency_filename = f"saliency_{safe_filename}"
            saliency_img_path = Path(app.config["UPLOAD_FOLDER"]) / saliency_filename
            saliency_path = ml_pipeline.generate_saliency_map(
                image_or_path=file_path,
                model=model,
                scaler=scaler,
                output_path=saliency_img_path,
            )

            elapsed = round((datetime.datetime.now() - start_time).total_seconds(), 2)

            # Store result details in session
            session["uploaded_file"] = safe_filename
            session["original_filename"] = orig_filename
            session["result_image"] = result_filename
            session["saliency_image"] = saliency_filename if (saliency_path and Path(saliency_path).exists()) else None
            session["prediction"] = prediction_result.prediction
            session["confidence"] = prediction_result.confidence
            session["accuracy"] = prediction_result.confidence  # Backwards compatibility
            session["processing_time"] = elapsed
            session["analysis_reasons"] = prediction_result.analysis_reasons
            session["probabilities"] = prediction_result.probabilities
            session["is_heuristic"] = prediction_result.is_heuristic_fallback
            session["model_used"] = prediction_result.model_used
            session["model_name"] = prediction_result.model_name

            return redirect(url_for("result_page"))

        except Exception as exc:
            app.logger.error("Unexpected upload processing error: %s", exc, exc_info=True)
            if file_path.exists():
                file_path.unlink()
            flash("An unexpected error occurred while processing the image. Please try again.", "error")
            return redirect(url_for("index"))

    @app.route("/result")
    @app.route("/api/index/result")
    def result_page():
        """Render results page."""
        uploaded_file = session.get("uploaded_file")
        result_img = session.get("result_image")

        if not uploaded_file or not result_img:
            flash("No active analysis found. Please upload an image first.", "error")
            return redirect(url_for("index"))

        result_path = Path(app.config["UPLOAD_FOLDER"]) / result_img
        if not result_path.exists():
            flash("The analyzed result image has expired or is missing. Please re-upload.", "error")
            return redirect(url_for("index"))

        return render_template(
            "result.html",
            image_filename=uploaded_file,
            filename=uploaded_file,  # Fixes legacy Jinja UndefinedError for 'filename'
            original_filename=session.get("original_filename", uploaded_file),
            result_img=result_img,
            saliency_img=session.get("saliency_image"),
            prediction=session.get("prediction", "Unknown"),
            accuracy=session.get("accuracy", 0.0),
            confidence=session.get("confidence", session.get("accuracy", 0.0)),
            processing_time=session.get("processing_time", 0.0),
            analysis_reasons=session.get("analysis_reasons", []),
            probabilities=session.get("probabilities", {}),
            is_heuristic=session.get("is_heuristic", False),
            model_used=session.get("model_used", False),
            model_name=session.get("model_name", "SVM Detector"),
            theme=session.get("theme", "dark"),
            model_loaded=app.config["ML_MODEL"] is not None,
        )

    @app.route("/download")
    @app.route("/api/index/download")
    def download():
        """Safely send the annotated result image to the client."""
        result_img = session.get("result_image")
        if not result_img:
            flash("No active result available for download. Upload an image first.", "error")
            return redirect(url_for("index"))

        clean_name = secure_filename(result_img)
        upload_dir = Path(app.config["UPLOAD_FOLDER"]).resolve()
        target_path = (upload_dir / clean_name).resolve()

        # Prevent directory traversal
        try:
            target_path.relative_to(upload_dir)
        except ValueError:
            flash("Unauthorized path access attempted.", "error")
            return redirect(url_for("index"))

        if not target_path.exists() or not target_path.is_file():
            flash("Requested download file was not found.", "error")
            return redirect(url_for("index"))

        download_name = f"deepfake_analysis_{clean_name}"
        return send_file(str(target_path), as_attachment=True, download_name=download_name)

    @app.route("/toggle-theme", methods=["GET", "POST"])
    @app.route("/api/index/toggle-theme", methods=["GET", "POST"])
    def toggle_theme():
        """Toggle UI theme between dark and light modes."""
        current = session.get("theme", "dark")
        new_theme = "light" if current == "dark" else "dark"
        session["theme"] = new_theme

        if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return jsonify({"theme": new_theme, "status": "success"})

        return redirect(request.referrer or url_for("index"))

    @app.route("/health")
    @app.route("/api/index/health")
    @app.route("/api/status")
    def health():
        """Diagnostic health check endpoint."""
        ml_info = app.config.get("ML_INFO", {})
        return jsonify({
            "status": "online",
            "model_loaded": app.config.get("ML_MODEL") is not None,
            "model_type": ml_info.get("model_type"),
            "features_expected": ml_info.get("n_features_in"),
            "model_path": ml_info.get("model_path"),
            "scaler_loaded": app.config.get("ML_SCALER") is not None,
            "upload_folder": app.config.get("UPLOAD_FOLDER"),
            "render_backend_url": app.config.get("RENDER_BACKEND_URL", ""),
            "timestamp": datetime.datetime.now().isoformat(),
        })

    @app.route("/api/predict", methods=["POST"])
    @app.route("/api/index/api/predict", methods=["POST"])
    def api_predict():
        """REST API endpoint for programmatic image prediction."""
        if "file" not in request.files or not request.files["file"].filename:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["file"]
        if not allowed_file(file.filename):
            return jsonify({"error": "Unsupported file format"}), 400

        temp_name = f"api_tmp_{secrets.token_hex(6)}_{secure_filename(file.filename)}"
        temp_path = Path(app.config["UPLOAD_FOLDER"]) / temp_name
        try:
            file.save(str(temp_path))
            res = ml_pipeline.predict_image(
                temp_path,
                model=app.config.get("ML_MODEL"),
                scaler=app.config.get("ML_SCALER"),
                allow_heuristic_fallback=True,
            )
            return jsonify(res.to_dict())
        finally:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

    # ==========================================
    # ERROR HANDLERS
    # ==========================================

    @app.errorhandler(413)
    def entity_too_large(e):
        return render_template(
            "error.html",
            error_message="Upload File Too Large",
            error_details="The selected file exceeds the 16MB maximum upload size. Please select a smaller image.",
            theme=session.get("theme", "dark"),
        ), 413

    @app.errorhandler(404)
    def page_not_found(e):
        return render_template(
            "error.html",
            error_message="Page Not Found",
            error_details="The requested resource or page does not exist.",
            theme=session.get("theme", "dark"),
        ), 404

    @app.errorhandler(500)
    def internal_server_error(e):
        app.logger.error("Unhandled internal server error: %s", e, exc_info=True)
        return render_template(
            "error.html",
            error_message="Internal Server Error",
            error_details="An unexpected server error occurred. Our engineering team has been notified.",
            theme=session.get("theme", "dark"),
        ), 500

    return app


# Default application instance for WSGI deployment
app = create_app()

if __name__ == "__main__":
    is_debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=is_debug, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
