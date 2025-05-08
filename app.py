from flask import Flask, render_template, request, send_file, redirect, url_for, session, flash
import os
import cv2
import numpy as np
import datetime
from PIL import Image, ImageDraw, ImageFont
import joblib
from sklearn.preprocessing import StandardScaler
from werkzeug.utils import secure_filename
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Logging setup
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load model
model, scaler = None, None
try:
    if os.path.exists('models/svm_model.pkl') and os.path.exists('models/svm_scaler.pkl'):
        model = joblib.load('models/svm_model.pkl')
        scaler = joblib.load('models/svm_scaler.pkl')
        app.logger.info("✅ Model loaded successfully")
    else:
        app.logger.warning("⚠️ Model files not found")
except Exception as e:
    app.logger.error(f"⚠️ Model load error: {str(e)}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_features(img):
    try:
        img = cv2.resize(img, (64, 64))
        features = img.flatten().tolist()
        features.append(cv2.Laplacian(img, cv2.CV_64F).var())

        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(img, None)
        features.append(len(kp) if kp is not None else 0)

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        features.extend(hist.flatten())
        return np.array(features)
    except Exception as e:
        app.logger.error(f"Feature extraction error: {str(e)}")
        return None

def preprocess_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        features = extract_features(img)
        return scaler.transform([features]) if scaler else features.reshape(1, -1)
    except Exception as e:
        app.logger.error(f"Preprocessing error: {str(e)}")
        return None

def detect_deepfake(image_path):
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return "Error: Invalid Image", 0.0, []

        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        orb = cv2.ORB_create(nfeatures=500)
        keypoint_count = len(orb.detect(img, None))

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_std = np.std(hist)

        confidence = (
            0.4 * min(1, laplacian_var / 85) +
            0.4 * min(1, keypoint_count / 180) +
            0.2 * min(1, hist_std / 25)
        ) * 100
        confidence = max(60, min(99, round(confidence, 1)))

        reasons = []
        if laplacian_var > 55:
            reasons.append("sharp and focused image")
        else:
            reasons.append("blurry texture")

        if keypoint_count > 120:
            reasons.append("high facial detail")
        else:
            reasons.append("limited facial detail")

        if hist_std > 10:
            reasons.append("natural contrast")
        else:
            reasons.append("flat grayscale distribution")

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            reasons.append("face detected")
        else:
            reasons.append("no face detected")

        if model and scaler:
            features = preprocess_image(image_path)
            if features is not None:
                prediction = model.predict(features)[0]
                proba = model.predict_proba(features)[0]
                confidence = round(max(proba) * 100, 1)
                reasons.append("classified using trained model")
                return "Real" if prediction == 0 else "Fake", confidence, reasons

        prediction = "Real" if (laplacian_var > 55 and keypoint_count > 120) else "Fake"
        return prediction, confidence, reasons

    except Exception as e:
        app.logger.error(f"Detection error: {str(e)}")
        return "Error: Detection Failed", 0.0, []

def generate_result_image(image_path, result, confidence):
    try:
        with Image.open(image_path).convert("RGB") as img:
            draw = ImageDraw.Draw(img)
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except IOError:
                font = ImageFont.load_default()

            text = f"{result} ({confidence}%)"
            x, y = img.size[0] // 2, img.size[1] - 50

            text_bbox = draw.textbbox((x, y), text, font=font, anchor="mm")
            bg = Image.new('RGBA', img.size, (0, 0, 0, 0))
            bg_draw = ImageDraw.Draw(bg)
            bg_draw.rectangle(
                (text_bbox[0] - 15, text_bbox[1] - 15, text_bbox[2] + 15, text_bbox[3] + 15),
                fill=(0, 0, 0, 150)
            )
            img = Image.alpha_composite(img.convert('RGBA'), bg).convert('RGB')
            draw = ImageDraw.Draw(img)
            draw.text((x, y), text, fill="#4CAF50" if result == "Real" else "#F44336", font=font, anchor="mm")

            result_name = f"result_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{secure_filename(os.path.basename(image_path))}"
            result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_name)
            img.save(result_path, quality=90, optimize=True)
            return result_path
    except Exception as e:
        app.logger.error(f"Result image generation error: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template("index.html", theme=session.get("theme", "dark"), model_loaded=model is not None)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files or request.files['file'].filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if not allowed_file(file.filename):
        flash('Invalid file type. Upload PNG, JPG, JPEG, or WEBP.', 'error')
        return redirect(url_for('index'))

    try:
        start = datetime.datetime.now()
        filename = f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{secure_filename(file.filename)}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            os.remove(file_path)
            flash('Invalid image file', 'error')
            return redirect(url_for('index'))

        result, accuracy, reasons = detect_deepfake(file_path)
        if result.startswith("Error"):
            os.remove(file_path)
            flash(result, 'error')
            return redirect(url_for('index'))

        result_img_path = generate_result_image(file_path, result, accuracy)
        if not result_img_path:
            os.remove(file_path)
            flash('Failed to generate result image', 'error')
            return redirect(url_for('index'))

        end = datetime.datetime.now()
        session.update({
            'uploaded_file': filename,
            'prediction': result,
            'accuracy': accuracy,
            'processing_time': round((end - start).total_seconds(), 2),
            'result_image': os.path.basename(result_img_path),
            'analysis_reasons': reasons
        })

        return redirect(url_for('result_page'))
    except Exception as e:
        app.logger.error(f"Upload error: {str(e)}")
        flash('Unexpected error during processing.', 'error')
        return redirect(url_for('index'))

@app.route('/result')
def result_page():
    if not session.get('uploaded_file'):
        flash('No analysis found. Upload an image first.', 'error')
        return redirect(url_for('index'))

    path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('result_image', ''))
    if not os.path.exists(path):
        flash('Result image not found.', 'error')
        return redirect(url_for('index'))

    return render_template("result.html",
        image_filename=session['uploaded_file'],
        result_img=session['result_image'],
        prediction=session['prediction'],
        accuracy=session['accuracy'],
        processing_time=session['processing_time'],
        analysis_reasons=session.get('analysis_reasons', []),
        theme=session.get("theme", "dark"),
        model_loaded=model is not None
    )

@app.route('/download')
def download():
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], session.get('result_image', ''))
    if not os.path.exists(result_path):
        flash('Download failed: file missing.', 'error')
        return redirect(url_for('index'))
    return send_file(result_path, as_attachment=True, download_name=f"deepfake_result_{session['result_image']}")

@app.route('/toggle-theme')
def toggle_theme():
    session['theme'] = 'light' if session.get('theme', 'dark') == 'dark' else 'dark'
    return redirect(request.referrer or url_for('index'))

if __name__ == '__main__':
    app.run(debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true')
