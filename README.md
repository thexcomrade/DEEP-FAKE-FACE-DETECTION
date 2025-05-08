# DEEP-FAKE-FACE-DETECTION

A Flask-based web application that detects deepfake images using classical computer vision techniques and a trained SVM model. Users can upload facial images, receive a real-time prediction, view detailed analysis, and download the result image with annotations.

🚀 Features
📤 Upload images (JPG, PNG, JPEG, WEBP)

🔍 Analyze image sharpness, facial detail, and contrast

🧪 Classify images as Real or Fake using:

Rule-based heuristics (fallback)

Trained SVM model (if available)

🖼️ Generate and display annotated result image with confidence score

🧾 Explanation of prediction (key reasons highlighted)

⬇️ Download processed result image

🌙 Supports dark/light theme toggle

🪵 Logging with rotating file logs (app.log)

🛠️ Tech Stack
Backend: Python, Flask

ML Model: Scikit-learn (SVM), StandardScaler

Image Processing: OpenCV, PIL

Frontend: HTML, Jinja2 Templates

Utilities: Joblib, Logging, Session Management

