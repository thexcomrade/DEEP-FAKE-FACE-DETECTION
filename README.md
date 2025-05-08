# DEEP-FAKE-FACE-DETECTION

A Flask-based web application that detects deepfake images using classical computer vision techniques and a trained SVM model. Users can upload facial images, receive a real-time prediction, view detailed analysis, and download the result image with annotations.

🔍 Features
Upload facial images (JPG, PNG, WEBP)
Predict Real or Fake with confidence score
Uses ORB keypoints, Laplacian variance, histogram analysis
Optionally integrates a trained svm_model.pkl
Annotated result image with download option
Dark/light theme toggle & explanation of prediction

🛠️ Tech Stack
Backend: Python, Flask
ML Model: Scikit-learn (SVM), StandardScaler
Image Processing: OpenCV, PIL
Frontend: HTML, Jinja2 Templates
Utilities: Joblib, Logging, Session Management

📌 Notes
The app still works without the model files using heuristic-based detection.
Max file upload size: 16MB
Only specific image types are allowed (for security and consistency).

🧠 How It Works
Preprocesses the uploaded image to grayscale and resizes it
Extracts handcrafted features:
Laplacian Variance: for image sharpness
ORB Keypoints: for facial detail
Histogram Stats: for grayscale distribution
Feeds the features into the trained SVM model or uses a heuristic fallback
Predicts and annotates the result with confidence score
Displays reasoning behind the prediction
