import os
import cv2
import numpy as np
import joblib  # To save and load model
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 🔹 Dataset Paths
DATASET_PATH = "FlaskApp/dataset/Train"
TEST_PATH = "FlaskApp/dataset/Test"

# 🔹 Feature Extraction Function
def extract_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None  # Skip invalid images
    
    # 1️⃣ Laplacian Variance (Blur Detection)
    laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()

    # 2️⃣ Canny Edge Density
    edges = cv2.Canny(img, 100, 200)
    edge_density = np.sum(edges) / (img.shape[0] * img.shape[1])

    # 3️⃣ ORB Keypoint Count
    orb = cv2.ORB_create()
    keypoints = orb.detect(img, None)
    keypoint_count = len(keypoints)

    # 4️⃣ Image Brightness
    brightness = np.mean(img)

    return [laplacian_var, edge_density, keypoint_count, brightness]

# 🔹 Load Dataset
def load_dataset(dataset_path):
    X, y = [], []
    
    for label, category in enumerate(["Real", "Fake"]):
        category_path = os.path.join(dataset_path, category)
        
        for filename in os.listdir(category_path):
            image_path = os.path.join(category_path, filename)
            features = extract_features(image_path)
            
            if features:
                X.append(features)
                y.append(label)  # 0 = Real, 1 = Fake
    
    return np.array(X), np.array(y)

# 🔹 Load Training Data
X, y = load_dataset(DATASET_PATH)

# 🔹 Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Standardize Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# 🔹 Train Classifier (SVM)
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)

# 🔹 Validate Model
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# 🔹 Save Model & Scaler
joblib.dump(svm_model, "FlaskApp/backend/deepfake_model.pkl")
joblib.dump(scaler, "FlaskApp/backend/scaler.pkl")

print("✅ Model Training Complete! Saved as deepfake_model.pkl")
