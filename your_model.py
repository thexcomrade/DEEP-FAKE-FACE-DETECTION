import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Image size must match training size
IMG_SIZE = (64, 64)

def preprocess_image(image_path):
    """Preprocess image: grayscale, resize, normalize"""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    image = cv2.resize(image, IMG_SIZE)
    feature_vector = image.flatten()  # Convert to 1D feature vector
    return feature_vector

# Load dataset
def load_data(dataset_path):
    X, y = [], []
    for label, category in enumerate(["Real", "Fake"]):
        category_path = os.path.join(dataset_path, category)
        
        if not os.path.exists(category_path):
            raise FileNotFoundError(f"❌ Directory not found: {category_path}")

        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            features = preprocess_image(image_path)
            if features is not None:
                X.append(features)
                y.append(label)
    
    return np.array(X), np.array(y)

# 🔹 Update dataset path with correct absolute path
dataset_path = r"D:\FlaskApp\dataset\Train"  # Change this to your actual dataset location
X, y = load_data(dataset_path)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel="linear", probability=True)
svm_model.fit(X_train, y_train)

# Save trained model and scaler
model_path = r"D:\FlaskApp\models\model.pkl"  # Ensure this path exists
joblib.dump((svm_model, scaler), model_path)
print(f"✅ Model saved at {model_path}")
