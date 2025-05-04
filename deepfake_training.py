import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Paths to dataset folders
TRAIN_DIR = "backend/dataset/Train"
VAL_DIR = "backend/dataset/Validation"

# Function to load images
def load_images_from_folder(folder, label):
    data = []
    labels = []
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Folder not found: {folder}")
        return data, labels

    for category in ["Real", "Fake"]:
        category_path = os.path.join(folder, category)
        if not os.path.exists(category_path):
            print(f"⚠️ Warning: Missing category folder: {category_path}")
            continue

        for filename in os.listdir(category_path):
            img_path = os.path.join(category_path, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️ Skipping invalid image: {img_path}")
                continue

            image = cv2.resize(image, (64, 64))  # Resize to uniform size
            data.append(image.flatten())  # Flatten image
            labels.append(0 if category == "Real" else 1)

    return np.array(data), np.array(labels)

# Load data
X_train, y_train = load_images_from_folder(TRAIN_DIR, "Train")
X_val, y_val = load_images_from_folder(VAL_DIR, "Validation")

# Ensure data is not empty
if len(X_train) == 0 or len(y_train) == 0:
    raise ValueError("🚨 No training data found. Add images to 'backend/dataset/Train'.")

if len(X_val) == 0 or len(y_val) == 0:
    print("⚠️ Warning: No validation data found. The model will train without validation.")

# Split training data for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Train SVM model
model = SVC(kernel="linear", probability=True)
model.fit(X_train, y_train)

# Save model
os.makedirs("backend", exist_ok=True)
joblib.dump(model, "backend/model.pkl")

print("✅ Model saved as backend/model.pkl")
