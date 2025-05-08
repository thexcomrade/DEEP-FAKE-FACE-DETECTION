import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Configurations
DATASET_PATH = r"D:\FlaskApp\Dataset\Train"
MODELS_PATH = "models"
IMAGE_SIZE = (64, 64)  # Use 64x64 image resolution for better feature extraction

def create_directory_structure():
    """Ensure required directories exist"""
    os.makedirs(os.path.join(DATASET_PATH, "Real"), exist_ok=True)
    os.makedirs(os.path.join(DATASET_PATH, "Fake"), exist_ok=True)
    os.makedirs(MODELS_PATH, exist_ok=True)
    print("✅ Directory structure verified")

def check_dataset():
    """Check if dataset contains enough real and fake images"""
    def count_images(folder):
        return len([f for f in os.listdir(folder) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if os.path.exists(folder) else 0

    real_count = count_images(os.path.join(DATASET_PATH, "Real"))
    fake_count = count_images(os.path.join(DATASET_PATH, "Fake"))

    print(f"\n📊 Dataset Summary:\n- Real: {real_count} images\n- Fake: {fake_count} images")
    return real_count > 0 and fake_count > 0

def extract_features(image):
    """
    Extract features for deepfake detection:
    1. Blur detection
    2. RGB color statistics
    3. Edge density (Canny)
    4. Texture (Sobel)
    """
    try:
        img = cv2.resize(image, IMAGE_SIZE)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 1. Blur detection (Laplacian variance)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()

        # 2. Color features (mean, std, hist mean for R, G, B)
        color_features = []
        for channel in cv2.split(img):
            hist = cv2.calcHist([channel], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            color_features.extend([np.mean(channel), np.std(channel), np.mean(hist)])

        # 3. Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.mean(edges > 0)

        # 4. Texture features using Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        texture = np.mean([np.std(sobelx), np.std(sobely)])

        return np.array([blur, edge_density, texture] + color_features)

    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return np.zeros(12)  # 3 base + 3 features per color channel

def load_images(directory, label):
    """Load and process images from a directory"""
    features, labels = [], []

    if not os.path.exists(directory):
        print(f"⚠️ Directory not found: {directory}")
        return np.array([]), np.array([])

    print(f"\n📂 Processing images from: {os.path.basename(directory)}")
    for filename in os.listdir(directory):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        path = os.path.join(directory, filename)
        image = cv2.imread(path)
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        feat = extract_features(image)
        features.append(feat)
        labels.append(label)

    print(f"✅ Processed {len(labels)} images")
    return np.array(features), np.array(labels)

def main():
    print("🚀 Starting training for Deepfake Detection (128x128 resolution)")
    create_directory_structure()

    if not check_dataset():
        print("❌ Please make sure both 'Real' and 'Fake' folders have images.")
        return

    # Load real and fake images
    X_real, y_real = load_images(os.path.join(DATASET_PATH, "Real"), 0)
    X_fake, y_fake = load_images(os.path.join(DATASET_PATH, "Fake"), 1)

    if len(X_real) == 0 or len(X_fake) == 0:
        print("❌ Insufficient images to train the model.")
        return

    # Combine data
    X = np.vstack((X_real, X_fake))
    y = np.concatenate((y_real, y_fake))

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    print("\n🧠 Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy: {accuracy:.4f}")
    print("\n📄 Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

    # Save model and scaler
    joblib.dump(model, os.path.join(MODELS_PATH, "deepfake_model_128.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_PATH, "deepfake_scaler_128.pkl"))
    print(f"\n💾 Model saved in '{MODELS_PATH}' folder.")

if __name__ == "__main__":
    main()
