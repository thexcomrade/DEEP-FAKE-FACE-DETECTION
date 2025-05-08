import os
import cv2
import numpy as np
import joblib  # Load trained model

# ✅ Load the trained model
MODEL_PATH = r"D:\FlaskApp\backend\models\model.pkl"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
    exit()

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")

# ✅ Function to preprocess test images
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ Error: Test image '{image_path}' not found!")
        exit()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"❌ Error: Unable to read image '{image_path}'")
        exit()

    img = cv2.resize(img, (64, 64))  # Resize to match training size
    img = img.flatten().reshape(1, -1)  # Flatten and reshape for SVM
    return img

# ✅ Path to the test image (Change this to test different images)
TEST_IMAGE_PATH = r"D:\FlaskApp\test_image.jpg"

# ✅ Process the test image
test_image = preprocess_image(TEST_IMAGE_PATH)

# ✅ Predict using the trained SVM model
prediction = model.predict(test_image)
confidence = model.predict_proba(test_image)[0]  # Get probability scores

# ✅ Display the result
if prediction[0] == 0:
    print(f"🔍 Prediction: Real ✅ with Confidence: {confidence[0]*100:.2f}%")
else:
    print(f"🔍 Prediction: Fake 🛑 with Confidence: {confidence[1]*100:.2f}%")
