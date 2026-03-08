import cv2
import tensorflow as tf
import numpy as np
import os

# -------------------------------
# Configuration
# -------------------------------
MODEL_PATH = "models/cnn_mudra_model.h5"
IMG_SIZE = 224

# -------------------------------
# Load Model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
class_names = sorted(os.listdir("segmented_images"))

print("\n=== Bharatanatyam Mudra Recognition ===")
image_path = input("Enter image path (.jpg / .png): ").strip()

# -------------------------------
# Validate Image Path
# -------------------------------
if not os.path.exists(image_path):
    raise FileNotFoundError(" Image file not found. Check the path.")

# -------------------------------
# Load & Preprocess Image
# -------------------------------
img = cv2.imread(image_path)

if img is None:
    raise ValueError(" Unable to read image. Invalid format.")

display_img = img.copy()

img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

# -------------------------------
# Prediction
# -------------------------------
preds = model.predict(img, verbose=0)
class_id = np.argmax(preds)
confidence = preds[0][class_id] * 100

label = f"{class_names[class_id]} : {confidence:.2f}%"

# -------------------------------
# Display Result
# -------------------------------
cv2.putText(display_img, label, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2)

cv2.imshow("Mudra Recognition - Image Input", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
