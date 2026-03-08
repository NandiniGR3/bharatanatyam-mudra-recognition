import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224

MODEL_PATH = "models/unet_hand.h5"
INPUT_DIR = "clean_images"
OUTPUT_DIR = "segmented_images"

# ===============================
# SAFETY CHECKS
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(" U-Net model not found")

if not os.path.exists(INPUT_DIR):
    raise FileNotFoundError(" clean_images folder not found")

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = load_model(MODEL_PATH)

saved_count = 0

# ===============================
# SEGMENTATION LOOP
# ===============================
for label in os.listdir(INPUT_DIR):
    in_class = os.path.join(INPUT_DIR, label)
    out_class = os.path.join(OUTPUT_DIR, label)
    os.makedirs(out_class, exist_ok=True)

    for img_name in os.listdir(in_class):
        img_path = os.path.join(in_class, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_norm = img_resized / 255.0

        pred = model.predict(
            np.expand_dims(img_norm, axis=0),
            verbose=0
        )[0]

        #  RELAX THRESHOLD (IMPORTANT)
        mask = (pred > 0.3).astype(np.uint8)

        #  If mask is empty, skip (avoid black images)
        if np.sum(mask) < 100:
            continue

        segmented = img_resized * mask
        save_path = os.path.join(out_class, img_name)

        cv2.imwrite(save_path, segmented)
        saved_count += 1

print(f" Segmentation completed. Images saved: {saved_count}")
