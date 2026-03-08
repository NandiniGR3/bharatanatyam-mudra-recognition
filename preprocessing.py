import cv2
import os
import numpy as np
from tqdm import tqdm

INPUT_DIR = "raw_frames"       
OUTPUT_DIR = "clean_images"

IMG_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)

for cls in os.listdir(INPUT_DIR):
    in_class = os.path.join(INPUT_DIR, cls)
    out_class = os.path.join(OUTPUT_DIR, cls)

    os.makedirs(out_class, exist_ok=True)

    print(f"\nProcessing class: {cls}")

    for img_name in tqdm(os.listdir(in_class)):
        img_path = os.path.join(in_class, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        # -------------------------------
        # 1. Resize
        # -------------------------------
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # -------------------------------
        # 2. Blur removal (variance of Laplacian)
        # -------------------------------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_score < 100:   # threshold
            continue

        # -------------------------------
        # 3. Normalize
        # -------------------------------
        img = img.astype(np.float32) / 255.0
        img = (img * 255).astype(np.uint8)

        # -------------------------------
        # 4. SAVE IMAGE (CRITICAL LINE)
        # -------------------------------
        save_path = os.path.join(out_class, img_name)
        cv2.imwrite(save_path, img)

print("\n Image preprocessing completed successfully!")
