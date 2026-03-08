import cv2
import os
import numpy as np

IMG_DIR = "clean_images"
MASK_DIR = "masks"

os.makedirs(MASK_DIR, exist_ok=True)

for cls in os.listdir(IMG_DIR):
    img_cls = os.path.join(IMG_DIR, cls)
    mask_cls = os.path.join(MASK_DIR, cls)
    os.makedirs(mask_cls, exist_ok=True)

    for img_name in os.listdir(img_cls):
        img_path = os.path.join(img_cls, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 20, 70])
        upper = np.array([20, 255, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # SAME NAME AS IMAGE
        cv2.imwrite(os.path.join(mask_cls, img_name), mask)

print("Masks regenerated correctly")
