import os, cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 224

# -------------------------------
# Load images and masks
# -------------------------------
def load_data(img_dir, mask_dir):
    X, y = [], []

    for label in os.listdir(img_dir):
        img_path = os.path.join(img_dir, label)
        mask_path = os.path.join(mask_dir, label)

        if not os.path.exists(mask_path):
            continue

        for img_name in os.listdir(img_path):
            img = cv2.imread(os.path.join(img_path, img_name))
            mask = cv2.imread(os.path.join(mask_path, img_name), 0)

            if img is None or mask is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
            mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE)) / 255.0

            X.append(img)
            y.append(mask.reshape(IMG_SIZE, IMG_SIZE, 1))

    return np.array(X), np.array(y)

X, y = load_data("clean_images", "masks")

print("Images loaded:", X.shape)
print("Masks loaded:", y.shape)

if len(X) == 0:
    raise ValueError(" No training data found")

# -------------------------------
# U-Net model
# -------------------------------
def unet():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))

    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D()(c2)

    b = Conv2D(256, 3, activation='relu', padding='same')(p2)

    u1 = UpSampling2D()(b)
    u1 = Concatenate()([u1, c2])
    c3 = Conv2D(128, 3, activation='relu', padding='same')(u1)

    u2 = UpSampling2D()(c3)
    u2 = Concatenate()([u2, c1])
    c4 = Conv2D(64, 3, activation='relu', padding='same')(u2)

    outputs = Conv2D(1, 1, activation='sigmoid')(c4)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = unet()
model.fit(X, y, epochs=10, batch_size=4, validation_split=0.1)

# -------------------------------
# SAVE MODEL (IMPORTANT)
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/unet_hand.h5")

print("U-Net model saved at models/unet_hand.h5")
