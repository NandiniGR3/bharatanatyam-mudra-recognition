import cv2
import tensorflow as tf
import numpy as np
import os
import mediapipe as mp

MODEL_PATH = "models/cnn_mudra_model.h5"
IMG_SIZE = 224

# Load CNN model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
DATASET_PATH = "clean_images"
class_names = sorted([
    f for f in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, f))
])

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Start camera
cap = cv2.VideoCapture(0)

print("Press Q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    label = "No Hand Detected"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            h, w, c = frame.shape
            x_list = []
            y_list = []

            # Draw red dots on palm
            for lm in hand_landmarks.landmark:

                cx = int(lm.x * w)
                cy = int(lm.y * h)

                x_list.append(cx)
                y_list.append(cy)

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

            mp_draw.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)

            # Crop hand region
            xmin = max(min(x_list)-20,0)
            xmax = min(max(x_list)+20,w)
            ymin = max(min(y_list)-20,0)
            ymax = min(max(y_list)+20,h)

            hand_img = frame[ymin:ymax, xmin:xmax]

            if hand_img.size != 0:

                try:
                    img = cv2.resize(hand_img,(IMG_SIZE,IMG_SIZE))
                    img = img.astype("float32")/255.0
                    img = np.expand_dims(img,axis=0)

                    preds = model.predict(img,verbose=0)

                    class_id = np.argmax(preds)
                    confidence = preds[0][class_id]

                    label = f"{class_names[class_id]} : {confidence*100:.2f}%"

                except:
                    label = "Prediction Error"

    cv2.putText(frame,label,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Bharatanatyam Mudra Recognition",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()