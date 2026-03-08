import cv2
import os

VIDEO_DIR = "videos"
FRAME_DIR = "raw_frames"
FRAME_RATE = 5  # extract 1 frame every 5 frames

os.makedirs(FRAME_DIR, exist_ok=True)

for video_file in os.listdir(VIDEO_DIR):
    label = video_file.split(".")[0]
    os.makedirs(f"{FRAME_DIR}/{label}", exist_ok=True)

    cap = cv2.VideoCapture(f"{VIDEO_DIR}/{video_file}")
    count = 0
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_RATE == 0:
            frame_path = f"{FRAME_DIR}/{label}/{label}_{frame_id}.jpg"
            cv2.imwrite(frame_path, frame)
            frame_id += 1

        count += 1

    cap.release()
print("Frame extraction completed.")