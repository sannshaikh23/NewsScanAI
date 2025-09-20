import cv2
import os

def extract_frames(video_path, output_folder="data/results/frames"):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{idx}.jpg")
        cv2.imwrite(frame_path, frame)
        frames.append(frame_path)
        idx += 30
        cap.set(1, idx)
    cap.release()
    return frames
