import cv2
import os

video_path = "C:/Users/andre/PycharmProjects/PC3D/data/videos/speaker.mp4"
output_folder = "data/videos"
  # Extract every 5th frame (adjust as needed)

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
saved_count = 0
_,frame = cap.read()
frame_skip = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_id % frame_skip == 0:
        filename = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_count += 1
    frame_id += 1

cap.release()
print(f"Extracted {saved_count} frames to {output_folder}/")
