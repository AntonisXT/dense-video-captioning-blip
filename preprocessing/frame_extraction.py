# preprocessing/frame_extraction.py
import os
import cv2


def extract_frames(video_path, output_folder, interval=1):
    """
    Extract frames from a video at fixed time intervals (e.g., 1 frame/sec).

    Parameters:
        video_path (str): path to video file
        output_folder (str): folder to save extracted frames
        interval (int): time interval in seconds between frames
    """
    os.makedirs(output_folder, exist_ok=True)
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    success, frame = video.read()

    count = 0
    saved_count = 0

    while success:
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        success, frame = video.read()
        count += 1

    video.release()
    print(f"✅ {saved_count} frames extracted from {video_path}")
