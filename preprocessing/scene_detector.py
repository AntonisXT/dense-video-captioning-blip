# preprocessing/scene_detector.py
import os
import re
import cv2
import numpy as np
import imagehash
from PIL import Image
from config import get_processing_config


def natural_sort_key(s):
    """Extract numeric index from filename for proper sorting."""
    return int(re.findall(r'\d+', s)[0])


class SceneDetector:
    """
    Detects scenes in a video using perceptual hashing and optical flow (motion).
    Generates scene boundaries with start/end frames and timestamps.
    """
    def __init__(self, hash_threshold=None, high_hash_threshold=None, 
                 motion_threshold=None, min_scene_length=None):
        # Get config values
        config = get_processing_config()
        
        # Use provided values or fall back to config
        self.hash_threshold = hash_threshold or config.hash_threshold
        self.high_hash_threshold = high_hash_threshold or config.high_hash_threshold
        self.motion_threshold = motion_threshold or config.motion_threshold
        self.min_scene_length = min_scene_length or config.min_scene_length

    def detect(self, video_path, frames_folder):
        """
        Detect scene boundaries in a video based on frame similarity and motion.

        Parameters:
            video_path (str): path to video file
            frames_folder (str): folder containing video frames

        Returns:
            list of dicts: [{scene_id, start_frame, end_frame, start_time, end_time}, ...]
        """
        frame_files = sorted(os.listdir(frames_folder), key=natural_sort_key)
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps else len(frame_files)

        scenes = []
        last_hash = None
        last_gray = None
        scene_start_frame = 0

        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_folder, frame_file)

            # Load image for hash
            img = Image.open(frame_path)
            current_hash = imagehash.phash(img)

            # Load image for motion
            current_img = cv2.imread(frame_path)
            gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)

            hash_diff = 0
            motion_diff = 0

            if last_hash is not None:
                hash_diff = abs(current_hash - last_hash)

            if last_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    last_gray, gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_diff = np.mean(mag)

            # Decide if this frame starts a new scene
            should_cut = False
            if last_hash is not None and last_gray is not None:
                if (hash_diff >= self.hash_threshold and motion_diff < self.motion_threshold) or \
                   hash_diff >= self.high_hash_threshold:
                    should_cut = True

            if should_cut:
                if i - scene_start_frame >= self.min_scene_length:
                    start_time = int(round((scene_start_frame / len(frame_files)) * duration))
                    end_time = int(round(((i - 1) / len(frame_files)) * duration))
                    if end_time <= start_time:
                        end_time = start_time + 1
                    scenes.append({
                        "scene_id": len(scenes) + 1,
                        "start_frame": scene_start_frame,
                        "end_frame": i - 1,
                        "start_time": start_time,
                        "end_time": end_time
                    })
                scene_start_frame = i

            last_hash = current_hash
            last_gray = gray

        # Append final scene
        if len(frame_files) - scene_start_frame >= self.min_scene_length:
            start_time = int(round((scene_start_frame / len(frame_files)) * duration))
            end_time = int(round(((len(frame_files) - 1) / len(frame_files)) * duration))
            if end_time <= start_time:
                end_time = start_time + 1
            scenes.append({
                "scene_id": len(scenes) + 1,
                "start_frame": scene_start_frame,
                "end_frame": len(frame_files) - 1,
                "start_time": start_time,
                "end_time": end_time
            })

        print(f"\n✅ Detected {len(scenes)} scenes:")
        for scene in scenes:
            print(f"Scene {scene['scene_id']} [{scene['start_time']}s - {scene['end_time']}s]")

        return scenes
