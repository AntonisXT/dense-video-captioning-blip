# main.py

import os
import re
import shlex
import json
import subprocess
import logging
import cv2

from config import get_app_config, get_processing_config
from preprocessing.frame_extraction import extract_frames
from preprocessing.scene_detector import SceneDetector
from utils.subtitle_generator import generate_srt
from utils.scene_merger import SceneMerger
from utils.outlier_filter import OutlierFilter
from models.clip_video_encoder import extract_clip_video_embedding
from models.frame_selector import FrameSelector
from models.summary_generator import SummaryGenerator
from utils.text_utils import fix_overlapping_scenes

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("pipeline.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)


class VideoCaptioning:
    """
    Video captioning pipeline coordinator. Handles:
    - Frame extraction
    - Scene segmentation
    - Frame-level captioning
    - Semantic scene merging
    - Summary generation
    - Optional subtitle embedding
    """

    def __init__(self, folder_name, generate_subtitles=True):
        self.folder_name = folder_name
        self.generate_subtitles = generate_subtitles

        # Configs
        self.app_config = get_app_config()
        self.processing_config = get_processing_config()

        # Init components (reused for all videos)
        self.frame_selector = FrameSelector()
        self.scene_detector = SceneDetector()
        self.scene_merger = SceneMerger()
        self.summary_generator = SummaryGenerator()
        self.outlier_filter = OutlierFilter()

    def get_video_paths(self):
        """Get list of video file paths from a folder."""
        base_folder = os.path.join(self.app_config.videos_dir, self.folder_name)

        if not os.path.isdir(base_folder):
            raise ValueError(f"❌ Folder not found: {base_folder}")

        return sorted([
            os.path.join(base_folder, f)
            for f in os.listdir(base_folder)
            if f.lower().endswith(tuple(f".{ext}" for ext in self.app_config.supported_formats))
        ])

    @staticmethod
    def get_video_name(path):
        """Extract file name without extension."""
        return os.path.splitext(os.path.basename(path))[0]

    def save_results_as_json(self, video_id, summary_caption, scene_captions, output_folder):
        """Save the generated captions and summary to a JSON file."""
        try:
            result = {
                "video_id": video_id,
                "summary": summary_caption,
                "scene_captions": [
                    {
                        "start_time": sc["start_time"],
                        "end_time": sc["end_time"],
                        "caption": sc["caption"]
                    } for sc in scene_captions
                ]
            }
            json_path = os.path.join(output_folder, f"{video_id}_captions.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ Saved results: {json_path}")
        except Exception as e:
            logger.error(f"❌ Failed saving JSON results for {video_id}: {e}")

    def process_video(self, video_path):
        """Complete processing of a single video file with error handling."""
        video_name = self.get_video_name(video_path)
        logger.info(f"\n🎬 Processing video: {video_name}")

        # Setup output folders
        frames_output = os.path.join(self.app_config.frames_dir, video_name)
        base_output = os.path.join(self.app_config.results_dir, self.folder_name, video_name)
        captions_output = os.path.join(base_output, "captions")
        subtitled_output = os.path.join(base_output, "subtitled")

        os.makedirs(frames_output, exist_ok=True)
        os.makedirs(captions_output, exist_ok=True)
        if self.generate_subtitles:
            os.makedirs(subtitled_output, exist_ok=True)

        # Step 1: Extract frames
        try:
            logger.info("🔹 Extracting frames...")
            extract_frames(video_path, frames_output, interval=self.processing_config.frame_interval)
        except Exception as e:
            logger.error(f"❌ Frame extraction failed: {e}")
            return

        # Step 2: Detect scenes
        try:
            logger.info("🔹 Detecting scenes...")
            scenes = self.scene_detector.detect(video_path, frames_output)
        except Exception as e:
            logger.error(f"❌ Scene detection failed: {e}")
            return

        # Step 3: Global CLIP embedding
        try:
            logger.info("🔹 Extracting global CLIP embedding...")
            clip_video_embedding = extract_clip_video_embedding(frames_output, max_frames=self.processing_config.max_frames)
        except Exception as e:
            logger.warning(f"⚠️ Failed extracting CLIP embedding: {e}")
            clip_video_embedding = None

        # Step 4: Captions per scene
        scene_captions = []
        try:
            logger.info("🔹 Generating scene captions...")
            frame_files = sorted(os.listdir(frames_output), key=lambda x: int(re.findall(r'\d+', x)[0]))
            for scene in scenes:
                try:
                    caption = self.frame_selector.select_best_caption(
                        scene,
                        frame_files,
                        frames_output,
                        clip_video_embedding
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Failed captioning scene {scene['scene_id']}: {e}")
                    caption = "Scene could not be captioned."
                scene_captions.append({
                    "scene_id": scene["scene_id"],
                    "start_time": scene["start_time"],
                    "end_time": scene["end_time"],
                    "caption": caption
                })
                logger.info(f"   ▶ Scene {scene['scene_id']} [{scene['start_time']}s - {scene['end_time']}s]: {caption}")
        except Exception as e:
            logger.error(f"❌ Scene captioning failed: {e}")
            return

        # Step 5: Merge scenes
        try:
            logger.info("🔹 Merging scenes...")
            scene_captions = self.outlier_filter.filter_by_lof(scene_captions)
            scene_captions = self.scene_merger.merge(scene_captions)
            scene_captions = fix_overlapping_scenes(scene_captions)
            scene_captions = sorted(scene_captions, key=lambda x: x["start_time"])
        except Exception as e:
            logger.warning(f"⚠️ Could not merge scenes: {e}")

        # Step 6: Generate summary
        try:
            logger.info("🔹 Generating summary caption...")
            summary = self.summary_generator.generate(
                scene_captions,
                clip_video_embedding,
                max_length=self.processing_config.max_caption_length
            )
            logger.info(f"✅ Video Summary: {summary}")
        except Exception as e:
            logger.error(f"❌ Failed summary generation: {e}")
            summary = "Unable to summarize video."

        # Step 7: Save results
        self.save_results_as_json(video_name, summary, scene_captions, captions_output)

        # Step 8: Optional: embed subtitles
        if self.generate_subtitles:
            try:
                srt_path = os.path.join(subtitled_output, f"{video_name}.srt")
                generate_srt(scene_captions, srt_path)

                output_video = os.path.join(subtitled_output, f"{video_name}_subtitled.mp4")
                ffmpeg_command = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-vf", f"subtitles={shlex.quote(srt_path.replace(os.sep, '/'))}",
                    "-c:a", "copy", output_video
                ]
                subprocess.run(ffmpeg_command, check=True)
                logger.info(f"✅ Subtitled video saved: {output_video}")
            except Exception as e:
                logger.warning(f"⚠️ Failed embedding subtitles: {e}")

    def run(self):
        """Run the full pipeline for all videos in folder."""
        try:
            video_paths = self.get_video_paths()
        except ValueError as e:
            logger.error(e)
            return

        for video_path in video_paths:
            self.process_video(video_path)


if __name__ == "__main__":
    folder_name = input("📂 Enter the folder name containing the videos: ").strip()
    generate_subtitles = input("🎬 Generate videos with embedded subtitles? (yes/no): ").strip().lower() == "yes"
    pipeline = VideoCaptioning(folder_name, generate_subtitles)
    pipeline.run()
