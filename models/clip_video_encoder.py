# models/clip_video_encoder.py

import os
import logging
import torch
from PIL import Image
from config import get_processing_config
from model_registry import get_clip

logger = logging.getLogger(__name__)

try:
    _clip_model, _preprocess = get_clip()
except Exception as e:
    logger.error(f"❌ Failed to acquire CLIP from registry: {e}")
    _clip_model, _preprocess = None, None

def extract_clip_video_embedding(frames_folder, max_frames=None):
    if _clip_model is None or _preprocess is None:
        logger.warning("⚠️ CLIP model is not available. Skipping video embedding.")
        return None
    try:
        if max_frames is None:
            max_frames = get_processing_config().max_frames
        if not os.path.isdir(frames_folder):
            logger.warning(f"⚠️ Frames folder does not exist: {frames_folder}")
            return None
        try:
            frame_files = sorted(os.listdir(frames_folder))[:max_frames]
        except Exception as e:
            logger.error(f"⚠️ Failed to list frames in '{frames_folder}': {e}")
            return None
        if not frame_files:
            logger.warning(f"⚠️ No frames found in folder: {frames_folder}")
            return None

        embeddings = []
        device = next(_clip_model.parameters()).device
        for filename in frame_files:
            frame_path = os.path.join(frames_folder, filename)
            try:
                image = Image.open(frame_path).convert("RGB")
                image_input = _preprocess(image).unsqueeze(0).to(device, non_blocking=True)
                with torch.no_grad():
                    image_features = _clip_model.encode_image(image_input)
                embeddings.append(image_features.detach().cpu())
            except Exception as e:
                logger.warning(f"⚠️ Failed to process frame '{filename}': {e}")
                continue

        if not embeddings:
            logger.warning(f"⚠️ All frames failed for folder: {frames_folder}")
            return None

        video_embedding = torch.stack(embeddings).mean(dim=0).squeeze()
        return video_embedding
    except Exception as e:
        logger.error(f"❌ Unexpected error in extract_clip_video_embedding: {e}")
        return None
