# models/frame_selector.py

import os
import re
import torch
import clip
from PIL import Image
from sentence_transformers import util
from config import get_motion_keywords
from utils.text_utils import clean_caption
from utils.embedding_cache import EmbeddingCache
from model_registry import get_blip, get_clip, get_sbert
import logging

logger = logging.getLogger(__name__)

class FrameSelector:
    """
    Selects the most representative frame caption for a video scene using:
    - Batch BLIP captioning for speed
    - Motion keyword detection
    - Semantic fusion of multiple captions
    - CLIP similarity with global video embedding (optional)
    """

    def __init__(self, batch_size: int = 8):
        self.batch_size = batch_size

        # Shared models from registry
        self.processor, self.blip_model = get_blip()   # BLIP on shared device
        self.clip_model, _ = get_clip()               # CLIP shared
        self.semantic_model = get_sbert()             # SBERT shared

        # Determine runtime device from model parameters for perfect match
        self.device = next(self.blip_model.parameters()).device

        self.embedding_cache = EmbeddingCache(device=self.device)
        self.motion_keywords = set(get_motion_keywords())

    def _generate_captions_batch(self, frame_paths):
        images = []
        valid_paths = []
        for p in frame_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load frame {p}: {e}")

        if not images:
            return {}

        try:
            inputs = self.processor(images, return_tensors="pt", padding=True)
            # Move inputs to the exact device of the BLIP model
            inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

            # Ensure model is on the same device
            self.blip_model.to(self.device)

            outputs = self.blip_model.generate(**inputs, max_new_tokens=35)
            captions = [clean_caption(self.processor.decode(o, skip_special_tokens=True)) for o in outputs]
            return {p: c for p, c in zip(valid_paths, captions)}
        except Exception as e:
            logger.error(f"❌ BLIP batch captioning failed: {e}")
            return {p: "unavailable frame" for p in valid_paths}

    def _contains_motion(self, caption: str) -> bool:
        return any(word in caption.lower() for word in self.motion_keywords)

    def _semantic_fusion(self, captions):
        if not captions:
            return "No caption"
        if len(set(captions)) == 1:
            return captions[0]

        embeddings = self.semantic_model.encode(captions, convert_to_tensor=True)
        mean_emb = embeddings.mean(dim=0)
        sims = util.cos_sim(embeddings, mean_emb).squeeze()

        def score(caption, idx):
            unique_words = len(set(re.findall(r'\b\w+\b', caption.lower())))
            return sims[idx].item() + 0.01 * unique_words

        scores = [score(c, i) for i, c in enumerate(captions)]
        return captions[int(torch.tensor(scores).argmax())]

    def select_best_caption(self, scene, frame_files, frames_folder, clip_video_embedding=None):
        start, end = scene["start_frame"], scene["end_frame"]
        mid = (start + end) // 2
        frame_indices = list(set([start, mid, end]))

        candidate_paths = [
            os.path.join(frames_folder, frame_files[i])
            for i in frame_indices if 0 <= i < len(frame_files)
        ]

        if not candidate_paths:
            return "No valid frames"

        captions_dict = self._generate_captions_batch(candidate_paths)
        candidates = [(p, captions_dict.get(p, "No caption")) for p in candidate_paths]

        for _, cap in candidates:
            if self._contains_motion(cap):
                return cap

        fused_caption = self._semantic_fusion([c[1] for c in candidates])

        if clip_video_embedding is not None:
            try:
                # Ensure tokens and model on same device
                tokens = clip.tokenize([fused_caption]).to(self.device)
                self.clip_model.to(self.device)
                clip_text_emb = self.clip_model.encode_text(tokens)
                # EmbeddingCache handles device alignment internally
                _ = self.embedding_cache.clip_similarity(clip_text_emb, clip_video_embedding)
                return fused_caption
            except Exception as e:
                logger.warning(f"⚠️ CLIP similarity failed: {e}")
                return max(candidates, key=lambda x: len(set(x[1].split())))[1]

        return fused_caption
