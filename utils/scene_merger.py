# utils/scene_merger.py

import re
import torch
import logging
from collections import Counter
from sentence_transformers import util
from config import get_processing_config, get_motion_keywords
from utils.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

EXPECTED_DIM = 384  # all-MiniLM-L6-v2

class SceneMerger:
    """
    Merges temporally adjacent video scenes with semantically similar captions.
    Uses sentence similarity, motion keywords, and caption frequency.
    """

    def __init__(self, similarity_threshold=None, short_scene_threshold=None):
        config = get_processing_config()
        self.similarity_threshold = similarity_threshold or config.similarity_threshold
        self.short_scene_threshold = short_scene_threshold or config.short_scene_threshold
        self.embedding_cache = EmbeddingCache()
        self.motion_keywords = set(get_motion_keywords())

    def _contains_motion(self, caption):
        caption = caption.lower()
        return any(word in caption for word in self.motion_keywords)

    def _count_unique_keywords(self, caption):
        words = re.findall(r'\b\w+\b', caption.lower())
        return len(set(words))

    def _select_best_caption(self, c1, c2, caption_freq):
        freq1 = caption_freq.get(c1.lower(), 0)
        freq2 = caption_freq.get(c2.lower(), 0)
        if freq1 > freq2:
            return c1
        if freq2 > freq2:
            return c2
        score1 = self._count_unique_keywords(c1) + (1 if self._contains_motion(c1) else 0)
        score2 = self._count_unique_keywords(c2) + (1 if self._contains_motion(c2) else 0)
        return c1 if score1 >= score2 else c2

    def _safe_as_row(self, emb):
        """
        Ensure embedding is a [1, D] tensor with expected D. Return None if not valid.
        """
        if not isinstance(emb, torch.Tensor):
            return None
        if emb.dim() == 1:
            emb = emb.reshape(1, -1)
        elif emb.dim() > 2:
            # collapse extra dims if possible
            emb = emb.view(1, -1)
        if emb.size(1) != EXPECTED_DIM:
            return None
        return emb

    def merge(self, captions):
        if not captions:
            return []

        try:
            caption_freq = Counter(c["caption"].strip().lower() for c in captions)
            all_texts = [c["caption"] for c in captions]
            all_embeddings = self.embedding_cache.encode_sentences(all_texts)  # [N, D]

            if all_embeddings.dim() != 2:
                logger.warning(f"⚠️ Unexpected embeddings dim: {all_embeddings.shape}")
                return captions

            merged = [captions[0]]
            last_emb = self._safe_as_row(all_embeddings)
            if last_emb is None:
                logger.warning(f"⚠️ Embedding dim mismatch at scene 0: got {all_embeddings.numel()} dims")
                # fabricate a zero vector [1, 384] to continue deterministically
                last_emb = torch.zeros(1, EXPECTED_DIM, device=all_embeddings.device)

            for i in range(1, len(captions)):
                current = captions[i]
                last = merged[-1]
                current_raw = all_embeddings[i]
                current_emb = self._safe_as_row(current_raw)

                sim = 0.0
                if current_emb is None:
                    logger.warning(f"⚠️ Embedding dim mismatch at scene {i}: {current_raw.shape[-1]} vs {EXPECTED_DIM}")
                else:
                    try:
                        sim_matrix = util.cos_sim(last_emb, current_emb)  # [1,1]
                        sim = float(sim_matrix[0, 0].item())
                    except Exception as e:
                        logger.warning(f"⚠️ Similarity computation failed at scene {i}: {e}")
                        sim = 0.0

                duration = current["end_time"] - current["start_time"]
                time_gap = current["start_time"] - last["end_time"]

                should_merge = (
                    sim >= self.similarity_threshold
                    or (duration <= self.short_scene_threshold and sim >= 0.55 and time_gap <= 1)
                )

                if should_merge:
                    old_caption = last["caption"]
                    last["end_time"] = current["end_time"]
                    last["caption"] = self._select_best_caption(old_caption, current["caption"], caption_freq)
                    try:
                        re_emb = self.embedding_cache.encode_sentences([last["caption"]])
                        if re_emb.dim() == 2 and re_emb.size(1) == EXPECTED_DIM:
                            last_emb = re_emb
                        else:
                            last_emb = torch.zeros(1, EXPECTED_DIM, device=all_embeddings.device)
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to re-embed merged caption: {e}")
                else:
                    merged.append(current)
                    last_emb = current_emb if current_emb is not None else torch.zeros(1, EXPECTED_DIM, device=all_embeddings.device)

            for idx, scene in enumerate(merged, start=1):
                scene["scene_id"] = idx

            return merged

        except Exception as e:
            logger.error(f"❌ Scene merge failed: {e}")
            return captions
