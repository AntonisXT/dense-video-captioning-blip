# models/summary_generator.py

import torch
import logging
from config import get_processing_config
from utils.text_utils import clean_caption
from utils.outlier_filter import OutlierFilter
from utils.embedding_cache import EmbeddingCache
from model_registry import get_t5

logger = logging.getLogger(__name__)

class SummaryGenerator:
    """
    Generates a video-level summary sentence using scene captions.
    Applies semantic filtering, CLIP-based scoring, and language modeling via FLAN-T5.
    """

    def __init__(self, model_name=None):
        try:
            self.tokenizer, self.model = get_t5()
            self.model_device = next(self.model.parameters()).device
            self.outlier_filter = OutlierFilter()
            self.embedding_cache = EmbeddingCache(device=self.model_device)
        except Exception as e:
            logger.error(f"❌ Failed to initialize SummaryGenerator: {e}")
            raise

    def _ensure_dict_list(self, items):
        """Keep only dicts with 'caption' key."""
        if not isinstance(items, list):
            return []
        return [c for c in items if isinstance(c, dict) and "caption" in c and isinstance(c["caption"], str)]

    def _normalize_captions(self, items):
        """
        Normalize arbitrary items into list[dict] with keys:
        - caption (str)
        - start_time/end_time (optional, default 0)
        """
        if not isinstance(items, list):
            return []
        normalized = []
        for c in items:
            if isinstance(c, dict) and "caption" in c and isinstance(c["caption"], str):
                # ensure timestamps exist
                start = int(c.get("start_time", 0) or 0)
                end = int(c.get("end_time", max(start + 1, 1)) or max(start + 1, 1))
                normalized.append({"caption": c["caption"], "start_time": start, "end_time": end})
            elif isinstance(c, str):
                normalized.append({"caption": c, "start_time": 0, "end_time": 1})
        return normalized

    def _score_captions(self, captions, clip_video_embedding):
        texts = [c["caption"].strip().lower() for c in captions]
        durations = [max(0, (c["end_time"] - c["start_time"])) for c in captions]

        embeddings = self.embedding_cache.encode_sentences(texts)
        semantic_scores = torch.nn.functional.cosine_similarity(
            embeddings, embeddings.mean(dim=0).unsqueeze(0)
        )

        if clip_video_embedding is not None:
            clip_embeddings = self.embedding_cache.encode_clip_texts(texts)
            clip_scores = self.embedding_cache.clip_similarity(clip_embeddings, clip_video_embedding)
        else:
            clip_scores = torch.zeros(len(texts), dtype=torch.float32, device=embeddings.device)

        durations_tensor = torch.tensor(durations, dtype=torch.float32, device=embeddings.device)
        total_scores = 0.6 * semantic_scores + 0.3 * clip_scores + 0.1 * torch.log1p(durations_tensor)
        return texts, total_scores

    def _generate_single(self, prompt, max_length):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model_device)
        out = self.model.generate(input_ids, max_length=max_length, min_length=5, num_beams=4)
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

    def _fallback_summary(self, captions, max_length):
        sorted_caps = sorted(captions, key=lambda x: x["start_time"])
        labels = ["First", "Next", "Finally"]
        try:
            if len(sorted_caps) == 2:
                prompts = [
                    f"Summarize the video in one short and natural sentence:\n{cap['caption'].rstrip('.')}"
                    for cap in sorted_caps
                ]
                outputs = [self._generate_single(p, max_length) for p in prompts]
                return f"{outputs} and then {outputs[1].lower()}"
            phrases = []
            for label, cap in zip(labels, sorted_caps):
                prompt = f"Summarize the scene in one natural sentence:\n{cap['caption'].rstrip('.')}"
                text = self._generate_single(prompt, max_length)
                phrases.append(f"{label}, {text}")
            return " ".join(phrases)
        except Exception as e:
            logger.error(f"❌ Fallback summary generation failed: {e}")
            return "a video."

    @torch.no_grad()
    def generate(self, captions, clip_video_embedding=None, max_length=None):
        try:
            if max_length is None:
                max_length = get_processing_config().max_caption_length
            if not captions:
                return "a video"

            # Normalize inputs to list[dict]
            valid = self._normalize_captions(captions)
            if not valid:
                return "a video"

            texts = [c["caption"].strip().lower() for c in valid]
            embeddings = self.embedding_cache.encode_sentences(texts)
            if embeddings.numel() == 0:
                return "a video"

            sim_matrix = torch.nn.functional.cosine_similarity(
                embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1
            )
            mean_sim = sim_matrix.mean().item()

            if len(valid) <= 3 and mean_sim < 0.4:
                return self._fallback_summary(valid, max_length)

            # Cluster-based filtering
            filtered = self.outlier_filter.filter_by_clustering(valid)
            filtered = self._normalize_captions(filtered)
            if not filtered:
                return "a video"

            if len(filtered) == 1 or len(set(c["caption"].lower() for c in filtered)) == 1:
                return clean_caption(filtered[0]["caption"])

            texts, scores = self._score_captions(filtered, clip_video_embedding)
            top_k = min(5, len(texts))
            top_indices = torch.topk(scores, k=top_k).indices.tolist()
            selected = list(dict.fromkeys([texts[i] for i in top_indices]))

            prompt_input = ". ".join([s.rstrip(".") for s in selected]) + "."
            prompt = (
                "Summarize the video in one short and natural sentence based on these scene descriptions:\n"
                f"{prompt_input}"
            )
            return clean_caption(self._generate_single(prompt, max_length).lower())
        except Exception as e:
            logger.error(f"❌ Summary generation failed: {e}")
            return "a video"
