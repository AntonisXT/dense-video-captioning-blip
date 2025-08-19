# utils/outlier_filter.py

import numpy as np
import torch
import logging
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from config import get_processing_config
from utils.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

class OutlierFilter:
    """
    Detects and removes semantic outlier captions based on duration,
    Local Outlier Factor, and cosine similarity clustering.
    """

    def __init__(self):
        self.embedding_cache = EmbeddingCache()
        self.config = get_processing_config()

    def _ensure_dict_list(self, captions):
        """Ensure output is a list of dicts with caption/time keys."""
        if not isinstance(captions, list):
            return []
        fixed = []
        for c in captions:
            if isinstance(c, dict) and "caption" in c:
                fixed.append(c)
        return fixed

    def filter_by_lof(
        self, captions,
        duration_threshold=None, lof_threshold=None,
        sim_threshold=None, sim_protect_threshold=None
    ):
        if not captions:
            return []
        if len(captions) < 4:
            return self._ensure_dict_list(captions)

        duration_threshold = duration_threshold or self.config.lof_duration_threshold
        lof_threshold = lof_threshold or self.config.lof_threshold
        sim_threshold = sim_threshold or self.config.lof_sim_threshold
        sim_protect_threshold = sim_protect_threshold or self.config.lof_sim_protect_threshold

        try:
            texts = [c["caption"].strip().lower() for c in captions]
            embeddings = self.embedding_cache.encode_sentences(texts)
            num_points = len(embeddings)

            if num_points <= 6:
                n_neighbors = 2
            elif num_points <= 10:
                n_neighbors = 3
            else:
                n_neighbors = 5

            lof = LocalOutlierFactor(
                n_neighbors=min(n_neighbors, num_points - 1),
                metric="cosine"
            )
            lof.fit(embeddings.cpu().numpy())
            lof_scores = lof.negative_outlier_factor_
            mean_embedding = embeddings.mean(dim=0)

            filtered = []
            for i, cap in enumerate(captions):
                duration = cap["end_time"] - cap["start_time"]
                emb = embeddings[i]
                sim = torch.nn.functional.cosine_similarity(emb, mean_embedding, dim=0).item()
                score = lof_scores[i]

                is_outlier = (
                    duration <= duration_threshold and
                    sim < sim_protect_threshold and
                    (score < lof_threshold or sim < sim_threshold)
                )
                if not is_outlier:
                    filtered.append(cap)

            return self._ensure_dict_list(filtered)

        except Exception as e:
            logger.error(f"⚠️ LOF filtering failed: {e}")
            return self._ensure_dict_list(captions)

    def filter_by_clustering(self, captions, eps=None, min_samples=None):
        if not captions:
            return []
        # Always enforce dict list shape
        captions = self._ensure_dict_list(captions)
        if not captions:
            return []

        if len(captions) <= 3:
            try:
                texts = [c["caption"].strip().lower() for c in captions]
                embeddings = self.embedding_cache.encode_sentences(texts)
                mean_emb = embeddings.mean(dim=0)
                sims = torch.nn.functional.cosine_similarity(embeddings, mean_emb.unsqueeze(0)).squeeze()
                durations = torch.tensor(
                    [c["end_time"] - c["start_time"] for c in captions],
                    dtype=torch.float32, device=sims.device
                )
                scores = sims + 0.05 * torch.log1p(durations)
                best_idx = int(torch.argmax(scores).item())
                # Return a list with the chosen dict
                return [captions[best_idx]]
            except Exception as e:
                logger.error(f"⚠️ Fallback clustering failed: {e}")
                return captions

        eps = eps or self.config.dbscan_eps
        min_samples = min_samples or self.config.dbscan_min_samples

        try:
            texts = [c["caption"].strip().lower() for c in captions]
            embeddings = self.embedding_cache.encode_sentences(texts)
            embeddings_np = embeddings.cpu().numpy()
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings_np)
            labels = clustering.labels_
            valid_labels = labels[labels >= 0]

            if len(valid_labels) == 0:
                return captions

            main_cluster = np.argmax(np.bincount(valid_labels))
            filtered = [c for c, label in zip(captions, labels) if label == main_cluster]
            return self._ensure_dict_list(filtered)

        except Exception as e:
            logger.error(f"⚠️ DBSCAN clustering failed: {e}")
            return captions
