import torch
import logging
import clip
from config import get_model_config
from model_registry import get_sbert, get_clip

logger = logging.getLogger(__name__)

def _resolve_device(pref_device=None):
    """
    Resolve a safe torch.device:
    - Use pref_device if provided (torch.device or str)
    - Else use config.model.device if valid
    - Else cuda if available, otherwise cpu
    """
    if isinstance(pref_device, torch.device):
        return pref_device
    if isinstance(pref_device, str):
        try:
            return torch.device(pref_device)
        except Exception:
            pass

    # fallback to config
    try:
        mc = get_model_config()
        if isinstance(mc.device, str):
            return torch.device(mc.device if (mc.device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    except Exception:
        pass

    # final fallback
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingCache:
    """
    Caches and batches sentence and CLIP text embeddings to avoid redundant computation.
    Uses shared model instances via model_registry to prevent repeated loads.
    """

    def __init__(self, device=None):
        self.device = _resolve_device(device)

        # Shared instances from registry
        try:
            self.semantic_model = get_sbert()
        except Exception as e:
            logger.error(f"❌ Failed getting shared SentenceTransformer: {e}")
            self.semantic_model = None

        try:
            self.clip_model, _ = get_clip()
        except Exception as e:
            logger.error(f"❌ Failed getting shared CLIP model: {e}")
            self.clip_model = None

        # CPU-side caches (store on CPU; move to device on demand)
        self._semantic_cache = {}
        self._clip_text_cache = {}

    def encode_sentences(self, texts):
        if not texts:
            return torch.empty(0, device=self.device)

        to_encode = [t for t in texts if t not in self._semantic_cache]
        if to_encode and self.semantic_model:
            try:
                new_embeddings = self.semantic_model.encode(to_encode, convert_to_tensor=True)
                for t, emb in zip(to_encode, new_embeddings):
                    self._semantic_cache[t] = emb.detach().cpu()
            except Exception as e:
                logger.error(f"⚠️ Sentence embedding failed: {e}")
                # Guess dimension (MiniLM-L6-v2 -> 384)
                for t in to_encode:
                    self._semantic_cache[t] = torch.zeros(384)

        batch = torch.stack([self._semantic_cache[t] for t in texts]).to(self.device)
        return batch

    def encode_clip_texts(self, texts):
        if not texts:
            return torch.empty(0, device=self.device)

        to_encode = [t for t in texts if t not in self._clip_text_cache]
        if to_encode and self.clip_model:
            try:
                tokens = clip.tokenize(to_encode).to(self.device)
                # ensure model device alignment
                self.clip_model.to(self.device)
                with torch.no_grad():
                    new_embeddings = self.clip_model.encode_text(tokens)
                for t, emb in zip(to_encode, new_embeddings):
                    self._clip_text_cache[t] = emb.detach().cpu()
            except Exception as e:
                logger.error(f"⚠️ CLIP text embedding failed: {e}")
                # CLIP ViT-B/32 text dim is 512
                for t in to_encode:
                    self._clip_text_cache[t] = torch.zeros(512)

        batch = torch.stack([self._clip_text_cache[t] for t in texts]).to(self.device)
        return batch

    def clip_similarity(self, clip_embeddings, video_embedding):
        """
        clip_embeddings: [N, D] on some device
        video_embedding: [D] (CPU usually) — align to clip_embeddings.device and dtype
        """
        try:
            if video_embedding is None or clip_embeddings.numel() == 0:
                return torch.zeros(len(clip_embeddings), device=clip_embeddings.device)

            ve = video_embedding
            if not isinstance(ve, torch.Tensor):
                ve = torch.tensor(ve)

            ve = ve.to(clip_embeddings.device)
            if ve.dtype != clip_embeddings.dtype:
                ve = ve.to(dtype=clip_embeddings.dtype)

            return torch.nn.functional.cosine_similarity(clip_embeddings, ve.unsqueeze(0)).squeeze()
        except Exception as e:
            logger.error(f"⚠️ CLIP similarity computation failed: {e}")
            return torch.zeros(len(clip_embeddings), device=clip_embeddings.device)

    def clear(self):
        self._semantic_cache.clear()
        self._clip_text_cache.clear()
        logger.info("Embedding cache cleared.")
