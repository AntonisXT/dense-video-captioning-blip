# model_registry.py

import logging
import torch
import clip
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from config import get_model_config

logger = logging.getLogger(__name__)

class _ModelRegistry:
    """
    Shared, lazily-loaded singleton instances for all core models.
    Prevents repeated loads and reduces startup overhead.
    """
    def __init__(self):
        self._device = None
        self._blip_processor = None
        self._blip_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._sbert = None
        self._t5_tokenizer = None
        self._t5_model = None

    @property
    def device(self):
        if self._device is None:
            model_config = get_model_config()
            self._device = torch.device("cuda" if torch.cuda.is_available() and str(model_config.device).startswith("cuda") else "cpu")
        return self._device

    def get_blip(self):
        """
        Returns (processor, model) for BLIP. We enforce use_fast=False to keep outputs stable.
        """
        if self._blip_processor is None or self._blip_model is None:
            mc = get_model_config()
            try:
                # Keep outputs stable: explicitly use slow processor
                self._blip_processor = BlipProcessor.from_pretrained(mc.blip_model, use_fast=False)
                self._blip_model = BlipForConditionalGeneration.from_pretrained(mc.blip_model).to(self.device)
                logger.info(f"BLIP loaded: {mc.blip_model}")
            except Exception as e:
                logger.error(f"Failed loading BLIP: {e}")
                raise
        return self._blip_processor, self._blip_model

    def get_clip(self):
        """
        Returns (clip_model, preprocess)
        """
        if self._clip_model is None or self._clip_preprocess is None:
            mc = get_model_config()
            try:
                self._clip_model, self._clip_preprocess = clip.load(mc.clip_model, device=self.device)
                logger.info(f"CLIP loaded: {mc.clip_model}")
            except Exception as e:
                logger.error(f"Failed loading CLIP: {e}")
                raise
        return self._clip_model, self._clip_preprocess

    def get_sbert(self):
        """
        Returns SentenceTransformer encoder (shared)
        """
        if self._sbert is None:
            mc = get_model_config()
            try:
                self._sbert = SentenceTransformer(mc.sentence_transformer, device=str(self.device))
                logger.info(f"SentenceTransformer loaded: {mc.sentence_transformer}")
            except Exception as e:
                logger.error(f"Failed loading SentenceTransformer: {e}")
                raise
        return self._sbert

    def get_t5(self):
        """
        Returns (tokenizer, model) for FLAN-T5 summarization
        """
        if self._t5_tokenizer is None or self._t5_model is None:
            mc = get_model_config()
            try:
                self._t5_tokenizer = AutoTokenizer.from_pretrained(mc.flan_t5_model)
                self._t5_model = AutoModelForSeq2SeqLM.from_pretrained(mc.flan_t5_model).to(self.device).eval()
                logger.info(f"T5 loaded: {mc.flan_t5_model}")
            except Exception as e:
                logger.error(f"Failed loading T5: {e}")
                raise
        return self._t5_tokenizer, self._t5_model


# Global singleton accessor
_registry = _ModelRegistry()

def get_device():
    return _registry.device

def get_blip():
    return _registry.get_blip()

def get_clip():
    return _registry.get_clip()

def get_sbert():
    return _registry.get_sbert()

def get_t5():
    return _registry.get_t5()
