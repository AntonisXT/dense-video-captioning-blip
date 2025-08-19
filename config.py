# config.py
"""
Centralized configuration for AI Video Captioning System
All settings and parameters are defined here for easy management
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import os


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # BLIP Model
    blip_model: str = "Salesforce/blip-image-captioning-base"
    
    # CLIP Model
    clip_model: str = "ViT-B/32"
    
    # FLAN-T5 Model
    flan_t5_model: str = "google/flan-t5-base"
    
    # SentenceTransformer Model
    sentence_transformer: str = "all-MiniLM-L6-v2"
    
    # Device configuration
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else "cpu"


@dataclass
class ProcessingConfig:
    """Configuration for video processing parameters"""
    # Frame extraction
    frame_interval: int = 1  # Extract 1 frame per second
    max_frames: int = 32     # Maximum frames for CLIP embedding
    
    # Scene detection
    hash_threshold: int = 10
    high_hash_threshold: int = 20
    motion_threshold: float = 0.5
    min_scene_length: int = 1
    min_scene_duration: float = 2.0
    
    # Scene merging
    similarity_threshold: float = 0.7
    short_scene_threshold: int = 1
    
    # Caption generation
    max_caption_length: int = 35
    
    # Outlier filtering
    lof_duration_threshold: float = 1.0
    lof_threshold: float = -1.4
    lof_sim_threshold: float = 0.3
    lof_sim_protect_threshold: float = 0.45
    
    # DBSCAN clustering
    dbscan_eps: float = 0.45
    dbscan_min_samples: int = 2


@dataclass
class MotionKeywords:
    """Keywords for motion detection in captions"""
    keywords: List[str] = field(default_factory=lambda: [
        "driving", "moving", "running", "walking", "flying", 
        "jumping", "dancing", "swimming", "riding", "rolling", 
        "sliding", "traveling", "cruising", "accelerating", 
        "walking down", "driving down"
    ])


@dataclass
class AppConfig:
    """Configuration for Streamlit application"""
    # App metadata
    title: str = "AI Video Captioning System"
    description: str = "Advanced descriptive scene captioning and intelligent video content summarization"
    page_icon: str = "🎬"
    
    # File handling
    max_file_size_mb: int = 500
    supported_formats: List[str] = field(default_factory=lambda: [
        'mp4', 'avi', 'mov', 'mkv', 'webm'
    ])
    
    # Directories
    data_dir: str = "data"
    videos_dir: str = "data/videos"
    frames_dir: str = "data/frames"
    results_dir: str = "results"
    temp_dir: str = "temp"
    demo_videos_path: str = "data/videos/app_demo"
    
    # UI settings
    layout: str = "wide"
    sidebar_state: str = "expanded"


@dataclass
class TrainingConfig:
    """Configuration for model training/fine-tuning"""
    # Data paths
    train_val_path: str = "data/captions/train_val_videodatainfo.json"
    test_path: str = "data/captions/test_videodatainfo.json"
    processed_data_path: str = "finetuning/data/processed/msrvtt_dataset"
    
    # Model output
    output_dir: str = "finetuning/models/flan-t5-msrvtt"
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    num_epochs: int = 5
    max_input_length: int = 512
    max_target_length: int = 64
    
    # LoRA parameters
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Training settings
    fp16: bool = True
    logging_steps: int = 100
    eval_steps: int = 500
    save_total_limit: int = 3
    seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for evaluation metrics"""
    metrics: List[str] = field(default_factory=lambda: [
        "bleu", "meteor", "rouge"
    ])
    output_dir: str = "evaluation"


@dataclass
class Config:
    """Main configuration class that holds all sub-configurations"""
    models: ModelConfig = field(default_factory=ModelConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    motion: MotionKeywords = field(default_factory=MotionKeywords)
    app: AppConfig = field(default_factory=AppConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load configuration from file or use defaults
        
        Args:
            config_path: Path to configuration file (JSON/YAML)
            
        Returns:
            Config instance
        """
        if config_path and os.path.exists(config_path):
            # Future: Load from JSON/YAML file
            pass
        
        # Return default configuration
        return cls()
    
    def save(self, config_path: str):
        """
        Save configuration to file
        
        Args:
            config_path: Path to save configuration
        """
        # Future: Save to JSON/YAML file
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "models": self.models.__dict__,
            "processing": self.processing.__dict__,
            "motion": self.motion.__dict__,
            "app": self.app.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__
        }


# Create a global configuration instance
config = Config()


# Helper functions for backward compatibility
def get_model_config() -> ModelConfig:
    """Get model configuration"""
    return config.models


def get_processing_config() -> ProcessingConfig:
    """Get processing configuration"""
    return config.processing


def get_app_config() -> AppConfig:
    """Get application configuration"""
    return config.app


def get_training_config() -> TrainingConfig:
    """Get training configuration"""
    return config.training


def get_motion_keywords() -> List[str]:
    """Get motion keywords list"""
    return config.motion.keywords
