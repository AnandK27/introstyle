"""
Configuration file for IntroStyle feature extraction.

This file contains default parameters and configuration options
for the IntroStyle model and feature extraction process.
"""

# Model Configuration
MODEL_CONFIG = {
    # Stable Diffusion model identifier
    "sd_model_id": "stabilityai/stable-diffusion-2-1",
    
    # Default diffusion timestep (1-1000)
    # Lower values: more semantic features
    # Higher values: more low-level features
    "default_timestep": 25,
    
    # Default upsampling feature index (0-3)
    # 0: highest resolution, most detailed
    # 3: lowest resolution, most semantic
    "default_up_ft_index": 1,
    
    # Ensemble size for feature extraction
    "ensemble_size": 4,
}

# Image Processing Configuration
IMAGE_CONFIG = {
    # Target image resolution
    "target_size": 512,
    
    # Interpolation method for resizing
    # 0: NEAREST, 1: LANCZOS, 2: BILINEAR, 3: BICUBIC
    "interpolation": 3,
    
    # Normalization parameters (for diffusion models)
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5],
    
    # Supported image formats
    "supported_extensions": ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'],
}

# Processing Configuration
PROCESSING_CONFIG = {
    # Default batch size (recommend 1 for memory efficiency)
    "default_batch_size": 1,
    
    # Default device selection
    "default_device": "auto",  # "auto", "cuda", "cpu"
    
    # Memory optimization settings
    "enable_attention_slicing": True,
    "enable_xformers": True,
    
    # Default output format
    "default_save_format": "npy",  # "npy" or "pt"
    
    # Progress bar settings
    "show_progress": True,
    "progress_desc": "Extracting features",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# Feature Extraction Presets
PRESETS = {
    "fast": {
        "timestep": 10,
        "up_ft_index": 0,
        "ensemble_size": 2,
        "description": "Fast extraction with basic features"
    },
    "default": {
        "timestep": 25,
        "up_ft_index": 1,
        "ensemble_size": 4,
        "description": "Balanced speed and quality"
    },
    "quality": {
        "timestep": 50,
        "up_ft_index": 2,
        "ensemble_size": 8,
        "description": "High-quality features (slower)"
    },
    "semantic": {
        "timestep": 100,
        "up_ft_index": 3,
        "ensemble_size": 4,
        "description": "Semantic-focused features"
    },
}

# Validation ranges
VALIDATION_RANGES = {
    "timestep": (1, 1000),
    "up_ft_index": (0, 3),
    "batch_size": (1, 32),
    "ensemble_size": (1, 16),
}

def get_preset(preset_name: str) -> dict:
    """Get configuration for a specific preset."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    return PRESETS[preset_name].copy()

def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    for param, (min_val, max_val) in VALIDATION_RANGES.items():
        if param in config:
            value = config[param]
            if not (min_val <= value <= max_val):
                raise ValueError(f"{param} must be between {min_val} and {max_val}, got {value}")
    return True

def get_default_config() -> dict:
    """Get default configuration dictionary."""
    return {
        **MODEL_CONFIG,
        **IMAGE_CONFIG,
        **PROCESSING_CONFIG,
        **LOGGING_CONFIG,
    }
