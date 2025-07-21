# IntroStyle: Introspective Style Attribution using Diffusion Features (ICCV '25)

[![arXiv](https://img.shields.io/badge/arXiv-2412.14432-b31b1b.svg)](https://arxiv.org/abs/2412.14432)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://anandk27.github.io/assets/html/IntroStyle/)
[![Code](https://img.shields.io/badge/Code-GitHub-green?logo=github)](https://github.com/AnandK27/IntroStyle)

## Abstract

Text-to-image (T2I) models have gained widespread adoption among content creators and the general public. Gradually, there is an increasing demand for T2I models to incorporate mechanisms that prevent the generation of specific artistic styles, thereby safeguarding intellectual property rights. Existing methods for style extraction typically necessitate the collection of custom datasets and the training of specialized models. This, however, is resource-intensive, time-consuming, and often impractical for real-time applications. We present a novel, training-free framework to solve the style attribution problem, using the features produced by a diffusion model alone, without any external modules or retraining. This is denoted as Introspective Style attribution (`IntroStyle`) and is shown to have superior performance to state-of-the-art models for style attribution. We also introduce a synthetic dataset of Artistic Style Split (`ArtSplit`) to isolate artistic style and evaluate fine-grained style attribution performance. Our experimental results show that our method adequately addresses the dynamic nature of artistic styles and the rapidly evolving landscape of digital art with no training overhead.

## 📋 Requirements

```bash
# Core dependencies
torch>=1.12.0
torchvision>=0.13.0
diffusers>=0.21.0
transformers>=4.21.0
accelerate>=0.12.0

# Additional dependencies
numpy>=1.21.0
Pillow>=8.3.0
tqdm>=4.64.0
scipy>=1.7.0
matplotlib>=3.5.0
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/cuixing100876/InstaStyle.git
   cd InstaStyle/IntroStyle
   ```

2. **Install dependencies:**
   ```bash
   pip install torch torchvision diffusers transformers accelerate
   pip install numpy Pillow tqdm scipy matplotlib
   ```

3. **Verify installation:**
   ```bash
   python -c "import torch; from diffusers import StableDiffusionPipeline; print('Installation successful!')"
   ```

## 💡 Quick Start

### Extract Features from Images

**Single Image:**
```bash
python extract_features.py --input_dir path/to/image.jpg --output_dir ./features --single_image
```

**Directory of Images:**
```bash
python extract_features.py --input_dir path/to/images/ --output_dir ./features
```

**Custom Parameters:**
```bash
python extract_features.py \
    --input_dir path/to/images/ \
    --output_dir ./features \
    --t 50 \
    --up_ft_index 2 \
    --device cuda \
    --batch_size 4
```

### Python API Usage

```python
from extract_features import FeatureExtractor
from PIL import Image
import torch

# Initialize extractor
extractor = FeatureExtractor(device='cuda', t=25, up_ft_index=1)

# Extract features from single image
features = extractor.extract_features_from_image(
    image_path='path/to/image.jpg',
    output_path='path/to/features.npy'
)

# Extract features from directory
extractor.extract_features_from_directory(
    input_dir='path/to/images/',
    output_dir='path/to/features/',
    recursive=True
)

print(f"Feature shape: {features.shape}")
```

## 🗒️ To-Dos
- [ ] Add `ArtSplit` dataset.
- [ ] Add evaluation scripts for style attribution.

## 🔧 Advanced Usage

### Custom Model Parameters

The feature extractor supports various customization options:

```python
extractor = FeatureExtractor(
    device='cuda',           # Device: 'cuda', 'cpu', or 'auto'
    batch_size=1,           # Batch size (recommend 1 for memory efficiency)
    t=25,                   # Diffusion timestep (1-1000)
    up_ft_index=1           # Upsampling layer index (0-3)
)
```

### Supported Formats

- **Input**: JPEG, PNG, BMP, TIFF, WebP
- **Output**: NumPy arrays (`.npy`) or PyTorch tensors (`.pt`)

### Memory Optimization

For large datasets or limited GPU memory:

```bash
# Use CPU processing
python extract_features.py --input_dir ./images --output_dir ./features --device cpu

# Process with smaller batch size
python extract_features.py --input_dir ./images --output_dir ./features --batch_size 1
```

## 📊 Feature Properties

| Property | Value | Description |
|----------|-------|-------------|
| **Feature Dimension** | Variable | Depends on upsampling layer and input resolution |
| **Default Resolution** | 512×512 | Images are resized and center-cropped |
| **Normalization** | [-1, 1] | Standard diffusion model normalization |
| **Ensemble Size** | 4 | Multiple forward passes for robustness |

## 📖 Citation

If you use IntroStyle in your research, please cite our paper:

```bibtex
@article{introstyle2025,
  title={IntroStyle: Introspective Style Attribution using Diffusion Features},
  author={Kumar, Anand and Mu, Jiteng and Vasconcelos, Nuno},
  journal={arXiv preprint arXiv:2024.2412.14432},
  year={2024}
}
```
