#!/usr/bin/env python3
"""
Example script demonstrating IntroStyle feature extraction.

This script shows how to use the IntroStyle model for extracting 
features from sample images.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extract_features import FeatureExtractor
from introstyle import IntroStyleModel

def create_sample_image(size=(512, 512), filename="sample_image.jpg"):
    """Create a sample image for testing."""
    # Create a simple gradient image
    image = Image.new('RGB', size)
    pixels = image.load()
    
    for i in range(size[0]):
        for j in range(size[1]):
            r = int(255 * i / size[0])
            g = int(255 * j / size[1])
            b = int(255 * (i + j) / (size[0] + size[1]))
            pixels[i, j] = (r, g, b)
    
    image.save(filename)
    return filename

def example_single_image():
    """Example: Extract features from a single image."""
    print("=" * 60)
    print("Example 1: Single Image Feature Extraction")
    print("=" * 60)
    
    # Create a sample image
    sample_image = create_sample_image()
    print(f"Created sample image: {sample_image}")
    
    try:
        # Initialize extractor
        print("Initializing IntroStyle feature extractor...")
        extractor = FeatureExtractor(device='auto', t=25, up_ft_index=1)
        
        # Extract features
        print("Extracting features...")
        features = extractor.extract_features_from_image(
            image_path=sample_image,
            output_path="sample_features.npy"
        )
        
        print(f"✓ Features extracted successfully!")
        print(f"  - Feature shape: {features.shape}")
        print(f"  - Feature dtype: {features.dtype}")
        print(f"  - Feature range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  - Features saved to: sample_features.npy")
        
        # Load saved features to verify
        loaded_features = np.load("sample_features.npy")
        print(f"✓ Verified saved features shape: {loaded_features.shape}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    finally:
        # Cleanup
        for file in [sample_image, "sample_features.npy"]:
            if os.path.exists(file):
                os.remove(file)
                print(f"Cleaned up: {file}")
    
    return True

def example_batch_processing():
    """Example: Extract features from multiple images."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)
    
    # Create sample directory with multiple images
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    output_dir = Path("sample_features")
    
    try:
        # Create multiple sample images
        print("Creating sample images...")
        sample_files = []
        for i in range(3):
            filename = sample_dir / f"sample_{i}.jpg"
            create_sample_image(filename=str(filename))
            sample_files.append(filename)
            print(f"  Created: {filename}")
        
        # Initialize extractor
        print("\nInitializing feature extractor...")
        extractor = FeatureExtractor(device='auto', t=25, up_ft_index=1)
        
        # Extract features from directory
        print("Extracting features from directory...")
        extractor.extract_features_from_directory(
            input_dir=sample_dir,
            output_dir=output_dir,
            recursive=False,
            save_format='npy'
        )
        
        # Verify output
        print("\nVerifying extracted features:")
        feature_files = list(output_dir.glob("*.npy"))
        for feature_file in feature_files:
            features = np.load(feature_file)
            print(f"  ✓ {feature_file.name}: shape {features.shape}")
        
        print(f"\n✓ Batch processing completed successfully!")
        print(f"  - Processed {len(sample_files)} images")
        print(f"  - Generated {len(feature_files)} feature files")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        for cleanup_dir in [sample_dir, output_dir]:
            if cleanup_dir.exists():
                shutil.rmtree(cleanup_dir)
                print(f"Cleaned up directory: {cleanup_dir}")
    
    return True

def example_model_parameters():
    """Example: Using different model parameters."""
    print("\n" + "=" * 60)
    print("Example 3: Different Model Parameters")
    print("=" * 60)
    
    # Create a sample image
    sample_image = create_sample_image()
    
    try:
        # Test different parameters
        params_list = [
            {"t": 25, "up_ft_index": 1, "name": "Default"},
            {"t": 50, "up_ft_index": 1, "name": "Higher timestep"},
            {"t": 25, "up_ft_index": 2, "name": "Different layer"},
        ]
        
        results = []
        
        for params in params_list:
            print(f"\nTesting {params['name']} (t={params['t']}, up_ft_index={params['up_ft_index']})...")
            
            extractor = FeatureExtractor(
                device='auto', 
                t=params['t'], 
                up_ft_index=params['up_ft_index']
            )
            
            features = extractor.extract_features_from_image(sample_image)
            results.append({
                'name': params['name'],
                'shape': features.shape,
                'mean': features.mean().item(),
                'std': features.std().item()
            })
            
            print(f"  ✓ Shape: {features.shape}")
            print(f"    Mean: {features.mean():.4f}, Std: {features.std():.4f}")
        
        print("\nSummary of parameter effects:")
        for result in results:
            print(f"  {result['name']:15} | Shape: {str(result['shape']):15} | "
                  f"Mean: {result['mean']:7.4f} | Std: {result['std']:.4f}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    finally:
        # Cleanup
        if os.path.exists(sample_image):
            os.remove(sample_image)
            print(f"\nCleaned up: {sample_image}")
    
    return True

def main():
    """Run all examples."""
    print("IntroStyle Feature Extraction Examples")
    print("=====================================")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name()}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("! CUDA not available, using CPU (slower)")
    
    examples = [
        ("Single Image Processing", example_single_image),
        ("Batch Processing", example_batch_processing),
        ("Parameter Variations", example_model_parameters),
    ]
    
    results = []
    for name, example_func in examples:
        try:
            success = example_func()
            results.append((name, "✓ Success" if success else "✗ Failed"))
        except Exception as e:
            results.append((name, f"✗ Error: {e}"))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, result in results:
        print(f"{name:25} | {result}")
    
    print(f"\nAll examples completed!")

if __name__ == "__main__":
    main()
