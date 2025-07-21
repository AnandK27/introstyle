#!/usr/bin/env python3
"""
IntroStyle Feature Extraction Script

This script extracts IntroStyle features from images using diffusion model features.
It processes all images in an input directory and saves the extracted features.

Usage:
    python extract_features.py --input_dir /path/to/images --output_dir /path/to/output [options]

Requirements:
    - PyTorch with CUDA support
    - diffusers
    - torchvision
    - PIL
    - numpy
    - tqdm
"""

import os
import gc
import argparse
import warnings
from pathlib import Path
from typing import List, Optional, Union
import logging

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import the IntroStyle components
from introstyle import IntroStyleModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Main class for extracting IntroStyle features from images."""
    
    def __init__(self, device: str = 'auto', batch_size: int = 1, t: int = 25, up_ft_index: int = 1):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
            batch_size: Batch size for processing (recommend 1 for memory efficiency)
            t: Timestep parameter for diffusion model
            up_ft_index: Up-sampling feature index to extract
        """
        self.batch_size = batch_size
        self.t = t
        self.up_ft_index = up_ft_index
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        if self.device == 'cpu':
            logger.warning("Using CPU for inference. This will be significantly slower than GPU.")
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        try:
            logger.info("Loading IntroStyle model...")
            self.model = IntroStyleModel()
            if self.device == 'cuda':
                self.model = self.model.cuda()
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_supported_extensions(self) -> List[str]:
        """Get list of supported image extensions."""
        return ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    def _is_valid_image(self, file_path: Path) -> bool:
        """Check if file is a valid image."""
        return file_path.suffix.lower() in self._get_supported_extensions()
    
    def _load_and_preprocess_image(self, image_path: Path) -> Optional[torch.Tensor]:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor or None if loading failed
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply preprocessing
            processed_image = self.model.preprocess(image)
            processed_image = processed_image.unsqueeze(0)  # Add batch dimension
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            return None
    
    def _save_features(self, features: torch.Tensor, output_path: Path, format: str = 'npy'):
        """
        Save features to disk.
        
        Args:
            features: Feature tensor to save
            output_path: Path to save the features
            format: Output format ('npy' or 'pt')
        """
        try:
            features_np = features.cpu().numpy()
            
            if format == 'npy':
                np.save(output_path, features_np)
            elif format == 'pt':
                torch.save(features, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to save features to {output_path}: {e}")
            raise
    
    def extract_features_from_directory(self, 
                                      input_dir: Union[str, Path], 
                                      output_dir: Union[str, Path],
                                      recursive: bool = True,
                                      save_format: str = 'npy',
                                      overwrite: bool = False) -> None:
        """
        Extract features from all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save extracted features
            recursive: Whether to search subdirectories recursively
            save_format: Format to save features ('npy' or 'pt')
            overwrite: Whether to overwrite existing feature files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Validate input directory
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
        
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        pattern = "**/*" if recursive else "*"
        all_files = list(input_dir.glob(pattern))
        image_files = [f for f in all_files if f.is_file() and self._is_valid_image(f)]
        
        if not image_files:
            logger.warning(f"No valid image files found in {input_dir}")
            return
        
        logger.info(f"Found {len(image_files)} image files")
        
        # Process images
        successful_extractions = 0
        failed_extractions = 0
        
        with torch.no_grad():
            for image_path in tqdm(image_files, desc="Extracting features"):
                try:
                    # Calculate relative path for maintaining directory structure
                    rel_path = image_path.relative_to(input_dir)
                    output_name = rel_path.stem + f"_features.{save_format}"
                    output_path = output_dir / rel_path.parent / output_name
                    
                    # Skip if file exists and not overwriting
                    if output_path.exists() and not overwrite:
                        logger.debug(f"Skipping {image_path.name} (features already exist)")
                        continue
                    
                    # Create output subdirectory if needed
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Load and preprocess image
                    image_tensor = self._load_and_preprocess_image(image_path)
                    if image_tensor is None:
                        failed_extractions += 1
                        continue
                    
                    # Move to device
                    if self.device == 'cuda':
                        image_tensor = image_tensor.cuda()
                    
                    # Extract features
                    features = self.model(image_tensor, t=self.t, up_ft_index=self.up_ft_index)
                    
                    # Save features
                    self._save_features(features, output_path, save_format)
                    successful_extractions += 1
                    
                    # Clear GPU memory
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    failed_extractions += 1
                    continue
        
        logger.info(f"Feature extraction completed: {successful_extractions} successful, {failed_extractions} failed")
    
    def extract_features_from_image(self, 
                                  image_path: Union[str, Path], 
                                  output_path: Optional[Union[str, Path]] = None,
                                  save_format: str = 'npy') -> torch.Tensor:
        """
        Extract features from a single image.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save features (optional)
            save_format: Format to save features ('npy' or 'pt')
            
        Returns:
            Extracted features as tensor
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        
        # Load and preprocess image
        image_tensor = self._load_and_preprocess_image(image_path)
        if image_tensor is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Move to device
        if self.device == 'cuda':
            image_tensor = image_tensor.cuda()
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor, t=self.t, up_ft_index=self.up_ft_index)
        
        # Save if output path provided
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_features(features, output_path, save_format)
        
        return features


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Extract IntroStyle features from images using diffusion model features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract features from all images in a directory
    python extract_features.py --input_dir ./images --output_dir ./features
    
    # Extract features with custom parameters
    python extract_features.py --input_dir ./images --output_dir ./features --t 50 --up_ft_index 2
    
    # Extract features from a single image
    python extract_features.py --input_dir ./image.jpg --output_dir ./features --single_image
    
    # Use CPU instead of GPU
    python extract_features.py --input_dir ./images --output_dir ./features --device cpu
        """
    )
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing images or path to single image')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory to save extracted features')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for inference (default: auto)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing (default: 1)')
    parser.add_argument('--t', type=int, default=25,
                       help='Timestep parameter for diffusion model (default: 25)')
    parser.add_argument('--up_ft_index', type=int, default=1,
                       help='Up-sampling feature index to extract (default: 1)')
    parser.add_argument('--recursive', action='store_true', default=True,
                       help='Search subdirectories recursively (default: True)')
    parser.add_argument('--no_recursive', action='store_true',
                       help='Do not search subdirectories recursively')
    parser.add_argument('--save_format', type=str, default='npy', choices=['npy', 'pt'],
                       help='Format to save features (default: npy)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing feature files')
    parser.add_argument('--single_image', action='store_true',
                       help='Process input as single image instead of directory')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle recursive flag
    recursive = args.recursive and not args.no_recursive
    
    try:
        # Initialize extractor
        extractor = FeatureExtractor(
            device=args.device,
            batch_size=args.batch_size,
            t=args.t,
            up_ft_index=args.up_ft_index
        )
        
        if args.single_image:
            # Process single image
            input_path = Path(args.input_dir)
            output_name = input_path.stem + f"_features.{args.save_format}"
            output_path = Path(args.output_dir) / output_name
            
            logger.info(f"Processing single image: {input_path}")
            features = extractor.extract_features_from_image(
                input_path, 
                output_path, 
                args.save_format
            )
            logger.info(f"Features saved to: {output_path}")
            logger.info(f"Feature shape: {features.shape}")
        else:
            # Process directory
            logger.info(f"Processing directory: {args.input_dir}")
            extractor.extract_features_from_directory(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                recursive=recursive,
                save_format=args.save_format,
                overwrite=args.overwrite
            )
        
        logger.info("Feature extraction completed successfully")
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
