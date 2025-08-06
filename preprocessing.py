#!/usr/bin/env python3
"""
Data preprocessing script for H-DenseUNet
Handles NIfTI data preparation, intensity normalization, and data validation.

Based on the H-DenseUNet paper preprocessing pipeline.
"""

import os
import glob
import numpy as np
import nibabel as nib
import argparse
from pathlib import Path
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def analyze_intensity_distribution(data_dir, output_dir):
    """
    Analyze intensity distribution of CT volumes to determine optimal clipping values
    """
    print("Analyzing intensity distribution...")
    
    nii_files = glob.glob(os.path.join(data_dir, '*.nii*'))
    if not nii_files:
        print(f"No NIfTI files found in {data_dir}")
        return
    
    all_intensities = []
    
    for file_path in tqdm(nii_files[:10], desc="Sampling files"):  # Sample first 10 files
        img = nib.load(file_path).get_fdata()
        # Sample 10000 random voxels to avoid memory issues
        sample_size = min(10000, img.size)
        sample_indices = np.random.choice(img.size, sample_size, replace=False)
        sampled_intensities = img.flat[sample_indices]
        all_intensities.extend(sampled_intensities)
    
    all_intensities = np.array(all_intensities)
    
    # Calculate statistics
    stats = {
        'min': np.min(all_intensities),
        'max': np.max(all_intensities),
        'mean': np.mean(all_intensities),
        'std': np.std(all_intensities),
        'median': np.median(all_intensities),
        'p1': np.percentile(all_intensities, 1),
        'p5': np.percentile(all_intensities, 5),
        'p95': np.percentile(all_intensities, 95),
        'p99': np.percentile(all_intensities, 99)
    }
    
    print("\nIntensity Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(all_intensities, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('Full Intensity Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Focus on reasonable CT range
    clipped_intensities = all_intensities[(all_intensities >= -200) & (all_intensities <= 300)]
    plt.hist(clipped_intensities, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel('Intensity Value')
    plt.ylabel('Frequency')
    plt.title('CT Range Intensity Distribution (-200 to 300)')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for suggested clipping values
    plt.axvline(stats['p1'], color='red', linestyle='--', label=f"P1: {stats['p1']:.1f}")
    plt.axvline(stats['p99'], color='red', linestyle='--', label=f"P99: {stats['p99']:.1f}")
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'intensity_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Intensity distribution plot saved to {output_dir}/intensity_distribution.png")
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'intensity_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("Intensity Statistics:\n")
        for key, value in stats.items():
            f.write(f"{key}: {value:.2f}\n")
        
        f.write(f"\nRecommended clipping range: [{stats['p1']:.0f}, {stats['p99']:.0f}]\n")
        f.write(f"Conservative clipping range: [-200, 250]\n")
    
    print(f"Statistics saved to {stats_file}")
    
    return stats

def validate_nifti_file(file_path):
    """
    Validate a single NIfTI file
    Returns: (is_valid, info_dict)
    """
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        
        info = {
            'file': os.path.basename(file_path),
            'shape': data.shape,
            'dtype': data.dtype,
            'min_val': np.min(data),
            'max_val': np.max(data),
            'mean_val': np.mean(data),
            'has_nan': np.isnan(data).any(),
            'has_inf': np.isinf(data).any(),
            'voxel_size': img.header.get_zooms(),
            'orientation': nib.aff2axcodes(img.affine)
        }
        
        is_valid = True
        issues = []
        
        # Check for common issues
        if info['has_nan']:
            issues.append("Contains NaN values")
            is_valid = False
            
        if info['has_inf']:
            issues.append("Contains infinite values")
            is_valid = False
            
        if len(data.shape) != 3:
            issues.append(f"Unexpected dimensions: {data.shape}")
            is_valid = False
            
        if data.size == 0:
            issues.append("Empty volume")
            is_valid = False
        
        info['issues'] = issues
        info['is_valid'] = is_valid
        
        return is_valid, info
        
    except Exception as e:
        return False, {
            'file': os.path.basename(file_path),
            'error': str(e),
            'is_valid': False
        }

def validate_dataset(data_dir, output_dir):
    """
    Validate all NIfTI files in a directory
    """
    print(f"Validating dataset in {data_dir}...")
    
    nii_files = glob.glob(os.path.join(data_dir, '*.nii*'))
    if not nii_files:
        print(f"No NIfTI files found in {data_dir}")
        return
    
    valid_files = []
    invalid_files = []
    all_info = []
    
    for file_path in tqdm(nii_files, desc="Validating files"):
        is_valid, info = validate_nifti_file(file_path)
        all_info.append(info)
        
        if is_valid:
            valid_files.append(file_path)
        else:
            invalid_files.append(file_path)
    
    print(f"\nValidation Results:")
    print(f"  Total files: {len(nii_files)}")
    print(f"  Valid files: {len(valid_files)}")
    print(f"  Invalid files: {len(invalid_files)}")
    
    if invalid_files:
        print("\nInvalid files:")
        for file_path in invalid_files:
            file_info = next(info for info in all_info if info['file'] == os.path.basename(file_path))
            print(f"  {file_info['file']}: {file_info.get('issues', file_info.get('error', 'Unknown error'))}")
    
    # Save validation report
    os.makedirs(output_dir, exist_ok=True)
    report_file = os.path.join(output_dir, 'validation_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("Dataset Validation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset directory: {data_dir}\n")
        f.write(f"Total files: {len(nii_files)}\n")
        f.write(f"Valid files: {len(valid_files)}\n")
        f.write(f"Invalid files: {len(invalid_files)}\n\n")
        
        if valid_files:
            f.write("Valid Files Details:\n")
            f.write("-" * 30 + "\n")
            for info in all_info:
                if info.get('is_valid', False):
                    f.write(f"File: {info['file']}\n")
                    f.write(f"  Shape: {info['shape']}\n")
                    f.write(f"  Voxel size: {info['voxel_size']}\n")
                    f.write(f"  Intensity range: [{info['min_val']:.2f}, {info['max_val']:.2f}]\n")
                    f.write(f"  Mean intensity: {info['mean_val']:.2f}\n")
                    f.write(f"  Orientation: {info['orientation']}\n\n")
        
        if invalid_files:
            f.write("Invalid Files:\n")
            f.write("-" * 20 + "\n")
            for info in all_info:
                if not info.get('is_valid', True):
                    f.write(f"File: {info['file']}\n")
                    f.write(f"  Issues: {info.get('issues', info.get('error', 'Unknown'))}\n\n")
    
    print(f"Validation report saved to {report_file}")
    
    return valid_files, invalid_files

def preprocess_volume(volume_path, output_path, config):
    """
    Preprocess a single volume according to config settings
    """
    try:
        # Load volume
        img = nib.load(volume_path)
        data = img.get_fdata()
        
        # Intensity clipping and normalization
        clip_min, clip_max = config['preprocessing']['intensity_clip']
        data_clipped = np.clip(data, clip_min, clip_max)
        data_normalized = (data_clipped - clip_min) / (clip_max - clip_min)
        
        # Save preprocessed volume
        preprocessed_img = nib.Nifti1Image(data_normalized.astype(np.float32), img.affine, img.header)
        nib.save(preprocessed_img, output_path)
        
        return True, {
            'original_range': [np.min(data), np.max(data)],
            'clipped_range': [np.min(data_clipped), np.max(data_clipped)],
            'normalized_range': [np.min(data_normalized), np.max(data_normalized)]
        }
        
    except Exception as e:
        return False, str(e)

def preprocess_dataset(input_dir, output_dir, config):
    """
    Preprocess entire dataset
    """
    print(f"Preprocessing dataset from {input_dir} to {output_dir}...")
    
    nii_files = glob.glob(os.path.join(input_dir, '*.nii*'))
    if not nii_files:
        print(f"No NIfTI files found in {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for file_path in tqdm(nii_files, desc="Preprocessing"):
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        
        success, info = preprocess_volume(file_path, output_path, config)
        
        if success:
            successful += 1
        else:
            failed += 1
            print(f"Failed to preprocess {filename}: {info}")
    
    print(f"\nPreprocessing completed:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

def main():
    parser = argparse.ArgumentParser(description='H-DenseUNet Data Preprocessing')
    parser.add_argument('--mode', choices=['analyze', 'validate', 'preprocess', 'all'],
                       default='all', help='Preprocessing mode')
    parser.add_argument('--input', type=str, help='Input directory containing NIfTI files')
    parser.add_argument('--output', type=str, default='preprocessing_output',
                       help='Output directory for results')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Configuration file {args.config} not found. Using default settings.")
        config = {
            'preprocessing': {
                'intensity_clip': [-200, 250],
                'resample_size': [256, 256]
            }
        }
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if not args.input:
        print("Please specify input directory with --input")
        return
    
    if not os.path.exists(args.input):
        print(f"Input directory {args.input} does not exist")
        return
    
    if args.mode in ['analyze', 'all']:
        print("\n" + "="*60)
        print("INTENSITY ANALYSIS")
        print("="*60)
        analyze_intensity_distribution(args.input, args.output)
    
    if args.mode in ['validate', 'all']:
        print("\n" + "="*60)
        print("DATASET VALIDATION")
        print("="*60)
        validate_dataset(args.input, args.output)
    
    if args.mode in ['preprocess', 'all']:
        print("\n" + "="*60)
        print("DATA PREPROCESSING")
        print("="*60)
        preprocess_output = os.path.join(args.output, 'preprocessed')
        preprocess_dataset(args.input, preprocess_output, config)
    
    print(f"\n✅ Preprocessing pipeline completed!")
    print(f"Results saved to: {args.output}")

if __name__ == '__main__':
    main()