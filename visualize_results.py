#!/usr/bin/env python3
"""
Visualization script for H-DenseUNet results
Creates side-by-side comparisons of CT scans, ground truth, and predictions.

Based on the H-DenseUNet paper visualization approach.
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from tqdm import tqdm
import cv2

def load_nifti_volume(file_path):
    """Load NIfTI volume and return data array"""
    try:
        img = nib.load(file_path)
        data = img.get_fdata()
        return data, img.affine
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None

def normalize_ct_display(ct_volume, window_center=50, window_width=350):
    """
    Normalize CT volume for display using windowing
    Args:
        ct_volume: CT data array
        window_center: HU center for display window
        window_width: HU width for display window
    """
    window_min = window_center - window_width // 2
    window_max = window_center + window_width // 2
    
    ct_windowed = np.clip(ct_volume, window_min, window_max)
    ct_normalized = (ct_windowed - window_min) / (window_max - window_min)
    
    return ct_normalized

def create_overlay_mask(segmentation, alpha=0.6):
    """
    Create colored overlay mask for segmentation
    Args:
        segmentation: segmentation array with labels 0=background, 1=liver, 2=tumor
        alpha: transparency for overlay
    """
    # Define colors: liver=green, tumor=red
    colors = {
        0: [0, 0, 0, 0],        # Background - transparent
        1: [0, 1, 0, alpha],    # Liver - green
        2: [1, 0, 0, alpha]     # Tumor - red
    }
    
    overlay = np.zeros((*segmentation.shape, 4))  
    
    for label, color in colors.items():
        mask = (segmentation == label)
        overlay[mask] = color
    
    return overlay

def calculate_dice_score(pred, gt, label):
    """Calculate Dice score for specific label"""
    pred_mask = (pred == label)
    gt_mask = (gt == label)
    
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / union
    return dice

def create_comparison_plot(ct_slice, gt_slice, pred_slice, slice_idx, case_name, 
                          dice_liver, dice_tumor, output_path):
    """
    Create side-by-side comparison plot
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # CT scan
    axes[0].imshow(ct_slice, cmap='gray')
    axes[0].set_title('CT Scan', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(ct_slice, cmap='gray')
    if gt_slice is not None:
        gt_overlay = create_overlay_mask(gt_slice)
        axes[1].imshow(gt_overlay)
        axes[1].set_title('Ground Truth', fontsize=14, fontweight='bold')
    else:
        axes[1].set_title('Ground Truth\n(Not Available)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(ct_slice, cmap='gray')
    pred_overlay = create_overlay_mask(pred_slice)
    axes[2].imshow(pred_overlay)
    axes[2].set_title('Prediction', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    # Comparison (if ground truth available)
    axes[3].imshow(ct_slice, cmap='gray')
    if gt_slice is not None:
        # Create comparison overlay showing correct/incorrect predictions
        comparison = np.zeros_like(pred_slice, dtype=np.float32)
        comparison[(gt_slice == 1) & (pred_slice == 1)] = 1  # Correct liver - green
        comparison[(gt_slice == 2) & (pred_slice == 2)] = 2  # Correct tumor - blue
        comparison[(gt_slice == 0) & (pred_slice > 0)] = 3   # False positive - yellow
        comparison[(gt_slice > 0) & (pred_slice == 0)] = 4   # False negative - red
        
        comparison_overlay = create_comparison_overlay(comparison)
        axes[3].imshow(comparison_overlay)
        axes[3].set_title('Comparison\n(Green=Correct, Red=FN, Yellow=FP)', fontsize=12, fontweight='bold')
    else:
        axes[3].imshow(pred_overlay)
        axes[3].set_title('Prediction\n(No GT for comparison)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    # Add case information and metrics
    info_text = f"Case: {case_name}\nSlice: {slice_idx}\n"
    if dice_liver is not None and dice_tumor is not None:
        info_text += f"Dice Liver: {dice_liver:.3f}\nDice Tumor: {dice_tumor:.3f}"
    else:
        info_text += "Dice scores: N/A (no GT)"
    
    fig.suptitle(info_text, fontsize=16, fontweight='bold', y=0.02)
    
    # Add legend
    liver_patch = patches.Patch(color='green', alpha=0.6, label='Liver')
    tumor_patch = patches.Patch(color='red', alpha=0.6, label='Tumor')
    fig.legend(handles=[liver_patch, tumor_patch], loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85, bottom=0.15)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_comparison_overlay(comparison, alpha=0.7):
    """Create overlay for comparison visualization"""
    colors = {
        0: [0, 0, 0, 0],        # Background - transparent
        1: [0, 1, 0, alpha],    # Correct liver - green
        2: [0, 0, 1, alpha],    # Correct tumor - blue
        3: [1, 1, 0, alpha],    # False positive - yellow
        4: [1, 0, 0, alpha]     # False negative - red
    }
    
    overlay = np.zeros((*comparison.shape, 4))
    
    for label, color in colors.items():
        mask = (comparison == label)
        overlay[mask] = color
    
    return overlay

def create_volume_summary(ct_volume, gt_volume, pred_volume, case_name, output_dir):
    """
    Create summary visualization showing key slices from the volume
    """
    D = ct_volume.shape[2]
    
    # Select representative slices (every 10th slice or specific percentiles)
    slice_indices = np.linspace(D//4, 3*D//4, 6, dtype=int)
    
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    
    for i, slice_idx in enumerate(slice_indices):
        ct_slice = normalize_ct_display(ct_volume[:, :, slice_idx])
        gt_slice = gt_volume[:, :, slice_idx] if gt_volume is not None else None
        pred_slice = pred_volume[:, :, slice_idx]
        
        # CT scan
        axes[0, i].imshow(ct_slice, cmap='gray')
        axes[0, i].set_title(f'Slice {slice_idx}', fontsize=10)
        axes[0, i].axis('off')
        
        # Ground truth
        axes[1, i].imshow(ct_slice, cmap='gray')
        if gt_slice is not None:
            gt_overlay = create_overlay_mask(gt_slice)
            axes[1, i].imshow(gt_overlay)
        axes[1, i].axis('off')
        
        # Prediction
        axes[2, i].imshow(ct_slice, cmap='gray')
        pred_overlay = create_overlay_mask(pred_slice)
        axes[2, i].imshow(pred_overlay)
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('CT Scan', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Ground Truth', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('Prediction', fontsize=14, fontweight='bold')
    
    # Calculate overall metrics
    if gt_volume is not None:
        dice_liver = calculate_dice_score(pred_volume, gt_volume, 1)
        dice_tumor = calculate_dice_score(pred_volume, gt_volume, 2)
        metrics_text = f"Overall Dice - Liver: {dice_liver:.3f}, Tumor: {dice_tumor:.3f}"
    else:
        metrics_text = "Ground truth not available - no metrics calculated"
    
    fig.suptitle(f"Case: {case_name}\n{metrics_text}", fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    summary_path = os.path.join(output_dir, f"{case_name}_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return summary_path

def visualize_case(ct_path, pred_path, gt_path=None, output_dir="visualizations", 
                   selected_slices=None, create_summary=True):
    """
    Visualize results for a single case
    """
    case_name = Path(ct_path).stem.replace('_segmentation', '')
    
    # Load volumes
    ct_volume, _ = load_nifti_volume(ct_path)
    pred_volume, _ = load_nifti_volume(pred_path)
    gt_volume = None
    
    if gt_path and os.path.exists(gt_path):
        gt_volume, _ = load_nifti_volume(gt_path)
    
    if ct_volume is None or pred_volume is None:
        print(f"Failed to load volumes for case {case_name}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create volume summary
    if create_summary:
        summary_path = create_volume_summary(ct_volume, gt_volume, pred_volume, case_name, output_dir)
        print(f"Created summary: {summary_path}")
    
    # Select slices to visualize
    D = ct_volume.shape[2]
    if selected_slices is None:
        # Automatically select slices with significant content
        slice_scores = []
        for i in range(D):
            pred_slice = pred_volume[:, :, i]
            score = np.sum(pred_slice > 0)  # Count non-background pixels
            slice_scores.append(score)
        
        # Select top 5 slices with most segmented content
        top_slice_indices = np.argsort(slice_scores)[-5:][::-1]
        selected_slices = [i for i in top_slice_indices if slice_scores[i] > 0]
    
    if not selected_slices:
        selected_slices = [D//4, D//2, 3*D//4]  # Default slices
    
    # Create detailed slice visualizations
    slice_dir = os.path.join(output_dir, f"{case_name}_slices")
    os.makedirs(slice_dir, exist_ok=True)
    
    for slice_idx in selected_slices:
        ct_slice = normalize_ct_display(ct_volume[:, :, slice_idx])
        gt_slice = gt_volume[:, :, slice_idx] if gt_volume is not None else None
        pred_slice = pred_volume[:, :, slice_idx]
        
        # Calculate slice-specific metrics
        dice_liver = None
        dice_tumor = None
        if gt_slice is not None:
            dice_liver = calculate_dice_score(pred_slice, gt_slice, 1)
            dice_tumor = calculate_dice_score(pred_slice, gt_slice, 2)
        
        slice_output = os.path.join(slice_dir, f"slice_{slice_idx:03d}.png")
        create_comparison_plot(ct_slice, gt_slice, pred_slice, slice_idx, 
                             case_name, dice_liver, dice_tumor, slice_output)
    
    print(f"Created {len(selected_slices)} slice visualizations for case {case_name}")

def main():
    parser = argparse.ArgumentParser(description='H-DenseUNet Results Visualization')
    parser.add_argument('--ct_dir', type=str, required=True,
                       help='Directory containing original CT volumes')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='Directory containing prediction results')
    parser.add_argument('--gt_dir', type=str, 
                       help='Directory containing ground truth labels (optional)')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--case_name', type=str,
                       help='Specific case to visualize (optional, will process all if not specified)')
    parser.add_argument('--slices', type=int, nargs='+',
                       help='Specific slice indices to visualize (optional)')
    parser.add_argument('--no_summary', action='store_true',
                       help='Skip creating volume summary plots')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ct_dir):
        print(f"CT directory {args.ct_dir} does not exist")
        return
    
    if not os.path.exists(args.pred_dir):
        print(f"Prediction directory {args.pred_dir} does not exist")
        return
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find prediction files
    pred_files = glob.glob(os.path.join(args.pred_dir, '*.nii*'))
    
    if not pred_files:
        print(f"No prediction files found in {args.pred_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    processed_cases = 0
    
    for pred_path in tqdm(pred_files, desc="Creating visualizations"):
        pred_name = Path(pred_path).stem.replace('_segmentation', '')
        
        # Skip if specific case requested and this isn't it
        if args.case_name and args.case_name not in pred_name:
            continue
        
        # Find corresponding CT file
        ct_candidates = [
            os.path.join(args.ct_dir, f"{pred_name}.nii.gz"),
            os.path.join(args.ct_dir, f"{pred_name}.nii"),
            os.path.join(args.ct_dir, pred_name + "_0000.nii.gz"),  # nnUNet format
        ]
        
        ct_path = None
        for candidate in ct_candidates:
            if os.path.exists(candidate):
                ct_path = candidate
                break
        
        if ct_path is None:
            print(f"Could not find CT file for prediction {pred_name}")
            continue
        
        # Find corresponding ground truth file (optional)
        gt_path = None
        if args.gt_dir:
            gt_candidates = [
                os.path.join(args.gt_dir, f"{pred_name}.nii.gz"),
                os.path.join(args.gt_dir, f"{pred_name}.nii"),
            ]
            
            for candidate in gt_candidates:
                if os.path.exists(candidate):
                    gt_path = candidate
                    break
        
        try:
            visualize_case(
                ct_path=ct_path,
                pred_path=pred_path,
                gt_path=gt_path,
                output_dir=args.output_dir,
                selected_slices=args.slices,
                create_summary=not args.no_summary
            )
            processed_cases += 1
            
        except Exception as e:
            print(f"Error processing case {pred_name}: {e}")
            continue
    
    print(f"\n✅ Visualization completed!")
    print(f"Processed {processed_cases} cases")
    print(f"Results saved to: {args.output_dir}")

if __name__ == '__main__':
    main()