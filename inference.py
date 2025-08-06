import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import yaml
import torch
import glob
import nibabel as nib
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from model_2d import DenseUNet2D
from model_3d import DenseUNet3D
from model_hff import HybridFeatureFusion, ProgressiveFeatureFusion
from postprocess import refine_segmentation
from utils import create_2d_features_from_slices
import torch.nn.functional as F

def preprocess_volume(volume, config):
    """
    Preprocess a single volume for inference
    """
    # Intensity clipping and normalization
    clip_min, clip_max = config['preprocessing']['intensity_clip']
    volume = np.clip(volume, clip_min, clip_max)
    volume = (volume - clip_min) / (clip_max - clip_min)
    
    # Resample to target size
    H, W = config['preprocessing']['resample_size']
    D = volume.shape[2]
    
    # Resize each slice
    volume_resized = np.stack([
        cv2.resize(volume[:, :, i], (W, H)) for i in range(D)
    ], axis=2)
    
    return volume_resized

def postprocess_predictions(predictions, original_shape, config):
    """
    Postprocess predictions back to original size
    """
    # Resize predictions back to original size
    H_orig, W_orig, D_orig = original_shape
    H_pred, W_pred = config['preprocessing']['resample_size']
    
    # Resize each slice prediction
    predictions_resized = np.zeros((H_orig, W_orig, D_orig), dtype=np.uint8)
    
    for d in range(D_orig):
        if d < predictions.shape[2]:
            pred_slice = predictions[:, :, d]
            pred_resized = cv2.resize(pred_slice.astype(np.uint8), 
                                    (W_orig, H_orig), 
                                    interpolation=cv2.INTER_NEAREST)
            predictions_resized[:, :, d] = pred_resized
    
    return predictions_resized

def inference():
    """
    Complete inference pipeline for H-DenseUNet
    """
    print("="*60)
    print("H-DenseUNet Inference Pipeline")
    print("="*60)
    
    config = yaml.safe_load(open('config.yaml'))
    
    # Check if model exists
    if not os.path.exists('hdenseunet_full.pth'):
        print("❌ Model file 'hdenseunet_full.pth' not found!")
        print("Please run training first or ensure model file exists.")
        return
    
    # Load models
    print("Loading trained models...")
    model2d = DenseUNet2D(in_channels=3, out_channels=3).cuda()
    model3d = DenseUNet3D(in_channels=4, out_channels=64).cuda()
    
    # 修复：使用与训练时相同的模型结构
    base_hff = HybridFeatureFusion(in_channels_2d=3, in_channels_3d=64, out_channels=3)
    model_hff = ProgressiveFeatureFusion(base_hff).cuda()
    # 设置为完整融合模式（推理时使用所有特征）
    model_hff.set_stage(2)
    
    # Load weights
    try:
        ckpt = torch.load('hdenseunet_full.pth', map_location='cuda')
        model2d.load_state_dict(ckpt['model2d'])
        model3d.load_state_dict(ckpt['model3d'])
        model_hff.load_state_dict(ckpt['model_hff'])
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    # Set models to evaluation mode
    model2d.eval()
    model3d.eval()
    model_hff.eval()
    
    # Create output directories
    output_dir = Path('./output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find test images
    test_images_path = config['data']['test_set']['images']
    if not os.path.exists(test_images_path):
        print(f"❌ Test images directory not found: {test_images_path}")
        return
    
    img_paths = sorted(glob.glob(os.path.join(test_images_path, '*.nii*')))
    
    if not img_paths:
        print(f"❌ No NIfTI files found in {test_images_path}")
        return
    
    print(f"Found {len(img_paths)} test volumes")
    
    # Process each volume
    for i, img_path in enumerate(tqdm(img_paths, desc="Processing volumes")):
        try:
            print(f"\nProcessing {Path(img_path).name} ({i+1}/{len(img_paths)})")
            
            # Load volume
            nii_img = nib.load(img_path)
            volume = nii_img.get_fdata()
            original_shape = volume.shape
            affine = nii_img.affine
            
            print(f"Original shape: {original_shape}")
            
            # Preprocess
            volume_processed = preprocess_volume(volume, config)
            print(f"Processed shape: {volume_processed.shape}")
            
            # Convert to tensor and add batch dimension
            D = volume_processed.shape[2]
            
            # Handle depth dimension for 3D processing
            D_fixed = config['preprocessing'].get('depth_size', 16)  # 从配置读取深度
            if D > D_fixed:
                # Take center slices
                start_idx = (D - D_fixed) // 2
                volume_input = volume_processed[:, :, start_idx:start_idx + D_fixed]
            elif D < D_fixed:
                # Pad with zeros
                pad_before = (D_fixed - D) // 2
                pad_after = D_fixed - D - pad_before
                volume_input = np.pad(volume_processed, 
                                    ((0, 0), (0, 0), (pad_before, pad_after)), 
                                    mode='constant', constant_values=0)
            else:
                volume_input = volume_processed
            
            # Convert to tensor [B, C, D, H, W]
            volume_tensor = torch.from_numpy(volume_input).float().cuda()
            volume_tensor = volume_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            
            # Inference
            with torch.no_grad():
                # Generate 2D features
                feat2d = create_2d_features_from_slices(volume_tensor, model2d, training=False)
                
                # 3D processing
                volume_3d_input = torch.cat([volume_tensor, feat2d], dim=1)
                feat3d = model3d(volume_3d_input)
                
                # Hybrid feature fusion
                output = model_hff(feat2d, feat3d)
                
                # Get predictions
                predictions = torch.softmax(output, dim=1)
                predictions = torch.argmax(predictions, dim=1)  # [B, D, H, W]
                predictions = predictions.squeeze().cpu().numpy()  # [D, H, W]
            
            # Convert back to original orientation [H, W, D]
            predictions = predictions.transpose(1, 2, 0)
            
            # Handle depth dimension mismatch
            if predictions.shape[2] != original_shape[2]:
                # Resize depth dimension
                predictions_orig_depth = np.zeros((predictions.shape[0], predictions.shape[1], original_shape[2]), dtype=np.uint8)
                if predictions.shape[2] < original_shape[2]:
                    # If predicted depth is smaller, center it
                    start_d = (original_shape[2] - predictions.shape[2]) // 2
                    predictions_orig_depth[:, :, start_d:start_d + predictions.shape[2]] = predictions
                else:
                    # If predicted depth is larger, take center
                    start_d = (predictions.shape[2] - original_shape[2]) // 2
                    predictions_orig_depth = predictions[:, :, start_d:start_d + original_shape[2]]
                predictions = predictions_orig_depth
            
            # Postprocess to original size
            predictions_final = postprocess_predictions(predictions, original_shape, config)
            
            # Apply post-processing refinement
            predictions_refined = refine_segmentation(predictions_final)
            
            # Save result
            p_path = Path(img_path)
            stem_name = p_path.name.replace('.nii.gz', '').replace('.nii', '')
            output_path = output_dir / f"{stem_name}_segmentation.nii.gz"
            
            # Create NIfTI image with same affine as input
            result_nii = nib.Nifti1Image(predictions_refined.astype(np.uint8), affine)
            nib.save(result_nii, str(output_path))
            
            print(f"✓ Saved: {output_path}")
            
            # Print some statistics
            unique_labels = np.unique(predictions_refined)
            print(f"Predicted labels: {unique_labels}")
            
            for label in unique_labels:
                if label > 0:
                    count = np.sum(predictions_refined == label)
                    percentage = (count / predictions_refined.size) * 100
                    label_name = ['Background', 'Liver', 'Tumor'][min(label, 2)]
                    print(f"  {label_name} (label {label}): {count} voxels ({percentage:.2f}%)")
                    
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")
            continue
    
    print(f"\n✅ Inference completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Processed {len(img_paths)} volumes")

if __name__ == '__main__':
    inference()