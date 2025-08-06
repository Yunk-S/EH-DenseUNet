import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def dice_coefficient(pred, target, smooth=1e-5):
    """
    Calculate Dice coefficient for evaluation
    Args:
        pred: predictions [B, C, ...] 
        target: ground truth labels [B, ...]
        smooth: smoothing factor
    Returns:
        dice: Dice coefficient per class
    """
    pred = torch.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, -1, *range(1, target_onehot.ndim-1)).float()
    
    dims = list(range(2, pred.ndim))
    inter = (pred * target_onehot).sum(dim=dims)
    union = pred.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2 * inter + smooth) / (union + smooth)
    return dice

class DiceLoss(nn.Module):
    """Dice Loss implementation"""
    def __init__(self, smooth=1e-5, weight=None):
        super().__init__()
        self.smooth = smooth
        self.weight = weight
    
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1])
        target_onehot = target_onehot.permute(0, -1, *range(1, target_onehot.ndim-1)).float()
        
        dims = list(range(2, pred.ndim))
        inter = (pred * target_onehot).sum(dim=dims)
        union = pred.sum(dim=dims) + target_onehot.sum(dim=dims)
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        loss = 1 - dice
        
        if self.weight is not None:
            loss = loss * self.weight
        
        return loss.mean()

class CombinedLoss(nn.Module):
    """Combined Cross Entropy + Dice Loss"""
    def __init__(self, ce_weight=None, dice_weight=1.0, ce_weight_factor=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice_loss = DiceLoss()
        self.dice_weight = dice_weight
        self.ce_weight_factor = ce_weight_factor
    
    def forward(self, pred, target):
        ce = self.ce_loss(pred, target) * self.ce_weight_factor
        dice = self.dice_loss(pred, target) * self.dice_weight
        return ce + dice

def calculate_iou(pred, target, num_classes=3):
    """Calculate IoU for each class"""
    pred = torch.argmax(pred, dim=1)
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        if union == 0:
            iou = torch.tensor(1.0)  # If no ground truth and no prediction, perfect score
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return torch.stack(ious)

def create_2d_features_from_slices(volume, model2d, training=True):
    """
    Generate 2D features by processing volume slice by slice
    Args:
        volume: [B, C, D, H, W] input volume
        model2d: 2D model
        training: bool, 是否在训练模式
    Returns:
        features_2d: [B, C_out, D, H, W] 2D features
    """
    B, C, D, H, W = volume.shape
    features_list = []
    
    # 根据训练状态决定是否使用梯度
    context_manager = torch.no_grad() if not training else torch.enable_grad()
    
    with context_manager:
        for d in range(D):
            # Get slice with neighboring slices for context
            if d == 0:
                slice_input = torch.cat([volume[:, :, 0:1], volume[:, :, 0:2]], dim=2)
            elif d == D - 1:
                slice_input = torch.cat([volume[:, :, D-2:D], volume[:, :, D-1:D]], dim=2)
            else:
                slice_input = volume[:, :, d-1:d+2]
            
            # Reshape for 2D processing: [B, C*3, H, W]
            slice_input = slice_input.view(B, C*3, H, W)
            
            # Process with 2D model
            slice_features = model2d(slice_input)  # [B, C_out, H, W]
            
            features_list.append(slice_features.unsqueeze(2))  # [B, C_out, 1, H, W]
    
    # Concatenate along depth dimension
    features_2d = torch.cat(features_list, dim=2)  # [B, C_out, D, H, W]
    return features_2d

# Legacy function for backward compatibility
def dice_loss(pred, target, smooth=1e-5):
    """Legacy dice loss function"""
    return DiceLoss(smooth=smooth)(pred, target)