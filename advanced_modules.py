#!/usr/bin/env python3
"""
Advanced optimization modules for H-DenseUNet
Implements state-of-the-art techniques to improve segmentation accuracy and training efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class SEBlock3D(nn.Module):
    """Squeeze-and-Excitation block for 3D features"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class ResidualDenseBlock3D(nn.Module):
    """Residual Dense Block with growth rate and dense connections"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm3d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False),
                SEBlock3D(growth_rate)  # Add SE attention
            )
            self.layers.append(layer)
            
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels + num_layers * growth_rate, in_channels, 1, bias=False),
            nn.BatchNorm3d(in_channels)
        )
        
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feature = layer(torch.cat(features, 1))
            features.append(new_feature)
        
        # Dense connection
        dense_features = torch.cat(features, 1)
        out = self.fusion(dense_features)
        
        # Residual connection
        return x + out

class PyramidPoolingModule(nn.Module):
    """Pyramid Pooling Module for multi-scale context"""
    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(size),
                nn.Conv3d(in_channels, in_channels // len(pool_sizes), 1, bias=False),
                nn.BatchNorm3d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ) for size in pool_sizes
        ])
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channels + (in_channels // len(pool_sizes)) * len(pool_sizes), 
                     in_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        size = x.size()[2:]
        out = [x]
        
        for stage in self.stages:
            upsampled = F.interpolate(stage(x), size=size, mode='trilinear', align_corners=False)
            out.append(upsampled)
            
        return self.bottleneck(torch.cat(out, 1))

class MultiScaleFeatureFusion(nn.Module):
    """Multi-scale feature fusion with adaptive weights"""
    def __init__(self, channels_list, out_channels):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(ch, out_channels, 1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ) for ch in channels_list
        ])
        
        # Learnable weights for different scales
        self.scale_weights = nn.Parameter(torch.ones(len(channels_list)))
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features):
        """features: list of tensors with different scales"""
        # Resize all features to the same size (largest)
        target_size = max([f.size()[2:] for f in features])
        
        weighted_features = []
        for i, (feat, conv) in enumerate(zip(features, self.convs)):
            # Resize to target size
            resized = F.interpolate(feat, size=target_size, mode='trilinear', align_corners=False)
            # Apply convolution and weight
            weighted = conv(resized) * torch.softmax(self.scale_weights, 0)[i]
            weighted_features.append(weighted)
        
        # Sum and fuse
        fused = sum(weighted_features)
        return self.fusion_conv(fused)

class BoundaryAwareLoss(nn.Module):
    """Boundary-aware loss function for better edge segmentation"""
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
        
    def get_boundary_mask(self, target):
        """Extract boundary from segmentation mask"""
        # Use morphological operations to get boundaries
        kernel = torch.ones(1, 1, 3, 3, 3).to(target.device)
        target_float = target.float().unsqueeze(1)
        
        # Dilate and erode to get boundary
        dilated = F.conv3d(target_float, kernel, padding=1)
        eroded = -F.conv3d(-target_float, kernel, padding=1)
        boundary = (dilated - eroded) > 0
        
        return boundary.squeeze(1)
    
    def forward(self, pred, target):
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(pred, target)
        
        # Boundary loss
        boundary_mask = self.get_boundary_mask(target)
        pred_softmax = F.softmax(pred, dim=1)
        
        # Focus on boundary regions
        boundary_loss = 0
        for c in range(pred.size(1)):
            class_pred = pred_softmax[:, c]
            class_target = (target == c).float()
            boundary_error = torch.abs(class_pred - class_target) * boundary_mask.float()
            boundary_loss += boundary_error.mean()
        
        return self.alpha * ce_loss + self.beta * boundary_loss

class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdversarialLoss(nn.Module):
    """Simple adversarial loss for better feature learning"""
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Conv3d(num_classes, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(256, 1, 4, stride=1, padding=1),
        )
        
    def forward(self, pred, target):
        # Discriminator tries to distinguish between pred and target
        pred_score = self.discriminator(F.softmax(pred, dim=1))
        target_onehot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 4, 1, 2, 3).float()
        target_score = self.discriminator(target_onehot)
        
        # Generator loss (segmentation model wants to fool discriminator)
        gen_loss = F.binary_cross_entropy_with_logits(pred_score, torch.ones_like(pred_score))
        
        # Discriminator loss
        real_loss = F.binary_cross_entropy_with_logits(target_score, torch.ones_like(target_score))
        fake_loss = F.binary_cross_entropy_with_logits(pred_score, torch.zeros_like(pred_score))
        disc_loss = (real_loss + fake_loss) / 2
        
        return gen_loss, disc_loss

class ConsistencyLoss(nn.Module):
    """Consistency loss for semi-supervised learning"""
    def __init__(self, consistency_weight=1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
        
    def forward(self, pred1, pred2):
        """Compute consistency between two predictions"""
        pred1_soft = F.softmax(pred1, dim=1)
        pred2_soft = F.softmax(pred2, dim=1)
        
        consistency_loss = F.mse_loss(pred1_soft, pred2_soft)
        return self.consistency_weight * consistency_loss

class AdvancedDataAugmentation:
    """Advanced data augmentation techniques"""
    
    @staticmethod
    def mixup(x, y, alpha=0.2):
        """MixUp augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    @staticmethod
    def cutmix(x, y, alpha=1.0):
        """CutMix augmentation"""
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        # Get random bounding box
        D, H, W = x.size()[2:]
        cut_rat = np.sqrt(1. - lam)
        cut_d = int(D * cut_rat)
        cut_h = int(H * cut_rat)  
        cut_w = int(W * cut_rat)
        
        # Random center
        cd = np.random.randint(D)
        ch = np.random.randint(H)
        cw = np.random.randint(W)
        
        # Bounding box
        bbd1 = np.clip(cd - cut_d // 2, 0, D)
        bbd2 = np.clip(cd + cut_d // 2, 0, D)
        bbh1 = np.clip(ch - cut_h // 2, 0, H)
        bbh2 = np.clip(ch + cut_h // 2, 0, H)
        bbw1 = np.clip(cw - cut_w // 2, 0, W)
        bbw2 = np.clip(cw + cut_w // 2, 0, W)
        
        x[:, :, bbd1:bbd2, bbh1:bbh2, bbw1:bbw2] = x[index, :, bbd1:bbd2, bbh1:bbh2, bbw1:bbw2]
        y[:, bbd1:bbd2, bbh1:bbh2, bbw1:bbw2] = y[index, bbd1:bbd2, bbh1:bbh2, bbw1:bbw2]
        
        # Adjust lambda
        lam = 1 - ((bbd2 - bbd1) * (bbh2 - bbh1) * (bbw2 - bbw1) / (D * H * W))
        
        return x, y, lam

class ModelEnsemble:
    """Model ensemble for improved predictions"""
    def __init__(self, models):
        self.models = models
        
    def predict(self, x):
        """Ensemble prediction using multiple models"""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(0)
        return ensemble_pred
    
    def predict_with_tta(self, x, tta_transforms=None):
        """Test-time augmentation ensemble"""
        if tta_transforms is None:
            tta_transforms = [
                lambda x: x,                                           # Original
                lambda x: torch.flip(x, dims=[2]),                   # Flip depth
                lambda x: torch.flip(x, dims=[3]),                   # Flip height
                lambda x: torch.flip(x, dims=[4]),                   # Flip width
                lambda x: torch.flip(x, dims=[3, 4]),               # Flip H&W
            ]
        
        predictions = []
        for transform in tta_transforms:
            # Apply transform
            x_aug = transform(x)
            
            # Get ensemble prediction
            pred = self.predict(x_aug)
            
            # Reverse transform on prediction
            if transform != tta_transforms[0]:  # Not original
                # Apply reverse transform (simplified)
                pred = transform(pred)
            
            predictions.append(pred)
        
        # Average TTA predictions
        return torch.stack(predictions).mean(0)

class EarlyStoppingCallback:
    """Early stopping with patience"""
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

class LearningRateScheduler:
    """Advanced learning rate scheduling"""
    def __init__(self, optimizer, mode='cosine', **kwargs):
        self.optimizer = optimizer
        self.mode = mode
        self.kwargs = kwargs
        
        if mode == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(optimizer, **kwargs)
        elif mode == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            self.scheduler = ReduceLROnPlateau(optimizer, **kwargs)
        elif mode == 'warmup_cosine':
            self.warmup_epochs = kwargs.get('warmup_epochs', 10)
            self.total_epochs = kwargs.get('total_epochs', 100)
            self.base_lr = optimizer.param_groups[0]['lr']
            self.min_lr = kwargs.get('min_lr', 1e-6)
            
    def step(self, epoch=None, metrics=None):
        if self.mode == 'warmup_cosine':
            if epoch < self.warmup_epochs:
                lr = self.base_lr * epoch / self.warmup_epochs
            else:
                lr = self.min_lr + (self.base_lr - self.min_lr) * \
                     (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / 
                                (self.total_epochs - self.warmup_epochs))) / 2
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            if self.mode == 'plateau' and metrics is not None:
                self.scheduler.step(metrics)
            else:
                self.scheduler.step()

class GradientAccumulator:
    """Gradient accumulation for large effective batch sizes"""
    def __init__(self, accumulation_steps=4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def step(self, loss, optimizer, scaler=None):
        # Scale loss by accumulation steps
        loss = loss / self.accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self.current_step += 1
        
        # Update weights when accumulation is complete
        if self.current_step % self.accumulation_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            return True
        
        return False