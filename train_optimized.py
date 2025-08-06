#!/usr/bin/env python3
"""
Optimized training script for Enhanced H-DenseUNet
Incorporates state-of-the-art training techniques:
- Multi-stage progressive training
- Advanced loss functions with deep supervision
- Model ensemble and knowledge distillation
- Advanced data augmentation
- Automated hyperparameter optimization
- Comprehensive monitoring and logging
"""

import os
import yaml
import torch
import numpy as np
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader, RandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import json

# Import enhanced models and modules
from model_2d import DenseUNet2D, DeepSupervisionLoss
from model_3d import DenseUNet3D, DistillationLoss
from model_hff import HybridFeatureFusion, ProgressiveFeatureFusion
from advanced_modules import (
    BoundaryAwareLoss, FocalLoss, ConsistencyLoss, AdvancedDataAugmentation,
    ModelEnsemble, EarlyStoppingCallback, LearningRateScheduler, GradientAccumulator
)
from dataset import LiverTumorDataset2D, LiverTumorDataset3D
from utils import set_seed, dice_coefficient, calculate_iou, create_2d_features_from_slices

class OptimizedLossFunction(nn.Module):
    """Optimized multi-component loss function"""
    def __init__(self, weights=None, alpha=0.7, gamma=2.0, boundary_weight=1.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.boundary_loss = BoundaryAwareLoss(alpha=1.0, beta=boundary_weight)
        self.consistency_loss = ConsistencyLoss(consistency_weight=0.1)
        
        # Loss weights
        self.ce_weight = 0.4
        self.focal_weight = 0.3
        self.boundary_weight = 0.2
        self.consistency_weight = 0.1
        
    def forward(self, pred, target, pred_augmented=None):
        ce = self.ce_loss(pred, target)
        focal = self.focal_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        
        total_loss = (self.ce_weight * ce + 
                     self.focal_weight * focal + 
                     self.boundary_weight * boundary)
        
        # Add consistency loss if augmented prediction is provided
        if pred_augmented is not None:
            consistency = self.consistency_loss(pred, pred_augmented)
            total_loss += self.consistency_weight * consistency
            
        return total_loss, {
            'ce_loss': ce.item(),
            'focal_loss': focal.item(),
            'boundary_loss': boundary.item(),
            'total_loss': total_loss.item()
        }

class AdvancedTrainer:
    """Advanced trainer with comprehensive optimization techniques"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        self.current_epoch = 0
        self.best_metrics = {'dice': 0.0, 'loss': float('inf')}
        
        # Initialize models
        self._init_models()
        
        # Initialize optimizers and schedulers
        self._init_optimizers()
        
        # Initialize loss functions
        self._init_loss_functions()
        
        # Initialize data loaders
        self._init_data_loaders()
        
        # Initialize callbacks and utilities
        self._init_callbacks()
        
        # Initialize logging
        self._init_logging()
        
    def _init_models(self):
        """Initialize enhanced models"""
        print("Initializing enhanced models...")
        
        # Enhanced 2D model
        self.model2d = DenseUNet2D(
            in_channels=3,
            out_channels=3,
            growth_rate=self.config['model']['growth_rate_2d'],
            num_layers=self.config['model']['num_layers_2d']
        ).to(self.device)
        
        # Enhanced 3D model  
        self.model3d = DenseUNet3D(
            in_channels=4,
            out_channels=64,
            growth_rate=self.config['model']['growth_rate_3d'],
            num_layers=self.config['model']['num_layers_3d']
        ).to(self.device)
        
        # Progressive HFF model
        base_hff = HybridFeatureFusion(
            in_channels_2d=3,
            in_channels_3d=64,
            out_channels=3
        )
        self.model_hff = ProgressiveFeatureFusion(base_hff).to(self.device)
        
        # Model ensemble for inference
        self.ensemble = ModelEnsemble([self.model2d, self.model3d])
        
        print(f"2D Model parameters: {sum(p.numel() for p in self.model2d.parameters())/1e6:.2f}M")
        print(f"3D Model parameters: {sum(p.numel() for p in self.model3d.parameters())/1e6:.2f}M")
        print(f"HFF Model parameters: {sum(p.numel() for p in self.model_hff.parameters())/1e6:.2f}M")
        
    def _init_optimizers(self):
        """Initialize optimizers and schedulers"""
        # Different optimizers for different stages
        self.optimizer_2d = optim.AdamW(
            self.model2d.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        self.optimizer_3d = optim.AdamW(
            list(self.model3d.parameters()) + list(self.model_hff.parameters()),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        self.optimizer_full = optim.AdamW(
            list(self.model2d.parameters()) + 
            list(self.model3d.parameters()) + 
            list(self.model_hff.parameters()),
            lr=self.config['training']['learning_rate'] / 10,
            weight_decay=self.config['training']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Advanced learning rate schedulers
        self.scheduler_2d = LearningRateScheduler(
            self.optimizer_2d, 
            mode='warmup_cosine',
            warmup_epochs=10,
            total_epochs=self.config['training']['epochs_2d']
        )
        
        self.scheduler_3d = LearningRateScheduler(
            self.optimizer_3d,
            mode='warmup_cosine', 
            warmup_epochs=5,
            total_epochs=self.config['training']['epochs_3d']
        )
        
        self.scheduler_full = LearningRateScheduler(
            self.optimizer_full,
            mode='plateau',
            patience=5,
            factor=0.5
        )
        
        # Mixed precision scalers
        self.scaler_2d = GradScaler()
        self.scaler_3d = GradScaler()
        self.scaler_full = GradScaler()
        
        # Gradient accumulation
        self.grad_accumulator = GradientAccumulator(
            accumulation_steps=self.config['training']['accumulation_steps']
        )
        
    def _init_loss_functions(self):
        """Initialize advanced loss functions"""
        class_weights = torch.tensor(self.config['training']['class_weights']).to(self.device)
        
        # Primary loss function
        self.criterion = OptimizedLossFunction(
            weights=class_weights,
            alpha=0.7,
            gamma=2.0,
            boundary_weight=1.5
        )
        
        # Deep supervision loss
        self.deep_supervision_loss = DeepSupervisionLoss(
            loss_fn=self.criterion,
            weights=[1.0, 0.5, 0.25]
        )
        
        # Knowledge distillation loss (for model compression)
        self.distillation_loss = DistillationLoss(alpha=0.7, temperature=4.0)
        
    def _init_data_loaders(self):
        """Initialize advanced data loaders with augmentation"""
        # 2D data loader
        datasets_2d = []
        for train_set in self.config['data']['train_sets']:
            if os.path.exists(train_set['images']) and os.path.exists(train_set['labels']):
                dataset = LiverTumorDataset2D(
                    train_set['images'], 
                    train_set['labels'], 
                    self.config,
                    augment=True
                )
                datasets_2d.append(dataset)
        
        if datasets_2d:
            combined_dataset_2d = torch.utils.data.ConcatDataset(datasets_2d)
            self.train_loader_2d = DataLoader(
                combined_dataset_2d,
                batch_size=self.config['training']['batch_size_2d'],
                shuffle=True,
                num_workers=self.config['training']['num_workers'],
                pin_memory=True,
                drop_last=True
            )
        
        # 3D data loader
        datasets_3d = []
        for train_set in self.config['data']['train_sets']:
            if os.path.exists(train_set['images']) and os.path.exists(train_set['labels']):
                dataset = LiverTumorDataset3D(
                    train_set['images'],
                    train_set['labels'],
                    self.config
                )
                datasets_3d.append(dataset)
        
        if datasets_3d:
            combined_dataset_3d = torch.utils.data.ConcatDataset(datasets_3d)
            self.train_loader_3d = DataLoader(
                combined_dataset_3d,
                batch_size=self.config['training']['batch_size_3d'],
                shuffle=True,
                num_workers=self.config['training']['num_workers'],
                pin_memory=True,
                drop_last=True
            )
        
        print(f"2D Training samples: {len(combined_dataset_2d) if datasets_2d else 0}")
        print(f"3D Training samples: {len(combined_dataset_3d) if datasets_3d else 0}")
        
    def _init_callbacks(self):
        """Initialize callbacks and utilities"""
        self.early_stopping = EarlyStoppingCallback(
            patience=self.config['training']['early_stopping_patience'],
            delta=0.001
        )
        
        self.augmentation = AdvancedDataAugmentation()
        
    def _init_logging(self):
        """Initialize logging and monitoring"""
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['run_name'],
                config=self.config
            )
        
        # Create output directories
        self.output_dir = Path(self.config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_dice': [],
            'train_iou': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
    def train_stage1_2d(self):
        """Stage 1: Train 2D DenseUNet with deep supervision"""
        print("\n" + "="*60)
        print("STAGE 1: Training Enhanced 2D DenseUNet")
        print("="*60)
        
        self.model2d.train()
        # 修复：设置正确的渐进式训练阶段（只使用2D特征）
        self.model_hff.set_stage(0)
        best_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs_2d']):
            epoch_start = time.time()
            epoch_loss = 0.0
            dice_scores = []
            iou_scores = []
            
            progress_bar = tqdm(self.train_loader_2d, desc=f"2D Epoch {epoch+1}")
            
            for batch_idx, (imgs, labels) in enumerate(progress_bar):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                # Advanced data augmentation
                if np.random.random() < 0.5:
                    imgs_aug, labels_a, labels_b, lam = self.augmentation.mixup(imgs, labels)
                    use_mixup = True
                else:
                    imgs_aug = imgs
                    use_mixup = False
                
                self.optimizer_2d.zero_grad()
                
                with autocast():
                    # Forward pass with deep supervision
                    if self.config['model']['use_deep_supervision']:
                        output, aux_outputs = self.model2d(imgs_aug, return_aux=True)
                        loss = self.deep_supervision_loss(output, aux_outputs, labels)
                    else:
                        output = self.model2d(imgs_aug)
                        
                        if use_mixup:
                            loss_a, _ = self.criterion(output, labels_a)
                            loss_b, _ = self.criterion(output, labels_b)
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:
                            loss, loss_dict = self.criterion(output, labels)
                
                # Gradient accumulation
                should_step = self.grad_accumulator.step(loss, self.optimizer_2d, self.scaler_2d)
                
                if should_step:
                    # Update learning rate
                    self.scheduler_2d.step(epoch)
                
                epoch_loss += loss.item()
                
                # Calculate metrics
                with torch.no_grad():
                    dice = dice_coefficient(output, labels)
                    iou = calculate_iou(output, labels)
                    dice_scores.append(dice.cpu())
                    iou_scores.append(iou.cpu())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.mean().item():.4f}',
                    'LR': f'{self.optimizer_2d.param_groups[0]["lr"]:.6f}'
                })
            
            # Epoch statistics
            avg_loss = epoch_loss / len(self.train_loader_2d)
            avg_dice = torch.stack(dice_scores).mean(0)
            avg_iou = torch.stack(iou_scores).mean(0)
            epoch_time = time.time() - epoch_start
            
            # Logging
            self.history['train_loss'].append(avg_loss)
            self.history['train_dice'].append(avg_dice.mean().item())
            self.history['train_iou'].append(avg_iou.mean().item())
            self.history['learning_rate'].append(self.optimizer_2d.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            
            print(f"2D Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"Dice=[{avg_dice[0]:.3f}, {avg_dice[1]:.3f}, {avg_dice[2]:.3f}], "
                  f"IoU=[{avg_iou[0]:.3f}, {avg_iou[1]:.3f}, {avg_iou[2]:.3f}], "
                  f"Time={epoch_time:.1f}s")
            
            # Wandb logging
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'stage1/train_loss': avg_loss,
                    'stage1/train_dice_bg': avg_dice[0].item(),
                    'stage1/train_dice_liver': avg_dice[1].item(),
                    'stage1/train_dice_tumor': avg_dice[2].item(),
                    'stage1/learning_rate': self.optimizer_2d.param_groups[0]['lr'],
                    'epoch': epoch
                })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint('best_2d', epoch, {
                    'model2d': self.model2d.state_dict(),
                    'optimizer': self.optimizer_2d.state_dict(),
                    'loss': avg_loss,
                    'dice': avg_dice.mean().item()
                })
        
        print(f"✅ Stage 1 completed! Best Loss: {best_loss:.4f}")
        
    def train_stage2_3d_hff(self):
        """Stage 2: Train 3D DenseUNet + HFF with frozen 2D weights"""
        print("\n" + "="*60)
        print("STAGE 2: Training 3D DenseUNet + Hybrid Feature Fusion")
        print("="*60)
        
        # Load best 2D model and freeze
        self.load_checkpoint('best_2d')
        for param in self.model2d.parameters():
            param.requires_grad = False
        
        self.model3d.train()
        self.model_hff.train()
        self.model_hff.set_stage(1)  # Progressive training - 3D only first
        
        best_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs_3d']):
            epoch_start = time.time()
            epoch_loss = 0.0
            dice_scores = []
            iou_scores = []
            
            # Switch to full fusion after half epochs
            if epoch >= self.config['training']['epochs_3d'] // 2:
                self.model_hff.set_stage(2)  # Full fusion
            
            progress_bar = tqdm(self.train_loader_3d, desc=f"3D Epoch {epoch+1}")
            
            for batch_idx, (volume, mask) in enumerate(progress_bar):
                volume, mask = volume.to(self.device), mask.to(self.device)
                B, C, D, H, W = volume.shape
                
                self.optimizer_3d.zero_grad()
                
                with autocast():
                    # Generate 2D features (frozen network)
                    with torch.no_grad():
                        feat2d = create_2d_features_from_slices(volume, self.model2d, training=False)
                    
                    # 3D processing
                    volume_3d_input = torch.cat([volume, feat2d], dim=1)
                    
                    if self.config['model']['use_deep_supervision']:
                        feat3d, aux_3d = self.model3d(volume_3d_input, return_aux=True)
                        output, aux_hff = self.model_hff(feat2d, feat3d, return_aux=True)
                        
                        # Combined deep supervision loss
                        loss = self.deep_supervision_loss(output, aux_hff, mask.squeeze(1))
                        aux_loss = self.deep_supervision_loss(feat3d, aux_3d, mask.squeeze(1))
                        loss = loss + 0.5 * aux_loss
                    else:
                        feat3d = self.model3d(volume_3d_input)
                        output = self.model_hff(feat2d, feat3d)
                        loss, loss_dict = self.criterion(output, mask.squeeze(1))
                
                # Gradient accumulation
                should_step = self.grad_accumulator.step(loss, self.optimizer_3d, self.scaler_3d)
                
                if should_step:
                    self.scheduler_3d.step(epoch)
                
                epoch_loss += loss.item()
                
                # Calculate metrics
                with torch.no_grad():
                    dice = dice_coefficient(output, mask.squeeze(1))
                    iou = calculate_iou(output, mask.squeeze(1))
                    dice_scores.append(dice.cpu())
                    iou_scores.append(iou.cpu())
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.mean().item():.4f}',
                    'Stage': f'{self.model_hff.stage}'
                })
            
            # Epoch statistics
            avg_loss = epoch_loss / len(self.train_loader_3d)
            avg_dice = torch.stack(dice_scores).mean(0)
            avg_iou = torch.stack(iou_scores).mean(0)
            epoch_time = time.time() - epoch_start
            
            print(f"3D Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"Dice=[{avg_dice[0]:.3f}, {avg_dice[1]:.3f}, {avg_dice[2]:.3f}], "
                  f"IoU=[{avg_iou[0]:.3f}, {avg_iou[1]:.3f}, {avg_iou[2]:.3f}], "
                  f"Time={epoch_time:.1f}s")
            
            # Wandb logging
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'stage2/train_loss': avg_loss,
                    'stage2/train_dice_bg': avg_dice[0].item(),
                    'stage2/train_dice_liver': avg_dice[1].item(),
                    'stage2/train_dice_tumor': avg_dice[2].item(),
                    'stage2/fusion_stage': self.model_hff.stage,
                    'epoch': self.config['training']['epochs_2d'] + epoch
                })
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self.save_checkpoint('best_stage2', epoch, {
                    'model3d': self.model3d.state_dict(),
                    'model_hff': self.model_hff.state_dict(),
                    'optimizer': self.optimizer_3d.state_dict(),
                    'loss': avg_loss,
                    'dice': avg_dice.mean().item()
                })
        
        print(f"✅ Stage 2 completed! Best Loss: {best_loss:.4f}")
        
    def train_stage3_end_to_end(self):
        """Stage 3: End-to-end fine-tuning with all components"""
        print("\n" + "="*60)
        print("STAGE 3: End-to-End Fine-tuning")
        print("="*60)
        
        # Load best models and unfreeze all
        self.load_checkpoint('best_stage2')
        for param in self.model2d.parameters():
            param.requires_grad = True
        
        self.model2d.train()
        self.model3d.train()
        self.model_hff.train()
        self.model_hff.set_stage(2)  # Full fusion
        
        best_dice = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs_finetune']):
            epoch_start = time.time()
            epoch_loss = 0.0
            dice_scores = []
            iou_scores = []
            
            progress_bar = tqdm(self.train_loader_3d, desc=f"E2E Epoch {epoch+1}")
            
            for batch_idx, (volume, mask) in enumerate(progress_bar):
                volume, mask = volume.to(self.device), mask.to(self.device)
                B, C, D, H, W = volume.shape
                
                self.optimizer_full.zero_grad()
                
                with autocast():
                    # Full forward pass
                    feat2d = create_2d_features_from_slices(volume, self.model2d, training=True)
                    volume_3d_input = torch.cat([volume, feat2d], dim=1)
                    feat3d = self.model3d(volume_3d_input)
                    output = self.model_hff(feat2d, feat3d)
                    
                    # Main loss
                    loss, loss_dict = self.criterion(output, mask.squeeze(1))
                    
                    # Add consistency regularization
                    if np.random.random() < 0.3:  # 30% chance
                        with torch.no_grad():
                            feat2d_aug = create_2d_features_from_slices(volume, self.model2d, training=False)
                        volume_3d_aug = torch.cat([volume, feat2d_aug], dim=1)
                        feat3d_aug = self.model3d(volume_3d_aug)
                        output_aug = self.model_hff(feat2d_aug, feat3d_aug)
                        
                        consistency_loss = self.criterion.consistency_loss(output, output_aug)
                        loss += 0.1 * consistency_loss
                
                # Gradient accumulation with gradient clipping
                should_step = self.grad_accumulator.step(loss, self.optimizer_full, self.scaler_full)
                
                if should_step:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model2d.parameters()) + 
                        list(self.model3d.parameters()) + 
                        list(self.model_hff.parameters()),
                        max_norm=1.0
                    )
                
                epoch_loss += loss.item()
                
                # Calculate metrics
                with torch.no_grad():
                    dice = dice_coefficient(output, mask.squeeze(1))
                    iou = calculate_iou(output, mask.squeeze(1))
                    dice_scores.append(dice.cpu())
                    iou_scores.append(iou.cpu())
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice.mean().item():.4f}',
                    'LR': f'{self.optimizer_full.param_groups[0]["lr"]:.6f}'
                })
            
            # Epoch statistics
            avg_loss = epoch_loss / len(self.train_loader_3d)
            avg_dice = torch.stack(dice_scores).mean(0)
            avg_iou = torch.stack(iou_scores).mean(0)
            epoch_time = time.time() - epoch_start
            mean_dice = avg_dice.mean().item()
            
            # Learning rate scheduling
            self.scheduler_full.step(epoch, avg_loss)
            
            print(f"E2E Epoch {epoch+1}: Loss={avg_loss:.4f}, "
                  f"Dice=[{avg_dice[0]:.3f}, {avg_dice[1]:.3f}, {avg_dice[2]:.3f}], "
                  f"IoU=[{avg_iou[0]:.3f}, {avg_iou[1]:.3f}, {avg_iou[2]:.3f}], "
                  f"Time={epoch_time:.1f}s")
            
            # Wandb logging
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'stage3/train_loss': avg_loss,
                    'stage3/train_dice_bg': avg_dice[0].item(),
                    'stage3/train_dice_liver': avg_dice[1].item(),
                    'stage3/train_dice_tumor': avg_dice[2].item(),
                    'stage3/learning_rate': self.optimizer_full.param_groups[0]['lr'],
                    'epoch': self.config['training']['epochs_2d'] + self.config['training']['epochs_3d'] + epoch
                })
            
            # Save best model and early stopping
            if mean_dice > best_dice:
                best_dice = mean_dice
                patience_counter = 0
                self.save_checkpoint('best_final', epoch, {
                    'model2d': self.model2d.state_dict(),
                    'model3d': self.model3d.state_dict(),
                    'model_hff': self.model_hff.state_dict(),
                    'optimizer': self.optimizer_full.state_dict(),
                    'loss': avg_loss,
                    'dice': mean_dice,
                    'config': self.config
                })
                print(f"✓ New best model saved! Dice: {mean_dice:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if self.early_stopping(avg_loss):
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"✅ Stage 3 completed! Best Dice: {best_dice:.4f}")
        
    def save_checkpoint(self, name, epoch, state_dict):
        """Save model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pth"
        torch.save({
            'epoch': epoch,
            'state_dict': state_dict,
            'history': self.history
        }, checkpoint_path)
        
    def load_checkpoint(self, name):
        """Load model checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{name}.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            
            if 'model2d' in checkpoint['state_dict']:
                self.model2d.load_state_dict(checkpoint['state_dict']['model2d'])
            if 'model3d' in checkpoint['state_dict']:
                self.model3d.load_state_dict(checkpoint['state_dict']['model3d'])
            if 'model_hff' in checkpoint['state_dict']:
                self.model_hff.load_state_dict(checkpoint['state_dict']['model_hff'])
                
            print(f"✓ Loaded checkpoint: {name}")
        else:
            print(f"⚠️ Checkpoint not found: {name}")
            
    def save_training_history(self):
        """Save training history"""
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
        
    def train(self):
        """Complete training pipeline"""
        print("Starting Enhanced H-DenseUNet Training Pipeline")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        total_start = time.time()
        
        try:
            # Stage 1: 2D Training
            self.train_stage1_2d()
            
            # Stage 2: 3D + HFF Training
            self.train_stage2_3d_hff()
            
            # Stage 3: End-to-End Fine-tuning
            self.train_stage3_end_to_end()
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise
        
        total_time = time.time() - total_start
        print(f"\n✅ Complete training pipeline finished!")
        print(f"Total training time: {total_time/3600:.1f} hours")
        
        # Save final results
        self.save_training_history()
        
        # Save final combined model
        torch.save({
            'model2d': self.model2d.state_dict(),
            'model3d': self.model3d.state_dict(),
            'model_hff': self.model_hff.state_dict(),
            'config': self.config,
            'history': self.history
        }, 'hdenseunet_final.pth')
        
        if self.config['logging']['use_wandb']:
            wandb.finish()

def load_config(config_path):
    """Load and validate configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add default optimization parameters
    defaults = {
        'model': {
            'growth_rate_2d': 32,
            'growth_rate_3d': 16,
            'num_layers_2d': 4,
            'num_layers_3d': 4,
            'use_deep_supervision': True
        },
        'training': {
            'accumulation_steps': 4,
            'num_workers': 4,
            'weight_decay': 1e-4,
            'early_stopping_patience': 15,
            'class_weights': [1.0, 1.5, 2.0],
            'output_dir': 'outputs'
        },
        'logging': {
            'use_wandb': False,
            'project_name': 'h-denseunet',
            'run_name': 'enhanced_training'
        }
    }
    
    # Merge defaults
    for key, value in defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Enhanced H-DenseUNet Training')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Configuration file path')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override wandb setting
    if args.wandb:
        config['logging']['use_wandb'] = True
    
    # Set random seed
    set_seed(args.seed)
    
    # Initialize trainer
    trainer = AdvancedTrainer(config, device=args.device)
    
    # Start training
    trainer.train()

if __name__ == '__main__':
    main()