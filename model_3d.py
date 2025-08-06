#!/usr/bin/env python3
"""
Enhanced 3D DenseUNet with advanced optimizations
- Residual dense blocks with 3D SE attention
- Pyramid pooling for multi-scale context
- Efficient 3D convolutions
- Memory-optimized implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from advanced_modules import ResidualDenseBlock3D, PyramidPoolingModule
import torch.utils.checkpoint as checkpoint

class Efficient3DConv(nn.Module):
    """Memory-efficient 3D convolution using (2+1)D factorization"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, bias=False):
        super().__init__()
        # Spatial convolution first
        self.spatial_conv = nn.Conv3d(
            in_channels, out_channels, 
            kernel_size=(1, kernel_size, kernel_size),
            padding=(0, padding, padding),
            bias=False
        )
        # Temporal convolution
        self.temporal_conv = nn.Conv3d(
            out_channels, out_channels,
            kernel_size=(kernel_size, 1, 1), 
            padding=(padding, 0, 0),
            bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)
        return self.relu(self.bn(x))

class Enhanced3DDenseBlock(nn.Module):
    """Memory-efficient 3D dense block with checkpointing"""
    def __init__(self, in_channels, growth_rate=16, num_layers=4, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm3d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                # Bottleneck layer
                nn.Conv3d(in_channels + i * growth_rate, growth_rate * 4, 1, bias=False),
                nn.BatchNorm3d(growth_rate * 4),
                nn.ReLU(inplace=True),
                nn.Dropout3d(dropout_rate),
                # Efficient 3D convolution
                Efficient3DConv(growth_rate * 4, growth_rate)
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
            # Use gradient checkpointing to save memory
            new_feature = checkpoint.checkpoint(layer, torch.cat(features, 1), use_reentrant=False)
            features.append(new_feature)
        
        # Dense connection
        dense_features = torch.cat(features, 1)
        out = self.fusion(dense_features)
        
        # Residual connection
        return x + out

class AdaptiveTransitionDown3D(nn.Module):
    """Adaptive transition layer with learnable pooling"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate)
        
        # Learnable pooling weights
        self.pool_weights = nn.Parameter(torch.ones(3))  # For D, H, W dimensions
        
    def forward(self, x):
        x = self.dropout(self.relu(self.bn(self.conv(x))))
        
        # Adaptive pooling based on learned weights
        weights = torch.softmax(self.pool_weights, 0)
        
        # Different pooling strategies
        avg_pool = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        max_pool = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Weighted combination
        pooled = weights[0] * avg_pool + weights[1] * max_pool + weights[2] * x[:, :, ::2, ::2, ::2]
        
        return pooled

class TransitionUp3D(nn.Module):
    """Enhanced 3D upsampling with skip connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(1, 2, 2), 
            stride=(1, 2, 2), 
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = self.relu(self.bn(self.conv(x)))
        if skip is not None:
            x = torch.cat([x, skip], 1)
            x = self.skip_conv(x)
        return x

class DenseUNet3D(nn.Module):
    """Enhanced 3D DenseUNet with memory optimization and advanced features"""
    def __init__(self, in_channels=4, out_channels=64, growth_rate=16, num_layers=4):
        super().__init__()
        
        # Efficient initial processing
        self.init_conv = nn.Sequential(
            nn.Conv3d(in_channels, 48, kernel_size=(1, 7, 7), padding=(0, 3, 3), bias=False),
            nn.BatchNorm3d(48),
            nn.ReLU(inplace=True),
            Efficient3DConv(48, 48)
        )
        
        # Encoder path with enhanced dense blocks
        self.dense1 = Enhanced3DDenseBlock(48, growth_rate, num_layers)
        num_features = 48 + num_layers * growth_rate
        self.trans1 = AdaptiveTransitionDown3D(num_features, num_features // 2)
        
        self.dense2 = Enhanced3DDenseBlock(num_features // 2, growth_rate, num_layers)
        num_features = num_features // 2 + num_layers * growth_rate
        self.trans2 = AdaptiveTransitionDown3D(num_features, num_features // 2)
        
        self.dense3 = Enhanced3DDenseBlock(num_features // 2, growth_rate, num_layers)
        num_features = num_features // 2 + num_layers * growth_rate
        self.trans3 = AdaptiveTransitionDown3D(num_features, num_features // 2)
        
        self.dense4 = Enhanced3DDenseBlock(num_features // 2, growth_rate, num_layers)
        num_features = num_features // 2 + num_layers * growth_rate
        self.trans4 = AdaptiveTransitionDown3D(num_features, num_features // 2)
        
        # Enhanced bottleneck with pyramid pooling
        bottleneck_channels = num_features // 2
        self.bottleneck = nn.Sequential(
            Enhanced3DDenseBlock(bottleneck_channels, growth_rate, num_layers),
            PyramidPoolingModule(bottleneck_channels + num_layers * growth_rate),
            nn.Conv3d(bottleneck_channels + num_layers * growth_rate, 256, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path with enhanced blocks
        self.up4 = TransitionUp3D(256, 128)
        self.dense_up4 = Enhanced3DDenseBlock(128, growth_rate // 2, num_layers // 2)
        
        self.up3 = TransitionUp3D(128 + (num_layers // 2) * (growth_rate // 2), 96)
        self.dense_up3 = Enhanced3DDenseBlock(96, growth_rate // 2, num_layers // 2)
        
        self.up2 = TransitionUp3D(96 + (num_layers // 2) * (growth_rate // 2), 80)
        self.dense_up2 = Enhanced3DDenseBlock(80, growth_rate // 2, num_layers // 2)
        
        self.up1 = TransitionUp3D(80 + (num_layers // 2) * (growth_rate // 2), 64)
        self.dense_up1 = Enhanced3DDenseBlock(64, growth_rate // 2, num_layers // 2)
        
        # Final feature processing
        final_channels = 64 + (num_layers // 2) * (growth_rate // 2)
        self.final_conv = nn.Sequential(
            nn.Conv3d(final_channels, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(128, out_channels, 1)
        )
        
        # Auxiliary outputs for deep supervision
        self.aux_head1 = nn.Conv3d(96 + (num_layers // 2) * (growth_rate // 2), out_channels, 1)
        self.aux_head2 = nn.Conv3d(80 + (num_layers // 2) * (growth_rate // 2), out_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, return_aux=False):
        """Forward pass with optional auxiliary outputs and gradient checkpointing"""
        # Initial processing
        x0 = self.init_conv(x)
        
        # Encoder path with checkpointing for memory efficiency
        x1 = checkpoint.checkpoint(self.dense1, x0, use_reentrant=False)
        x1_down = self.trans1(x1)
        
        x2 = checkpoint.checkpoint(self.dense2, x1_down, use_reentrant=False)
        x2_down = self.trans2(x2)
        
        x3 = checkpoint.checkpoint(self.dense3, x2_down, use_reentrant=False)
        x3_down = self.trans3(x3)
        
        x4 = checkpoint.checkpoint(self.dense4, x3_down, use_reentrant=False)
        x4_down = self.trans4(x4)
        
        # Bottleneck
        bottleneck = checkpoint.checkpoint(self.bottleneck, x4_down, use_reentrant=False)
        
        # Decoder path with skip connections
        up4 = self.up4(bottleneck, x4)
        up4 = checkpoint.checkpoint(self.dense_up4, up4, use_reentrant=False)
        
        up3 = self.up3(up4, x3)
        up3 = checkpoint.checkpoint(self.dense_up3, up3, use_reentrant=False)
        aux1 = self.aux_head1(up3)  # Auxiliary output 1
        
        up2 = self.up2(up3, x2)
        up2 = checkpoint.checkpoint(self.dense_up2, up2, use_reentrant=False)
        aux2 = self.aux_head2(up2)  # Auxiliary output 2
        
        up1 = self.up1(up2, x1)
        up1 = checkpoint.checkpoint(self.dense_up1, up1, use_reentrant=False)
        
        # Final output
        output = self.final_conv(up1)
        
        if return_aux:
            return output, [aux1, aux2]
        else:
            return output

class DistillationLoss(nn.Module):
    """Knowledge distillation loss for model compression"""
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        
    def forward(self, student_logits, teacher_logits, target, base_loss_fn):
        # Base task loss
        base_loss = base_loss_fn(student_logits, target)
        
        # Distillation loss
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        return self.alpha * base_loss + (1 - self.alpha) * distill_loss