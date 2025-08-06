#!/usr/bin/env python3
"""
Enhanced 2D DenseUNet with advanced optimizations
- Residual dense blocks with SE attention
- Multi-scale feature fusion
- Improved skip connections
- Memory-efficient implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
# 修复导入问题：导入正确的模块或在本文件中定义

class SEBlock2D(nn.Module):
    """2D version of Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c = x.size(0), x.size(1)
        y = self.global_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class EnhancedDenseBlock2D(nn.Module):
    """Enhanced 2D Dense Block with SE attention and residual connections"""
    def __init__(self, in_channels, growth_rate=32, num_layers=4, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate * 4, 1, bias=False),
                nn.BatchNorm2d(growth_rate * 4),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout_rate),
                nn.Conv2d(growth_rate * 4, growth_rate, 3, padding=1, bias=False),
                SEBlock2D(growth_rate)
            )
            self.layers.append(layer)
            
        # Feature fusion with 1x1 conv
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels + num_layers * growth_rate, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels)
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

class TransitionDown2D(nn.Module):
    """Enhanced transition layer for downsampling"""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) 
        self.dropout = nn.Dropout2d(dropout_rate)
        self.pool = nn.AvgPool2d(2, stride=2)  # Use average pooling instead of max
    
    def forward(self, x):
        x = self.dropout(self.relu(self.bn(self.conv(x))))
        return self.pool(x)

class TransitionUp2D(nn.Module):
    """Enhanced transition layer for upsampling with skip connections"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection processing
        self.skip_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip=None):
        x = self.relu(self.bn(self.conv(x)))
        if skip is not None:
            x = torch.cat([x, skip], 1)
            x = self.skip_conv(x)
        return x

class DenseUNet2D(nn.Module):
    """Enhanced 2D DenseUNet with advanced optimizations"""
    def __init__(self, in_channels=3, out_channels=3, growth_rate=32, num_layers=4):
        super().__init__()
        
        # Initial convolution with larger receptive field
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder path with enhanced dense blocks
        self.dense1 = EnhancedDenseBlock2D(64, growth_rate, num_layers)
        num_features = 64 + num_layers * growth_rate
        self.trans1 = TransitionDown2D(num_features, num_features // 2)
        
        self.dense2 = EnhancedDenseBlock2D(num_features // 2, growth_rate, num_layers)
        num_features = num_features // 2 + num_layers * growth_rate
        self.trans2 = TransitionDown2D(num_features, num_features // 2)
        
        self.dense3 = EnhancedDenseBlock2D(num_features // 2, growth_rate, num_layers)
        num_features = num_features // 2 + num_layers * growth_rate
        self.trans3 = TransitionDown2D(num_features, num_features // 2)
        
        self.dense4 = EnhancedDenseBlock2D(num_features // 2, growth_rate, num_layers)
        num_features = num_features // 2 + num_layers * growth_rate
        self.trans4 = TransitionDown2D(num_features, num_features // 2)
        
        # Bottleneck with enhanced processing
        bottleneck_channels = num_features // 2
        self.bottleneck = nn.Sequential(
            EnhancedDenseBlock2D(bottleneck_channels, growth_rate, num_layers),
            nn.Conv2d(bottleneck_channels + num_layers * growth_rate, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder path with enhanced upsampling
        self.up4 = TransitionUp2D(512, 256)
        self.dense_up4 = EnhancedDenseBlock2D(256, growth_rate // 2, num_layers // 2)
        
        self.up3 = TransitionUp2D(256 + (num_layers // 2) * (growth_rate // 2), 128)
        self.dense_up3 = EnhancedDenseBlock2D(128, growth_rate // 2, num_layers // 2)
        
        self.up2 = TransitionUp2D(128 + (num_layers // 2) * (growth_rate // 2), 96)
        self.dense_up2 = EnhancedDenseBlock2D(96, growth_rate // 2, num_layers // 2)
        
        self.up1 = TransitionUp2D(96 + (num_layers // 2) * (growth_rate // 2), 64)
        self.dense_up1 = EnhancedDenseBlock2D(64, growth_rate // 2, num_layers // 2)
        
        # Final classification layers with deep supervision
        self.final_conv = nn.Sequential(
            nn.Conv2d(64 + (num_layers // 2) * (growth_rate // 2), 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, out_channels, 1)
        )
        
        # Deep supervision heads
        self.aux_head1 = nn.Conv2d(128 + (num_layers // 2) * (growth_rate // 2), out_channels, 1)
        self.aux_head2 = nn.Conv2d(96 + (num_layers // 2) * (growth_rate // 2), out_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x, return_aux=False):
        """Forward pass with optional auxiliary outputs"""
        # Initial processing
        x0 = self.init_conv(x)
        
        # Encoder path
        x1 = self.dense1(x0)
        x1_down = self.trans1(x1)
        
        x2 = self.dense2(x1_down)
        x2_down = self.trans2(x2)
        
        x3 = self.dense3(x2_down)  
        x3_down = self.trans3(x3)
        
        x4 = self.dense4(x3_down)
        x4_down = self.trans4(x4)
        
        # Bottleneck
        bottleneck = self.bottleneck(x4_down)
        
        # Decoder path with skip connections
        up4 = self.up4(bottleneck, x4)
        up4 = self.dense_up4(up4)
        
        up3 = self.up3(up4, x3)
        up3 = self.dense_up3(up3)
        aux1 = self.aux_head1(up3)  # Auxiliary output 1
        
        up2 = self.up2(up3, x2)
        up2 = self.dense_up2(up2)
        aux2 = self.aux_head2(up2)  # Auxiliary output 2
        
        up1 = self.up1(up2, x1)
        up1 = self.dense_up1(up1)
        
        # Final output
        output = self.final_conv(up1)
        
        if return_aux:
            # Return main output and auxiliary outputs for deep supervision
            return output, [aux1, aux2]
        else:
            return output

class DeepSupervisionLoss(nn.Module):
    """Deep supervision loss for multi-scale training"""
    def __init__(self, loss_fn, weights=[1.0, 0.5, 0.25]):
        super().__init__()
        self.loss_fn = loss_fn
        self.weights = weights
        
    def forward(self, main_output, aux_outputs, target):
        # Main loss
        main_loss = self.loss_fn(main_output, target)
        
        # Auxiliary losses
        aux_losses = []
        for i, aux_output in enumerate(aux_outputs):
            # Resize target to match auxiliary output size
            target_resized = F.interpolate(
                target.unsqueeze(1).float(), 
                size=aux_output.shape[2:], 
                mode='nearest'
            ).squeeze(1).long()
            
            aux_loss = self.loss_fn(aux_output, target_resized)
            aux_losses.append(aux_loss * self.weights[i + 1])
        
        # Combined loss
        total_loss = main_loss * self.weights[0] + sum(aux_losses)
        return total_loss