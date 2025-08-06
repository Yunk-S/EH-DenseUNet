#!/usr/bin/env python3
"""
Enhanced Hybrid Feature Fusion (HFF) module with advanced optimizations
- Multi-head self-attention
- Cross-modal attention between 2D and 3D features
- Feature pyramid fusion
- Adaptive feature recalibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention3D(nn.Module):
    """Multi-head self-attention for 3D features"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # Reshape for attention: [B, D*H*W, C]
        x_flat = x.view(B, C, -1).transpose(1, 2)
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x_flat).chunk(3, dim=-1)
        q, k, v = [t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2) 
                   for t in qkv]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = (attn @ v).transpose(1, 2).contiguous().view(B, -1, C)
        out = self.out_proj(out)
        
        # Reshape back to 3D: [B, C, D, H, W]
        out = out.transpose(1, 2).view(B, C, D, H, W)
        
        return x + out  # Residual connection

class CrossModalAttention(nn.Module):
    """Cross-modal attention between 2D and 3D features"""
    def __init__(self, channels_2d, channels_3d, embed_dim=256):
        super().__init__()
        
        # Project features to common embedding space
        self.proj_2d = nn.Conv3d(channels_2d, embed_dim, 1)
        self.proj_3d = nn.Conv3d(channels_3d, embed_dim, 1)
        
        # Cross-attention layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.cross_attn_2d_to_3d = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.cross_attn_3d_to_2d = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        
        # Output projections
        self.out_proj_2d = nn.Conv3d(embed_dim, channels_2d, 1)
        self.out_proj_3d = nn.Conv3d(embed_dim, channels_3d, 1)
        
    def forward(self, feat2d, feat3d):
        B, _, D, H, W = feat2d.shape
        
        # Project to embedding space
        embed_2d = self.proj_2d(feat2d)  # [B, embed_dim, D, H, W]
        embed_3d = self.proj_3d(feat3d)  # [B, embed_dim, D, H, W]
        
        # Flatten for attention: [B, D*H*W, embed_dim]
        embed_2d_flat = embed_2d.view(B, embed_2d.size(1), -1).transpose(1, 2)
        embed_3d_flat = embed_3d.view(B, embed_3d.size(1), -1).transpose(1, 2)
        
        # Cross-attention
        embed_2d_flat = self.norm1(embed_2d_flat)
        embed_3d_flat = self.norm2(embed_3d_flat)
        
        # 2D queries attend to 3D keys/values
        enhanced_2d, _ = self.cross_attn_2d_to_3d(embed_2d_flat, embed_3d_flat, embed_3d_flat)
        
        # 3D queries attend to 2D keys/values  
        enhanced_3d, _ = self.cross_attn_3d_to_2d(embed_3d_flat, embed_2d_flat, embed_2d_flat)
        
        # Reshape back to 3D
        enhanced_2d = enhanced_2d.transpose(1, 2).view(B, -1, D, H, W)
        enhanced_3d = enhanced_3d.transpose(1, 2).view(B, -1, D, H, W)
        
        # Project back to original dimensions
        out_2d = self.out_proj_2d(enhanced_2d) + feat2d
        out_3d = self.out_proj_3d(enhanced_3d) + feat3d
        
        return out_2d, out_3d

class FeaturePyramidFusion(nn.Module):
    """Feature Pyramid Network-style fusion for multi-scale features"""
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        
        # Lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv3d(in_ch, out_channels, 1) for in_ch in in_channels_list
        ])
        
        # Output convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv3d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from different scales, from high-res to low-res
        """
        # Build laterals
        laterals = [lateral_conv(feat) for lateral_conv, feat in zip(self.lateral_convs, features)]
        
        # Build top-down pathway
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] += F.interpolate(
                laterals[i + 1], 
                size=laterals[i].shape[2:],
                mode='trilinear', 
                align_corners=False
            )
        
        # Build outputs
        outputs = [fpn_conv(lateral) for fpn_conv, lateral in zip(self.fpn_convs, laterals)]
        
        return outputs

class AdaptiveFeatureRecalibration(nn.Module):
    """Adaptive feature recalibration using global context"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        # Global context extraction
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Recalibration network
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
        # Local context extraction
        self.local_conv = nn.Sequential(
            nn.Conv3d(channels, channels // 4, 1),
            nn.BatchNorm3d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Global recalibration
        b, c = x.size(0), x.size(1)
        global_context = self.global_pool(x).view(b, c)
        global_weights = self.fc(global_context).view(b, c, 1, 1, 1)
        
        # Local recalibration
        local_weights = self.local_conv(x)
        
        # Adaptive combination
        combined_weights = global_weights * local_weights
        
        return x * combined_weights

class HybridFeatureFusion(nn.Module):
    """
    Enhanced Hybrid Feature Fusion with state-of-the-art techniques:
    - Multi-head self-attention
    - Cross-modal attention
    - Feature pyramid fusion
    - Adaptive feature recalibration
    """
    def __init__(self, in_channels_2d=3, in_channels_3d=64, out_channels=3):
        super().__init__()
        
        # Feature alignment with enhanced processing
        self.align_2d = nn.Sequential(
            nn.Conv3d(in_channels_2d, 64, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            AdaptiveFeatureRecalibration(64)
        )
        
        self.align_3d = nn.Sequential(
            nn.Conv3d(in_channels_3d, 64, 1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            AdaptiveFeatureRecalibration(64)
        )
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(64, 64, embed_dim=128)
        
        # Self-attention for enhanced features
        self.self_attention_2d = MultiHeadAttention3D(64, num_heads=8)
        self.self_attention_3d = MultiHeadAttention3D(64, num_heads=8)
        
        # Multi-scale fusion pathway
        self.fusion_conv1 = nn.Sequential(
            nn.Conv3d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
        
        self.fusion_conv2 = nn.Sequential(
            nn.Conv3d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1)
        )
        
        self.fusion_conv3 = nn.Sequential(
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True)
        )
        
        # Feature pyramid fusion
        self.fpn = FeaturePyramidFusion([128, 256, 128, 64], out_channels=64)
        
        # Skip connections with attention
        self.skip_attention1 = AdaptiveFeatureRecalibration(128)
        self.skip_attention2 = AdaptiveFeatureRecalibration(256)
        self.skip_attention3 = AdaptiveFeatureRecalibration(128)
        
        # Final prediction layers with deep supervision
        self.final_conv = nn.Sequential(
            nn.Conv3d(64 * 4, 128, 3, padding=1, bias=False),  # 4 pyramid levels
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Dropout3d(0.1),
            nn.Conv3d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, out_channels, 1)
        )
        
        # Auxiliary prediction heads for deep supervision
        self.aux_head1 = nn.Conv3d(256, out_channels, 1)
        self.aux_head2 = nn.Conv3d(128, out_channels, 1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, feat2d, feat3d, return_aux=False):
        """
        Enhanced forward pass with multiple attention mechanisms
        
        Args:
            feat2d: 2D features [B, C_2d, D, H, W]
            feat3d: 3D features [B, C_3d, D, H, W]
            return_aux: Whether to return auxiliary outputs
        
        Returns:
            fused_output: [B, out_channels, D, H, W]
            aux_outputs: List of auxiliary outputs (if return_aux=True)
        """
        # Feature alignment with adaptive recalibration
        feat2d_aligned = self.align_2d(feat2d)  # [B, 64, D, H, W]
        feat3d_aligned = self.align_3d(feat3d)  # [B, 64, D, H, W]
        
        # Cross-modal attention between 2D and 3D features
        feat2d_cross, feat3d_cross = self.cross_attention(feat2d_aligned, feat3d_aligned)
        
        # Self-attention for enhanced feature representation
        feat2d_enhanced = self.self_attention_2d(feat2d_cross)
        feat3d_enhanced = self.self_attention_3d(feat3d_cross)
        
        # Initial fusion
        fused = torch.cat([feat2d_enhanced, feat3d_enhanced], dim=1)  # [B, 128, D, H, W]
        skip1_input = self.skip_attention1(fused)
        
        # Multi-scale processing with skip connections
        x1 = self.fusion_conv1(fused)  # [B, 256, D, H, W]
        skip2_input = self.skip_attention2(x1)
        aux1 = self.aux_head1(x1)  # Auxiliary output 1
        
        x2 = self.fusion_conv2(x1)     # [B, 128, D, H, W]
        skip3_input = self.skip_attention3(x2)
        aux2 = self.aux_head2(x2)      # Auxiliary output 2
        
        x3 = self.fusion_conv3(x2)     # [B, 64, D, H, W]
        
        # Feature pyramid fusion
        pyramid_features = [skip1_input, skip2_input, skip3_input, x3]
        fpn_outputs = self.fpn(pyramid_features)
        
        # Concatenate all pyramid levels
        pyramid_concat = torch.cat(fpn_outputs, dim=1)  # [B, 64*4, D, H, W]
        
        # Final prediction
        output = self.final_conv(pyramid_concat)
        
        if return_aux:
            return output, [aux1, aux2]
        else:
            return output

class ProgressiveFeatureFusion(nn.Module):
    """Progressive feature fusion for curriculum learning"""
    def __init__(self, base_hff):
        super().__init__()
        self.base_hff = base_hff
        self.stage = 0  # 0: 2D only, 1: 3D only, 2: full fusion
        
    def set_stage(self, stage):
        """Set the current training stage"""
        self.stage = stage
        
    def forward(self, feat2d, feat3d, return_aux=False):
        if self.stage == 0:
            # Stage 1: Use only 2D features
            dummy_3d = torch.zeros_like(feat3d)
            return self.base_hff(feat2d, dummy_3d, return_aux)
        elif self.stage == 1:
            # Stage 2: Use only 3D features
            dummy_2d = torch.zeros_like(feat2d)
            return self.base_hff(dummy_2d, feat3d, return_aux)
        else:
            # Stage 3: Full fusion
            return self.base_hff(feat2d, feat3d, return_aux)