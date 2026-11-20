"""
Balanced Mode - EfficientNet-based Deepfake Detector
Uses EfficientNet-B0 architecture for balanced speed/accuracy.
Target: 15-30 FPS on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class Swish(nn.Module):
    """Swish activation function."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, in_channels: int, squeeze_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, 1)
        self.activation = Swish()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = self.activation(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class MBConv(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution (MBConv) block.
    Core building block of EfficientNet.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1,
                 expand_ratio: int = 6, se_ratio: float = 0.25,
                 drop_rate: float = 0.0):
        super().__init__()
        
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_rate = drop_rate
        
        expanded_channels = in_channels * expand_ratio
        
        # Expansion phase
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, expanded_channels, 1, bias=False),
                nn.BatchNorm2d(expanded_channels),
                Swish()
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise convolution
        padding = (kernel_size - 1) // 2
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size,
                     stride, padding, groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            Swish()
        )
        
        # Squeeze-and-Excitation
        squeeze_channels = max(1, int(in_channels * se_ratio))
        self.se = SqueezeExcitation(expanded_channels, squeeze_channels)
        
        # Projection phase
        self.project = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        
        if self.use_residual:
            if self.drop_rate > 0 and self.training:
                x = self._drop_connect(x)
            x = x + identity
            
        return x
    
    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        """Apply drop connect regularization."""
        keep_prob = 1 - self.drop_rate
        mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < keep_prob
        x = x / keep_prob * mask.float()
        return x


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B0 based deepfake detector.
    
    Balanced between speed and accuracy.
    Parameters: ~5.3M
    Target FPS: 15-30 on CPU
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.2,
                 drop_connect_rate: float = 0.2):
        """
        Initialize EfficientNet detector.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate for classifier
            drop_connect_rate: Drop connect rate for MBConv blocks
        """
        super().__init__()
        
        # EfficientNet-B0 configuration
        # (expand_ratio, channels, repeats, stride, kernel_size)
        self.config = [
            (1, 16, 1, 1, 3),    # Stage 1
            (6, 24, 2, 2, 3),    # Stage 2
            (6, 40, 2, 2, 5),    # Stage 3
            (6, 80, 3, 2, 3),    # Stage 4
            (6, 112, 3, 1, 5),   # Stage 5
            (6, 192, 4, 2, 5),   # Stage 6
            (6, 320, 1, 1, 3),   # Stage 7
        ]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # Build MBConv stages
        self.stages = self._build_stages(drop_connect_rate)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )
        
        # Global pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes)
        )
        
        self._initialize_weights()
        
    def _build_stages(self, drop_connect_rate: float) -> nn.ModuleList:
        """Build MBConv stages."""
        stages = nn.ModuleList()
        in_channels = 32
        total_blocks = sum(c[2] for c in self.config)
        block_idx = 0
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in self.config:
            blocks = []
            for i in range(repeats):
                current_stride = stride if i == 0 else 1
                drop_rate = drop_connect_rate * block_idx / total_blocks
                
                blocks.append(MBConv(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=current_stride,
                    expand_ratio=expand_ratio,
                    drop_rate=drop_rate
                ))
                block_idx += 1
                
            stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
            
        return stages
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.stem(x)
        
        for stage in self.stages:
            x = stage(x)
            
        x = self.head(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_fake_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Get fake probability score."""
        proba = self.predict_proba(x)
        return proba[:, 1]
    
    def get_features(self, x: torch.Tensor, layer: int = -1) -> torch.Tensor:
        """
        Extract intermediate features for explainability.
        
        Args:
            x: Input tensor
            layer: Stage index (-1 for final features)
            
        Returns:
            Feature tensor
        """
        x = self.stem(x)
        
        target_layer = layer if layer >= 0 else len(self.stages) + layer
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == target_layer:
                return x
                
        return self.head(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MobileNetV3Detector(nn.Module):
    """
    Alternative balanced detector using MobileNetV3-Small architecture.
    Even faster than EfficientNet but slightly less accurate.
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.2):
        super().__init__()
        
        # Simplified MobileNetV3-Small
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True),
            
            # Inverted residual blocks
            self._make_block(16, 16, 3, 2, 16, use_se=True),
            self._make_block(16, 24, 3, 2, 72, use_se=False),
            self._make_block(24, 24, 3, 1, 88, use_se=False),
            self._make_block(24, 40, 5, 2, 96, use_se=True),
            self._make_block(40, 40, 5, 1, 240, use_se=True),
            self._make_block(40, 48, 5, 1, 120, use_se=True),
            self._make_block(48, 96, 5, 2, 144, use_se=True),
            self._make_block(96, 96, 5, 1, 288, use_se=True),
            
            # Final conv
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True),
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
        
    def _make_block(self, in_ch: int, out_ch: int, kernel: int, 
                    stride: int, exp_ch: int, use_se: bool) -> nn.Sequential:
        """Create inverted residual block."""
        layers = [
            nn.Conv2d(in_ch, exp_ch, 1, bias=False),
            nn.BatchNorm2d(exp_ch),
            nn.Hardswish(inplace=True),
            
            nn.Conv2d(exp_ch, exp_ch, kernel, stride, kernel//2, 
                     groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            nn.Hardswish(inplace=True),
        ]
        
        if use_se:
            layers.append(SqueezeExcitation(exp_ch, exp_ch // 4))
            
        layers.extend([
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_efficient_model(model_type: str = 'efficientnet',
                          pretrained: bool = False,
                          checkpoint_path: Optional[str] = None,
                          device: str = 'cpu') -> nn.Module:
    """
    Factory function to create balanced mode model.
    
    Args:
        model_type: 'efficientnet' or 'mobilenetv3'
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint
        device: Target device
        
    Returns:
        Model instance
    """
    if model_type == 'efficientnet':
        model = EfficientNetDetector(num_classes=2)
    else:
        model = MobileNetV3Detector(num_classes=2)
    
    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)
