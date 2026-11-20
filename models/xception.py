"""
Accuracy Mode - Xception-based Deepfake Detector
Uses Xception architecture with depthwise separable convolutions.
Highest accuracy, target: 5-10 FPS on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SeparableConv2d(nn.Module):
    """Depthwise separable convolution."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, 
                 padding: int = 1, bias: bool = False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class EntryFlowBlock(nn.Module):
    """Entry flow block with residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 first_relu: bool = True, downsample: bool = True):
        super().__init__()
        
        self.first_relu = first_relu
        stride = 2 if downsample else 1
        
        # Main path
        self.sep_conv1 = SeparableConv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.sep_conv2 = SeparableConv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, stride, 1) if downsample else nn.Identity()
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if in_channels != out_channels or downsample else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        if self.first_relu:
            x = F.relu(x)
        x = self.sep_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.sep_conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        
        return x + residual


class MiddleFlowBlock(nn.Module):
    """Middle flow block - repeated 8 times in Xception."""
    
    def __init__(self, channels: int = 728):
        super().__init__()
        
        self.sep_conv1 = SeparableConv2d(channels, channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.sep_conv2 = SeparableConv2d(channels, channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.sep_conv3 = SeparableConv2d(channels, channels, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        x = F.relu(x)
        x = self.sep_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.sep_conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.sep_conv3(x)
        x = self.bn3(x)
        
        return x + residual


class ExitFlowBlock(nn.Module):
    """Exit flow block."""
    
    def __init__(self, in_channels: int = 728, mid_channels: int = 728,
                 out_channels: int = 1024):
        super().__init__()
        
        # First separable conv block with pooling
        self.sep_conv1 = SeparableConv2d(in_channels, mid_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.sep_conv2 = SeparableConv2d(mid_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(3, 2, 1)
        
        # Skip connection
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Final separable convs
        self.sep_conv3 = SeparableConv2d(out_channels, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.sep_conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        
        x = F.relu(x)
        x = self.sep_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.sep_conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        
        x = x + residual
        
        x = self.sep_conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.sep_conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        return x


class XceptionDetector(nn.Module):
    """
    Xception-based deepfake detector.
    
    Architecture based on "Xception: Deep Learning with Depthwise
    Separable Convolutions" - optimized for deepfake detection.
    
    Parameters: ~22M
    Target FPS: 5-10 on CPU
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.5,
                 num_middle_blocks: int = 8):
        """
        Initialize Xception detector.
        
        Args:
            num_classes: Number of output classes
            dropout: Dropout rate before final layer
            num_middle_blocks: Number of middle flow blocks (default 8)
        """
        super().__init__()
        
        # Entry flow
        self.entry_conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
        self.entry_bn1 = nn.BatchNorm2d(32)
        self.entry_conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.entry_bn2 = nn.BatchNorm2d(64)
        
        self.entry_block1 = EntryFlowBlock(64, 128, first_relu=False)
        self.entry_block2 = EntryFlowBlock(128, 256)
        self.entry_block3 = EntryFlowBlock(256, 728)
        
        # Middle flow
        self.middle_flow = nn.Sequential(*[
            MiddleFlowBlock(728) for _ in range(num_middle_blocks)
        ])
        
        # Exit flow
        self.exit_flow = ExitFlowBlock(728, 728, 1024)
        
        # Global pooling and classifier
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes)
        )
        
        self._initialize_weights()
        
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
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, H, W)
            return_features: If True, also return intermediate features
            
        Returns:
            Logits tensor, optionally with features
        """
        # Entry flow
        x = self.entry_conv1(x)
        x = self.entry_bn1(x)
        x = F.relu(x)
        x = self.entry_conv2(x)
        x = self.entry_bn2(x)
        x = F.relu(x)
        
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)
        
        # Middle flow
        x = self.middle_flow(x)
        middle_features = x.clone() if return_features else None
        
        # Exit flow
        x = self.exit_flow(x)
        exit_features = x.clone() if return_features else None
        
        # Classification
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        
        if return_features:
            return logits, middle_features, exit_features
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_fake_probability(self, x: torch.Tensor) -> torch.Tensor:
        """Get fake probability score."""
        proba = self.predict_proba(x)
        return proba[:, 1]
    
    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features for explainability.
        
        Returns:
            Tuple of (middle_flow_features, exit_flow_features)
        """
        _, middle, exit_feat = self.forward(x, return_features=True)
        return middle, exit_feat
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class XceptionLite(nn.Module):
    """
    Lighter Xception variant for faster inference.
    Reduced middle blocks and channels.
    """
    
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        
        # Entry flow (reduced channels)
        self.entry_conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False)
        self.entry_bn1 = nn.BatchNorm2d(24)
        self.entry_conv2 = nn.Conv2d(24, 48, 3, padding=1, bias=False)
        self.entry_bn2 = nn.BatchNorm2d(48)
        
        self.entry_block1 = EntryFlowBlock(48, 96, first_relu=False)
        self.entry_block2 = EntryFlowBlock(96, 192)
        self.entry_block3 = EntryFlowBlock(192, 384)
        
        # Middle flow (reduced to 4 blocks)
        self.middle_flow = nn.Sequential(*[
            MiddleFlowBlock(384) for _ in range(4)
        ])
        
        # Simplified exit
        self.exit_conv = nn.Sequential(
            SeparableConv2d(384, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            SeparableConv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.entry_bn1(self.entry_conv1(x)))
        x = F.relu(self.entry_bn2(self.entry_conv2(x)))
        
        x = self.entry_block1(x)
        x = self.entry_block2(x)
        x = self.entry_block3(x)
        x = self.middle_flow(x)
        x = self.exit_conv(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VisionTransformerDetector(nn.Module):
    """
    Vision Transformer (ViT) based deepfake detector.
    Alternative to Xception for highest accuracy mode.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, num_classes: int = 2,
                 embed_dim: int = 384, depth: int = 6,
                 num_heads: int = 6, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, 
                                     kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_output = x[:, 0]
        
        return self.head(cls_output)


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


def create_xception_model(model_type: str = 'xception',
                          pretrained: bool = False,
                          checkpoint_path: Optional[str] = None,
                          device: str = 'cpu') -> nn.Module:
    """
    Factory function to create accuracy mode model.
    
    Args:
        model_type: 'xception', 'xception_lite', or 'vit'
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint
        device: Target device
        
    Returns:
        Model instance
    """
    if model_type == 'xception':
        model = XceptionDetector(num_classes=2)
    elif model_type == 'xception_lite':
        model = XceptionLite(num_classes=2)
    else:
        model = VisionTransformerDetector(num_classes=2)
    
    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)
