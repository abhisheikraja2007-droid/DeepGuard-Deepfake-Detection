"""
Fast Mode CNN - Lightweight Deepfake Detector
~2M parameters, optimized for 60+ FPS on CPU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and ReLU."""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FastCNN(nn.Module):
    """
    Lightweight CNN for fast deepfake detection.
    
    Architecture:
    - 4 convolutional blocks with increasing channels
    - MaxPooling for downsampling
    - Global Average Pooling
    - Single FC layer for classification
    
    Parameters: ~2M
    Target FPS: 60+ on CPU
    """
    
    def __init__(self, num_classes: int = 2, input_size: int = 224, 
                 dropout: float = 0.3):
        """
        Initialize FastCNN.
        
        Args:
            num_classes: Number of output classes (2 for real/fake)
            input_size: Input image size (224x224)
            dropout: Dropout rate before final layer
        """
        super().__init__()
        
        self.input_size = input_size
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            
            # Block 2: 112x112x32 -> 56x56x64
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            
            # Block 3: 56x56x64 -> 28x28x128
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            
            # Block 4: 28x28x128 -> 14x14x256
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights using Kaiming initialization."""
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get probability predictions.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor (B, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict_fake_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get fake probability score.
        
        Args:
            x: Input tensor
            
        Returns:
            Fake probability tensor (B,)
        """
        proba = self.predict_proba(x)
        return proba[:, 1]  # Assuming class 1 is 'fake'
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature embeddings (for explainability).
        
        Args:
            x: Input tensor
            
        Returns:
            Feature tensor before GAP (B, 256, H, W)
        """
        return self.features(x)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FastCNNWithAttention(nn.Module):
    """
    FastCNN with spatial attention for improved detection.
    Slightly larger but more accurate than base FastCNN.
    """
    
    def __init__(self, num_classes: int = 2, input_size: int = 224,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_size = input_size
        
        # Feature extraction
        self.features = nn.Sequential(
            ConvBlock(3, 32, kernel_size=3, stride=2, padding=1),
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            ConvBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
        )
        
        # Spatial Attention
        self.attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        
        # Apply spatial attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Global pooling and classification
        x = self.gap(features)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_attention_map(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get attention map for visualization.
        
        Returns:
            Tuple of (predictions, attention_map)
        """
        features = self.features(x)
        attention_weights = self.attention(features)
        
        features_attended = features * attention_weights
        pooled = self.gap(features_attended)
        pooled = pooled.view(pooled.size(0), -1)
        logits = self.classifier(pooled)
        
        return logits, attention_weights


def create_fast_model(pretrained: bool = False, 
                      checkpoint_path: Optional[str] = None,
                      device: str = 'cpu') -> FastCNN:
    """
    Factory function to create FastCNN model.
    
    Args:
        pretrained: Load pretrained weights
        checkpoint_path: Path to checkpoint file
        device: Target device
        
    Returns:
        FastCNN model
    """
    model = FastCNN(num_classes=2)
    
    if pretrained and checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(device)
