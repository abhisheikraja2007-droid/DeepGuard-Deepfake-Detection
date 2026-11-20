"""
Explainability Module
GRAD-CAM heatmaps and visual explanations for deepfake detection.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass


@dataclass
class ExplanationResult:
    """Result from explainability analysis."""
    heatmap: np.ndarray  # GRAD-CAM heatmap (H, W)
    overlay: np.ndarray  # Heatmap overlaid on image (H, W, C)
    attention_regions: List[Dict]  # High attention regions
    explanation_text: str  # Human-readable explanation
    confidence_breakdown: Dict[str, float]  # Contribution of each factor


class GradCAM:
    """
    GRAD-CAM implementation for CNN explainability.
    Generates class-discriminative localization maps.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: Optional[str] = None):
        """
        Initialize GRAD-CAM.
        
        Args:
            model: PyTorch model
            target_layer: Name of layer to visualize (None = last conv layer)
        """
        self.model = model
        self.model.eval()
        
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer
        
        # Find target layer
        self._find_and_hook_layer()
    
    def _find_and_hook_layer(self):
        """Find target layer and register hooks."""
        if self.target_layer:
            # Use specified layer
            layer = dict(self.model.named_modules()).get(self.target_layer)
            if layer is None:
                raise ValueError(f"Layer {self.target_layer} not found")
        else:
            # Find last convolutional layer
            layer = None
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    layer = module
                    self.target_layer = name
            
            if layer is None:
                raise ValueError("No convolutional layer found")
        
        # Register hooks
        layer.register_forward_hook(self._forward_hook)
        layer.register_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        """Store activations during forward pass."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Store gradients during backward pass."""
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor: torch.Tensor, 
                 target_class: Optional[int] = None) -> np.ndarray:
        """
        Generate GRAD-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (B, C, H, W)
            target_class: Target class index (None = predicted class)
            
        Returns:
            Heatmap as numpy array (H, W)
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap / (heatmap.max() + 1e-8)
        
        return heatmap.cpu().numpy()
    
    def generate_overlay(self, input_image: np.ndarray,
                         heatmap: np.ndarray,
                         alpha: float = 0.5,
                         colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            input_image: Original image (H, W, C) in BGR or RGB
            heatmap: GRAD-CAM heatmap
            alpha: Blending factor
            colormap: OpenCV colormap
            
        Returns:
            Overlaid image (H, W, C)
        """
        # Resize heatmap to image size
        heatmap_resized = cv2.resize(heatmap, (input_image.shape[1], input_image.shape[0]))
        
        # Convert to colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap
        )
        
        # Blend with original
        overlay = cv2.addWeighted(input_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay


class ExplainabilityEngine:
    """
    Complete explainability system for deepfake detection.
    Combines GRAD-CAM with region analysis and text explanations.
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize Explainability Engine.
        
        Args:
            model: Detection model
            device: Compute device
        """
        self.model = model
        self.device = torch.device(device)
        self.gradcam = GradCAM(model)
        
        # Facial region definitions (relative coordinates)
        self.facial_regions = {
            'eyes': {'y': (0.15, 0.35), 'x': (0.1, 0.9)},
            'nose': {'y': (0.35, 0.55), 'x': (0.3, 0.7)},
            'mouth': {'y': (0.55, 0.75), 'x': (0.25, 0.75)},
            'cheeks': {'y': (0.35, 0.65), 'x': (0.05, 0.95)},
            'forehead': {'y': (0.0, 0.2), 'x': (0.15, 0.85)},
            'chin': {'y': (0.75, 1.0), 'x': (0.3, 0.7)},
            'boundary': {'y': (0.0, 1.0), 'x': (0.0, 0.15)},
        }
    
    def explain(self, image: np.ndarray,
                face_tensor: torch.Tensor,
                prediction: float) -> ExplanationResult:
        """
        Generate complete explanation for detection result.
        
        Args:
            image: Original face image (H, W, C)
            face_tensor: Preprocessed face tensor
            prediction: Fake probability
            
        Returns:
            ExplanationResult with all explanations
        """
        # Generate GRAD-CAM heatmap
        face_tensor = face_tensor.to(self.device)
        heatmap = self.gradcam.generate(face_tensor, target_class=1)  # class 1 = fake
        
        # Create overlay
        overlay = self.gradcam.generate_overlay(image, heatmap)
        
        # Analyze attention regions
        attention_regions = self._analyze_regions(heatmap)
        
        # Generate confidence breakdown
        confidence_breakdown = self._compute_breakdown(heatmap, prediction)
        
        # Generate text explanation
        explanation_text = self._generate_text(
            attention_regions, confidence_breakdown, prediction
        )
        
        return ExplanationResult(
            heatmap=heatmap,
            overlay=overlay,
            attention_regions=attention_regions,
            explanation_text=explanation_text,
            confidence_breakdown=confidence_breakdown
        )
    
    def _analyze_regions(self, heatmap: np.ndarray) -> List[Dict]:
        """Analyze which facial regions have highest attention."""
        h, w = heatmap.shape
        regions = []
        
        for region_name, coords in self.facial_regions.items():
            y1 = int(coords['y'][0] * h)
            y2 = int(coords['y'][1] * h)
            x1 = int(coords['x'][0] * w)
            x2 = int(coords['x'][1] * w)
            
            region_values = heatmap[y1:y2, x1:x2]
            mean_attention = float(np.mean(region_values))
            max_attention = float(np.max(region_values))
            
            regions.append({
                'name': region_name,
                'mean_attention': mean_attention,
                'max_attention': max_attention,
                'bbox': [x1, y1, x2, y2],
                'importance': 'high' if mean_attention > 0.5 else 'medium' if mean_attention > 0.2 else 'low'
            })
        
        # Sort by attention level
        regions.sort(key=lambda x: x['mean_attention'], reverse=True)
        
        return regions
    
    def _compute_breakdown(self, heatmap: np.ndarray,
                           prediction: float) -> Dict[str, float]:
        """Compute contribution breakdown."""
        h, w = heatmap.shape
        
        # Calculate regional contributions
        breakdown = {}
        total_attention = np.sum(heatmap)
        
        for region_name, coords in self.facial_regions.items():
            y1 = int(coords['y'][0] * h)
            y2 = int(coords['y'][1] * h)
            x1 = int(coords['x'][0] * w)
            x2 = int(coords['x'][1] * w)
            
            region_attention = np.sum(heatmap[y1:y2, x1:x2])
            contribution = region_attention / (total_attention + 1e-8)
            breakdown[region_name] = float(contribution * prediction)
        
        return breakdown
    
    def _generate_text(self, regions: List[Dict],
                       breakdown: Dict[str, float],
                       prediction: float) -> str:
        """Generate human-readable explanation."""
        lines = []
        
        # Overall assessment
        if prediction < 0.3:
            lines.append("This image appears to be authentic.")
        elif prediction < 0.6:
            lines.append("Some suspicious features detected, but inconclusive.")
        else:
            lines.append("Strong indicators of manipulation detected.")
        
        # Top contributing regions
        high_regions = [r for r in regions if r['importance'] == 'high']
        
        if high_regions:
            region_names = [r['name'].replace('_', ' ') for r in high_regions[:3]]
            lines.append(f"Key areas of concern: {', '.join(region_names)}.")
            
            # Specific observations
            for region in high_regions[:2]:
                name = region['name']
                if name == 'boundary':
                    lines.append("- Face boundary shows blending artifacts.")
                elif name == 'eyes':
                    lines.append("- Eye region shows unnatural characteristics.")
                elif name == 'mouth':
                    lines.append("- Mouth area shows potential lip-sync manipulation.")
                elif name == 'cheeks':
                    lines.append("- Skin texture appears artificially smoothed.")
        else:
            lines.append("No specific manipulation regions identified.")
        
        return " ".join(lines)
    
    def explain_simple(self, image: np.ndarray,
                       face_tensor: torch.Tensor,
                       prediction: float) -> str:
        """
        Generate simple "Explain Like I'm 5" explanation.
        
        Args:
            image: Original image
            face_tensor: Preprocessed tensor
            prediction: Fake probability
            
        Returns:
            Simple explanation string
        """
        result = self.explain(image, face_tensor, prediction)
        
        # Simplify for non-technical users
        if prediction < 0.3:
            return "✅ This looks like a real photo! The face looks natural and consistent."
        elif prediction < 0.5:
            return "🤔 This might be real, but some parts look a bit unusual. Hard to tell for sure."
        elif prediction < 0.7:
            return "⚠️ Something seems off! Parts of the face don't look quite natural. Could be edited."
        else:
            top_region = result.attention_regions[0]['name'] if result.attention_regions else "face"
            return f"🚨 This is likely fake! The {top_region} area looks artificially modified or generated."


class PixelAnomalyDetector:
    """
    Detect pixel-level anomalies that indicate manipulation.
    """
    
    def __init__(self):
        self.noise_patterns = None
    
    def detect_anomalies(self, image: np.ndarray) -> Dict:
        """
        Detect pixel-level anomalies.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Anomaly detection results
        """
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        results = {
            'noise_inconsistency': self._check_noise_consistency(gray),
            'edge_artifacts': self._detect_edge_artifacts(gray),
            'compression_blocks': self._detect_compression_blocks(image),
            'color_anomalies': self._detect_color_anomalies(image) if len(image.shape) == 3 else {}
        }
        
        # Compute overall anomaly score
        scores = [v.get('score', 0) for v in results.values() if isinstance(v, dict)]
        results['overall_score'] = sum(scores) / len(scores) if scores else 0
        
        return results
    
    def _check_noise_consistency(self, gray: np.ndarray) -> Dict:
        """Check for noise level inconsistencies."""
        # Compute local noise variance
        h, w = gray.shape
        block_size = 32
        
        variances = []
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size].astype(float)
                
                # High-pass filter to extract noise
                blur = cv2.GaussianBlur(block, (5, 5), 0)
                noise = block - blur
                variances.append(np.var(noise))
        
        if not variances:
            return {'score': 0, 'detail': 'No blocks analyzed'}
        
        variance_std = np.std(variances)
        variance_mean = np.mean(variances)
        
        # High variance inconsistency suggests manipulation
        inconsistency = variance_std / (variance_mean + 1e-8)
        score = min(1, inconsistency / 2)  # Normalize
        
        return {
            'score': float(score),
            'variance_std': float(variance_std),
            'detail': 'High noise inconsistency' if score > 0.5 else 'Normal noise pattern'
        }
    
    def _detect_edge_artifacts(self, gray: np.ndarray) -> Dict:
        """Detect unnatural edge artifacts."""
        # Sobel edge detection
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        
        # Check for unnaturally sharp edges
        edge_threshold = np.percentile(edges, 95)
        sharp_edges = edges > edge_threshold
        
        # Count sharp edge regions
        contours, _ = cv2.findContours(
            sharp_edges.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Many small sharp regions suggest artificial boundaries
        small_contours = [c for c in contours if cv2.contourArea(c) < 100]
        score = min(1, len(small_contours) / 50)
        
        return {
            'score': float(score),
            'sharp_regions': len(small_contours),
            'detail': 'Potential boundary artifacts' if score > 0.5 else 'Normal edges'
        }
    
    def _detect_compression_blocks(self, image: np.ndarray) -> Dict:
        """Detect JPEG compression blocking artifacts."""
        if len(image.shape) != 3:
            return {'score': 0}
        
        # Convert to YCrCb and analyze Y channel
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y_channel = ycrcb[:, :, 0].astype(float)
        
        # Check 8x8 block boundaries
        h, w = y_channel.shape
        block_diffs = []
        
        for y in range(8, h - 8, 8):
            for x in range(8, w - 8, 8):
                # Horizontal boundary
                h_diff = abs(y_channel[y-1, x] - y_channel[y, x])
                # Vertical boundary
                v_diff = abs(y_channel[y, x-1] - y_channel[y, x])
                block_diffs.append((h_diff + v_diff) / 2)
        
        if not block_diffs:
            return {'score': 0}
        
        blocking_strength = np.mean(block_diffs)
        score = min(1, blocking_strength / 30)
        
        return {
            'score': float(score),
            'blocking_strength': float(blocking_strength),
            'detail': 'Strong compression artifacts' if score > 0.5 else 'Normal compression'
        }
    
    def _detect_color_anomalies(self, image: np.ndarray) -> Dict:
        """Detect color inconsistencies."""
        # Convert to LAB for perceptual color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(float)
        
        # Check color variance in local regions
        h, w = lab.shape[:2]
        block_size = 32
        
        color_vars = []
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block_a = lab[y:y+block_size, x:x+block_size, 1]
                block_b = lab[y:y+block_size, x:x+block_size, 2]
                color_vars.append(np.var(block_a) + np.var(block_b))
        
        if not color_vars:
            return {'score': 0}
        
        var_std = np.std(color_vars)
        score = min(1, var_std / 500)
        
        return {
            'score': float(score),
            'color_variance_std': float(var_std),
            'detail': 'Color inconsistency detected' if score > 0.5 else 'Normal color distribution'
        }
    
    def generate_anomaly_map(self, image: np.ndarray) -> np.ndarray:
        """
        Generate visual anomaly map.
        
        Args:
            image: Input image
            
        Returns:
            Anomaly heatmap (H, W)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        h, w = gray.shape
        anomaly_map = np.zeros((h, w), dtype=np.float32)
        
        block_size = 16
        
        for y in range(0, h - block_size, block_size // 2):
            for x in range(0, w - block_size, block_size // 2):
                block = gray[y:y+block_size, x:x+block_size].astype(float)
                
                # Local noise analysis
                blur = cv2.GaussianBlur(block, (3, 3), 0)
                noise_var = np.var(block - blur)
                
                # Edge analysis
                sobel = cv2.Sobel(block, cv2.CV_64F, 1, 1, ksize=3)
                edge_score = np.mean(np.abs(sobel))
                
                # Combined anomaly score
                anomaly_score = (noise_var / 100 + edge_score / 50) / 2
                anomaly_map[y:y+block_size, x:x+block_size] = max(
                    anomaly_map[y:y+block_size, x:x+block_size].max(),
                    anomaly_score
                )
        
        # Normalize
        anomaly_map = anomaly_map / (anomaly_map.max() + 1e-8)
        
        return anomaly_map
