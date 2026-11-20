"""
Data Augmentation Pipeline for Deepfake Detection
Includes realistic augmentations that simulate deepfake artifacts.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict, Callable
import random


class DeepfakeAugmentor:
    """
    Comprehensive augmentation pipeline for training deepfake detectors.
    Includes both standard augmentations and deepfake-specific artifacts.
    """
    
    def __init__(self, p: float = 0.5, seed: Optional[int] = None):
        """
        Initialize augmentor.
        
        Args:
            p: Base probability for each augmentation
            seed: Random seed for reproducibility
        """
        self.p = p
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def __call__(self, image: np.ndarray, 
                 augmentations: Optional[List[str]] = None) -> np.ndarray:
        """
        Apply augmentations to image.
        
        Args:
            image: Input image (H, W, C) in BGR format
            augmentations: List of augmentation names to apply (None = all)
            
        Returns:
            Augmented image
        """
        if augmentations is None:
            augmentations = ['blur', 'scale', 'color', 'jpeg', 'noise']
        
        aug_map = {
            'blur': self.add_blur,
            'scale': self.random_scale,
            'color': self.color_jitter,
            'jpeg': self.jpeg_compression,
            'noise': self.add_noise,
            'flip': self.horizontal_flip,
            'rotate': self.random_rotation,
            'cutout': self.random_cutout,
            'deepfake_artifact': self.add_deepfake_artifacts
        }
        
        result = image.copy()
        for aug_name in augmentations:
            if aug_name in aug_map and random.random() < self.p:
                result = aug_map[aug_name](result)
        
        return result
    
    def add_blur(self, image: np.ndarray, 
                 blur_type: Optional[str] = None) -> np.ndarray:
        """
        Add blur to simulate compression/processing artifacts.
        
        Args:
            image: Input image
            blur_type: 'gaussian', 'motion', or None for random
            
        Returns:
            Blurred image
        """
        if blur_type is None:
            blur_type = random.choice(['gaussian', 'motion', 'box'])
        
        if blur_type == 'gaussian':
            kernel_size = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        
        elif blur_type == 'motion':
            # Create motion blur kernel
            size = random.randint(5, 15)
            angle = random.uniform(0, 360)
            kernel = self._create_motion_kernel(size, angle)
            return cv2.filter2D(image, -1, kernel)
        
        else:  # box blur
            kernel_size = random.choice([3, 5])
            return cv2.blur(image, (kernel_size, kernel_size))
    
    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel."""
        kernel = np.zeros((size, size))
        center = size // 2
        
        # Calculate line endpoints
        rad = np.deg2rad(angle)
        dx = int(center * np.cos(rad))
        dy = int(center * np.sin(rad))
        
        cv2.line(kernel, (center - dx, center - dy), 
                 (center + dx, center + dy), 1, thickness=1)
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def random_scale(self, image: np.ndarray,
                     scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """
        Random scale augmentation.
        
        Args:
            image: Input image
            scale_range: (min_scale, max_scale)
            
        Returns:
            Scaled image (resized back to original)
        """
        h, w = image.shape[:2]
        scale = random.uniform(*scale_range)
        
        # Scale down then up to simulate resolution changes
        new_h, new_w = int(h * scale), int(w * scale)
        
        if scale < 1:
            # Downscale then upscale
            small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            result = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            # Just slight upscale/crop
            result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # Center crop back to original size
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            result = result[start_h:start_h+h, start_w:start_w+w]
        
        return result
    
    def color_jitter(self, image: np.ndarray,
                     brightness: float = 0.2,
                     contrast: float = 0.2,
                     saturation: float = 0.2,
                     hue: float = 0.1) -> np.ndarray:
        """
        Apply color jittering.
        
        Args:
            image: Input image (BGR)
            brightness: Max brightness change
            contrast: Max contrast change
            saturation: Max saturation change
            hue: Max hue change
            
        Returns:
            Color-augmented image
        """
        result = image.astype(np.float32)
        
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(-brightness, brightness)
            result = result + 255 * factor
        
        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(1 - contrast, 1 + contrast)
            mean = result.mean()
            result = (result - mean) * factor + mean
        
        # Saturation (convert to HSV)
        if random.random() < 0.5:
            hsv = cv2.cvtColor(result.clip(0, 255).astype(np.uint8), 
                               cv2.COLOR_BGR2HSV).astype(np.float32)
            factor = random.uniform(1 - saturation, 1 + saturation)
            hsv[:, :, 1] = hsv[:, :, 1] * factor
            result = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), 
                                   cv2.COLOR_HSV2BGR).astype(np.float32)
        
        # Hue
        if random.random() < 0.5:
            hsv = cv2.cvtColor(result.clip(0, 255).astype(np.uint8), 
                               cv2.COLOR_BGR2HSV).astype(np.float32)
            shift = random.uniform(-hue * 180, hue * 180)
            hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
            result = cv2.cvtColor(hsv.astype(np.uint8), 
                                   cv2.COLOR_HSV2BGR).astype(np.float32)
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def jpeg_compression(self, image: np.ndarray,
                         quality_range: Tuple[int, int] = (40, 95)) -> np.ndarray:
        """
        Simulate JPEG compression artifacts.
        
        Args:
            image: Input image
            quality_range: (min_quality, max_quality)
            
        Returns:
            Compressed image
        """
        quality = random.randint(*quality_range)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return decoded
    
    def add_noise(self, image: np.ndarray,
                  noise_type: Optional[str] = None) -> np.ndarray:
        """
        Add noise to image.
        
        Args:
            image: Input image
            noise_type: 'gaussian', 'salt_pepper', or None for random
            
        Returns:
            Noisy image
        """
        if noise_type is None:
            noise_type = random.choice(['gaussian', 'salt_pepper'])
        
        if noise_type == 'gaussian':
            mean = 0
            std = random.uniform(5, 25)
            noise = np.random.normal(mean, std, image.shape)
            noisy = image.astype(np.float32) + noise
            return np.clip(noisy, 0, 255).astype(np.uint8)
        
        else:  # salt_pepper
            prob = random.uniform(0.01, 0.05)
            noisy = image.copy()
            
            # Salt
            salt_mask = np.random.random(image.shape[:2]) < (prob / 2)
            noisy[salt_mask] = 255
            
            # Pepper
            pepper_mask = np.random.random(image.shape[:2]) < (prob / 2)
            noisy[pepper_mask] = 0
            
            return noisy
    
    def horizontal_flip(self, image: np.ndarray) -> np.ndarray:
        """Horizontal flip."""
        return cv2.flip(image, 1)
    
    def random_rotation(self, image: np.ndarray,
                        max_angle: float = 10) -> np.ndarray:
        """
        Random rotation.
        
        Args:
            image: Input image
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Rotated image
        """
        h, w = image.shape[:2]
        angle = random.uniform(-max_angle, max_angle)
        
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h), 
                                  borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def random_cutout(self, image: np.ndarray,
                      max_holes: int = 3,
                      max_size: int = 30) -> np.ndarray:
        """
        Random cutout augmentation.
        
        Args:
            image: Input image
            max_holes: Maximum number of holes
            max_size: Maximum hole size
            
        Returns:
            Image with random regions blacked out
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        num_holes = random.randint(1, max_holes)
        
        for _ in range(num_holes):
            hole_h = random.randint(10, max_size)
            hole_w = random.randint(10, max_size)
            
            top = random.randint(0, h - hole_h)
            left = random.randint(0, w - hole_w)
            
            result[top:top+hole_h, left:left+hole_w] = 0
        
        return result
    
    def add_deepfake_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Add synthetic deepfake-like artifacts for training.
        Simulates common deepfake visual issues.
        
        Args:
            image: Input image
            
        Returns:
            Image with artificial artifacts
        """
        result = image.copy()
        h, w = image.shape[:2]
        
        artifact_type = random.choice([
            'boundary_blur',
            'color_shift',
            'texture_anomaly',
            'lighting_inconsistency'
        ])
        
        if artifact_type == 'boundary_blur':
            # Simulate face boundary blur
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 3
            
            # Create mask for face region
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            
            # Blur the boundary
            blurred = cv2.GaussianBlur(result, (15, 15), 5)
            
            # Blend at boundary
            boundary_mask = cv2.GaussianBlur(mask, (21, 21), 10)
            boundary_mask = np.stack([boundary_mask] * 3, axis=-1)
            
            result = (result * boundary_mask + blurred * (1 - boundary_mask)).astype(np.uint8)
        
        elif artifact_type == 'color_shift':
            # Localized color shift (common in face swaps)
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 3
            
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            mask = cv2.GaussianBlur(mask, (21, 21), 10)
            
            # Shift colors slightly
            shift = np.array([random.randint(-20, 20) for _ in range(3)])
            shifted = result.astype(np.float32) + shift * mask[:, :, np.newaxis]
            result = np.clip(shifted, 0, 255).astype(np.uint8)
        
        elif artifact_type == 'texture_anomaly':
            # Add subtle texture differences
            center_x, center_y = w // 2, h // 2
            radius = min(w, h) // 4
            
            # Create noise texture
            noise = np.random.randn(h, w) * 10
            noise = cv2.GaussianBlur(noise.astype(np.float32), (5, 5), 1)
            
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), radius, 1, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            
            result = result.astype(np.float32) + noise[:, :, np.newaxis] * mask[:, :, np.newaxis]
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        else:  # lighting_inconsistency
            # Add lighting difference in face region
            center_x, center_y = w // 2, h // 2
            
            # Create gradient
            y_coords = np.arange(h).reshape(-1, 1)
            gradient = (y_coords / h - 0.5) * random.uniform(20, 40)
            
            mask = np.zeros((h, w), dtype=np.float32)
            cv2.circle(mask, (center_x, center_y), min(w, h) // 3, 1, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 5)
            
            result = result.astype(np.float32) + gradient * mask[:, :, np.newaxis]
            result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result


class TemporalAugmentor:
    """Augmentations for video sequences."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
        self.frame_augmentor = DeepfakeAugmentor(p=1.0)
    
    def temporal_sample(self, frames: List[np.ndarray],
                        num_samples: int = 16,
                        strategy: str = 'uniform') -> List[np.ndarray]:
        """
        Sample frames from video.
        
        Args:
            frames: List of video frames
            num_samples: Number of frames to sample
            strategy: 'uniform', 'random', or 'segment'
            
        Returns:
            Sampled frames
        """
        n_frames = len(frames)
        
        if n_frames <= num_samples:
            return frames
        
        if strategy == 'uniform':
            indices = np.linspace(0, n_frames - 1, num_samples, dtype=int)
        elif strategy == 'random':
            indices = sorted(random.sample(range(n_frames), num_samples))
        else:  # segment
            segment_size = n_frames // num_samples
            indices = [random.randint(i * segment_size, (i + 1) * segment_size - 1) 
                      for i in range(num_samples)]
        
        return [frames[i] for i in indices]
    
    def temporal_dropout(self, frames: List[np.ndarray],
                         drop_prob: float = 0.1) -> List[np.ndarray]:
        """
        Randomly repeat frames (simulates frozen frames).
        
        Args:
            frames: List of video frames
            drop_prob: Probability of replacing frame with previous
            
        Returns:
            Augmented frames
        """
        result = [frames[0]]
        
        for i in range(1, len(frames)):
            if random.random() < drop_prob:
                result.append(result[-1].copy())  # Repeat previous
            else:
                result.append(frames[i])
        
        return result
    
    def consistent_augment(self, frames: List[np.ndarray],
                           augmentations: List[str]) -> List[np.ndarray]:
        """
        Apply same augmentation to all frames (for consistency).
        
        Args:
            frames: List of video frames
            augmentations: Augmentations to apply
            
        Returns:
            Augmented frames with consistent parameters
        """
        # Set seed for consistency
        seed = random.randint(0, 10000)
        
        result = []
        for frame in frames:
            random.seed(seed)
            np.random.seed(seed)
            aug_frame = self.frame_augmentor(frame, augmentations)
            result.append(aug_frame)
        
        return result
