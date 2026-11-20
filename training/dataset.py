"""
Training Dataset Module
PyTorch Dataset for deepfake detection training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from typing import Optional, Callable, Tuple, List
import random
import json


class DeepfakeDataset(Dataset):
    """
    PyTorch Dataset for deepfake detection training.
    Supports FaceForensics++, DFDC, and custom datasets.
    """
    
    def __init__(self, root_dir: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 num_frames: int = 16, frame_size: Tuple[int, int] = (224, 224),
                 balance_classes: bool = True):
        """
        Initialize dataset.
        
        Args:
            root_dir: Path to dataset root
            split: 'train', 'val', or 'test'
            transform: Optional augmentation transforms
            num_frames: Frames to sample per video
            frame_size: Output frame size
            balance_classes: Balance real/fake samples
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.num_frames = num_frames
        self.frame_size = frame_size
        
        # Load samples
        self.samples = self._load_samples()
        
        if balance_classes:
            self.samples = self._balance_samples()
    
    def _load_samples(self) -> List[dict]:
        """Load video samples from dataset."""
        samples = []
        split_dir = self.root_dir / self.split
        
        if not split_dir.exists():
            # Try flat structure
            split_dir = self.root_dir
        
        # Check for metadata file
        meta_path = self.root_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
            
            for video_name, info in metadata.items():
                video_path = self.root_dir / video_name
                if video_path.exists():
                    samples.append({
                        'path': str(video_path),
                        'label': 1 if info.get('label') == 'FAKE' else 0,
                        'method': info.get('method', 'unknown')
                    })
        else:
            # Directory-based structure: real/ and fake/
            for label, label_name in [(0, 'real'), (1, 'fake')]:
                label_dir = split_dir / label_name
                if label_dir.exists():
                    for video_path in label_dir.glob('*'):
                        if video_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                            samples.append({
                                'path': str(video_path),
                                'label': label,
                                'method': 'unknown'
                            })
        
        return samples
    
    def _balance_samples(self) -> List[dict]:
        """Balance real and fake samples."""
        real = [s for s in self.samples if s['label'] == 0]
        fake = [s for s in self.samples if s['label'] == 1]
        
        min_count = min(len(real), len(fake))
        
        if min_count == 0:
            return self.samples
        
        random.shuffle(real)
        random.shuffle(fake)
        
        return real[:min_count] + fake[:min_count]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        # Load and sample frames
        frames = self._load_video_frames(sample['path'])
        
        # Stack frames
        if frames:
            frame_tensor = torch.stack([self._process_frame(f) for f in frames])
        else:
            frame_tensor = torch.zeros(self.num_frames, 3, *self.frame_size)
        
        return frame_tensor, sample['label']
    
    def _load_video_frames(self, path: str) -> List[np.ndarray]:
        """Load and sample frames from video."""
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            cap.release()
            return []
        
        # Sample frame indices
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                
                if self.transform:
                    frame = self.transform(frame)
                
                frames.append(frame)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((*self.frame_size, 3), dtype=np.uint8))
        
        return frames[:self.num_frames]
    
    def _process_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to tensor."""
        tensor = torch.from_numpy(frame).float() / 255.0
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        return tensor


class FaceDataset(Dataset):
    """Dataset for face-level deepfake detection (pre-extracted faces)."""
    
    def __init__(self, root_dir: str, split: str = 'train',
                 transform: Optional[Callable] = None,
                 image_size: Tuple[int, int] = (224, 224)):
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.image_size = image_size
        
        self.samples = []
        
        for label, label_name in [(0, 'real'), (1, 'fake')]:
            label_dir = self.root_dir / split / label_name
            if label_dir.exists():
                for img_path in label_dir.glob('*.jpg'):
                    self.samples.append({'path': str(img_path), 'label': label})
                for img_path in label_dir.glob('*.png'):
                    self.samples.append({'path': str(img_path), 'label': label})
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        
        img = cv2.imread(sample['path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        
        if self.transform:
            img = self.transform(img)
        
        tensor = torch.from_numpy(img).float() / 255.0
        tensor = tensor.permute(2, 0, 1)
        
        return tensor, sample['label']
