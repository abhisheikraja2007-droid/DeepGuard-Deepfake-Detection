"""
Video Processor Module
Handles video loading, frame extraction, and preprocessing.
"""

import cv2
import numpy as np
from typing import Generator, List, Tuple, Optional, Dict
from pathlib import Path
import warnings


class VideoProcessor:
    """
    Video processing utilities for deepfake detection.
    Supports MP4, AVI, MOV formats and webcam feeds.
    """
    
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv']
    
    def __init__(self, target_fps: Optional[float] = None,
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True):
        """
        Initialize Video Processor.
        
        Args:
            target_fps: Target FPS for processing (None = original)
            target_size: Target frame size (W, H)
            normalize: Normalize pixel values to [0, 1]
        """
        self.target_fps = target_fps
        self.target_size = target_size
        self.normalize = normalize
    
    def load_video(self, video_path: str) -> cv2.VideoCapture:
        """
        Load video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            OpenCV VideoCapture object
            
        Raises:
            ValueError: If file format not supported
            FileNotFoundError: If file doesn't exist
        """
        path = Path(video_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {path.suffix}. "
                           f"Supported: {self.SUPPORTED_FORMATS}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        return cap
    
    def get_video_info(self, cap: cv2.VideoCapture) -> Dict:
        """
        Get video metadata.
        
        Args:
            cap: VideoCapture object
            
        Returns:
            Dictionary with video info
        """
        return {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
            if cap.get(cv2.CAP_PROP_FPS) > 0 else 0
        }
    
    def extract_frames(self, video_path: str,
                       max_frames: Optional[int] = None,
                       sample_rate: int = 1) -> Generator[np.ndarray, None, None]:
        """
        Extract frames from video file.
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            sample_rate: Extract every Nth frame
            
        Yields:
            Frame as numpy array (H, W, C) in BGR format
        """
        cap = self.load_video(video_path)
        
        try:
            frame_idx = 0
            extracted = 0
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    if self.target_size and (frame.shape[1], frame.shape[0]) != self.target_size:
                        frame = cv2.resize(frame, self.target_size)
                    
                    yield frame
                    extracted += 1
                    
                    if max_frames and extracted >= max_frames:
                        break
                
                frame_idx += 1
                
        finally:
            cap.release()
    
    def extract_frames_batch(self, video_path: str,
                              batch_size: int = 32,
                              **kwargs) -> Generator[List[np.ndarray], None, None]:
        """
        Extract frames in batches.
        
        Args:
            video_path: Path to video file
            batch_size: Number of frames per batch
            **kwargs: Additional arguments for extract_frames
            
        Yields:
            List of frames
        """
        batch = []
        
        for frame in self.extract_frames(video_path, **kwargs):
            batch.append(frame)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def extract_all_frames(self, video_path: str, **kwargs) -> List[np.ndarray]:
        """
        Extract all frames into memory.
        
        Args:
            video_path: Path to video file
            **kwargs: Additional arguments
            
        Returns:
            List of all frames
        """
        return list(self.extract_frames(video_path, **kwargs))
    
    def preprocess_frame(self, frame: np.ndarray,
                         to_rgb: bool = True) -> np.ndarray:
        """
        Preprocess single frame for model input.
        
        Args:
            frame: Input frame (H, W, C) BGR
            to_rgb: Convert to RGB
            
        Returns:
            Preprocessed frame
        """
        # Resize if needed
        if self.target_size and (frame.shape[1], frame.shape[0]) != self.target_size:
            frame = cv2.resize(frame, self.target_size)
        
        # Convert BGR to RGB
        if to_rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def preprocess_batch(self, frames: List[np.ndarray],
                         to_tensor: bool = True) -> np.ndarray:
        """
        Preprocess batch of frames.
        
        Args:
            frames: List of frames
            to_tensor: Convert to tensor format (N, C, H, W)
            
        Returns:
            Preprocessed batch
        """
        processed = [self.preprocess_frame(f) for f in frames]
        batch = np.stack(processed, axis=0)
        
        if to_tensor:
            # Convert (N, H, W, C) to (N, C, H, W)
            batch = np.transpose(batch, (0, 3, 1, 2))
        
        return batch
    
    def extract_audio(self, video_path: str,
                       output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Extract audio from video file.
        
        Args:
            video_path: Path to video file
            output_path: Optional path to save audio
            
        Returns:
            Tuple of (audio_waveform, sample_rate)
        """
        try:
            import librosa
            
            # Load audio from video
            audio, sr = librosa.load(video_path, sr=16000, mono=True)
            
            if output_path:
                import soundfile as sf
                sf.write(output_path, audio, sr)
            
            return audio, sr
            
        except Exception as e:
            warnings.warn(f"Could not extract audio: {e}")
            return np.array([]), 16000


class WebcamProcessor:
    """
    Real-time webcam processing for live deepfake detection.
    """
    
    def __init__(self, camera_id: int = 0,
                 target_size: Tuple[int, int] = (224, 224),
                 buffer_size: int = 5):
        """
        Initialize Webcam Processor.
        
        Args:
            camera_id: Camera device ID
            target_size: Target frame size
            buffer_size: Frame buffer size for temporal analysis
        """
        self.camera_id = camera_id
        self.target_size = target_size
        self.buffer_size = buffer_size
        self._cap = None
        self._frame_buffer = []
    
    def start(self) -> bool:
        """
        Start webcam capture.
        
        Returns:
            True if started successfully
        """
        self._cap = cv2.VideoCapture(self.camera_id)
        
        if not self._cap.isOpened():
            warnings.warn(f"Could not open camera {self.camera_id}")
            return False
        
        # Set camera properties
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_size[1])
        
        return True
    
    def stop(self):
        """Stop webcam capture."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._frame_buffer.clear()
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read single frame from webcam.
        
        Returns:
            Frame or None if failed
        """
        if self._cap is None:
            return None
        
        ret, frame = self._cap.read()
        
        if not ret:
            return None
        
        # Resize if different from target
        if (frame.shape[1], frame.shape[0]) != self.target_size:
            frame = cv2.resize(frame, self.target_size)
        
        # Update buffer
        self._frame_buffer.append(frame)
        if len(self._frame_buffer) > self.buffer_size:
            self._frame_buffer.pop(0)
        
        return frame
    
    def get_buffer(self) -> List[np.ndarray]:
        """Get current frame buffer."""
        return self._frame_buffer.copy()
    
    def stream_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Stream frames continuously.
        
        Yields:
            Frames from webcam
        """
        while True:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame
    
    def is_active(self) -> bool:
        """Check if webcam is active."""
        return self._cap is not None and self._cap.isOpened()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FrameNormalizer:
    """
    Frame normalization utilities for different model requirements.
    """
    
    # Standard normalization parameters
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    @staticmethod
    def normalize_imagenet(frame: np.ndarray) -> np.ndarray:
        """
        Normalize frame using ImageNet statistics.
        
        Args:
            frame: Frame in [0, 1] range, shape (H, W, C)
            
        Returns:
            Normalized frame
        """
        mean = np.array(FrameNormalizer.IMAGENET_MEAN)
        std = np.array(FrameNormalizer.IMAGENET_STD)
        
        return (frame - mean) / std
    
    @staticmethod
    def denormalize_imagenet(frame: np.ndarray) -> np.ndarray:
        """
        Denormalize ImageNet-normalized frame.
        
        Args:
            frame: Normalized frame
            
        Returns:
            Frame in [0, 1] range
        """
        mean = np.array(FrameNormalizer.IMAGENET_MEAN)
        std = np.array(FrameNormalizer.IMAGENET_STD)
        
        return frame * std + mean
    
    @staticmethod
    def normalize_minmax(frame: np.ndarray) -> np.ndarray:
        """Normalize to [-1, 1] range."""
        return (frame.astype(np.float32) / 127.5) - 1.0
    
    @staticmethod
    def normalize_standard(frame: np.ndarray) -> np.ndarray:
        """Normalize to [0, 1] range."""
        return frame.astype(np.float32) / 255.0
