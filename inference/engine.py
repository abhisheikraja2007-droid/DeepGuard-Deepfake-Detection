"""
Main Inference Engine
Orchestrates deepfake detection across all models and modalities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import warnings
import time

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.face_detector import FaceDetector
from models.fast_cnn import FastCNN, create_fast_model
from models.efficient_net import EfficientNetDetector, create_efficient_model
from models.xception import XceptionDetector, create_xception_model
from models.audio_analyzer import AudioAnalyzer
from models.fusion import FusionEngine, FrameResult, AudioResult, FusionResult, DetectionMode
from preprocessing.video_processor import VideoProcessor, WebcamProcessor


from inference.anti_spoof import AntiSpoofDetector

@dataclass
class InferenceConfig:
    """Configuration for inference engine."""
    mode: DetectionMode = DetectionMode.BALANCED
    device: str = 'cpu'
    face_detector_threshold: float = 0.7
    batch_size: int = 8
    frame_sample_rate: int = 3  # Process every Nth frame
    analyze_audio: bool = True
    use_onnx: bool = False
    model_paths: Optional[Dict[str, str]] = None


class InferenceEngine:
    """
    Main inference engine for deepfake detection.
    Supports multiple detection modes and ONNX export.
    """
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        """
        Initialize Inference Engine.
        
        Args:
            config: Inference configuration
        """
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize components
        self.face_detector = FaceDetector(device=self.config.device)
        self.video_processor = VideoProcessor(target_size=(224, 224))
        self.audio_analyzer = AudioAnalyzer(device=self.config.device) if self.config.analyze_audio else None
        self.fusion_engine = FusionEngine()
        self.anti_spoof = AntiSpoofDetector()
        
        # Load appropriate model based on mode
        self.model = self._load_model()
        self.model.eval()
        
        # ONNX session (if using ONNX)
        self.onnx_session = None
        if self.config.use_onnx and self.config.model_paths:
            self._load_onnx_model()
    
    def _load_model(self) -> torch.nn.Module:
        """Load detection model based on mode."""
        if self.config.mode == DetectionMode.FAST:
            model = create_fast_model(device=self.config.device)
        elif self.config.mode == DetectionMode.BALANCED:
            model = create_efficient_model(device=self.config.device)
        else:  # ACCURACY
            model = create_xception_model(device=self.config.device)
        
        # Load weights if provided
        if self.config.model_paths:
            mode_key = self.config.mode.value
            if mode_key in self.config.model_paths:
                checkpoint = torch.load(
                    self.config.model_paths[mode_key],
                    map_location=self.device
                )
                model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(self.device)
    
    def _load_onnx_model(self):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            
            mode_key = self.config.mode.value
            if mode_key in self.config.model_paths:
                onnx_path = self.config.model_paths[mode_key].replace('.pth', '.onnx')
                if Path(onnx_path).exists():
                    self.onnx_session = ort.InferenceSession(onnx_path)
        except ImportError:
            warnings.warn("ONNX Runtime not available")
    
    def analyze_video(self, video_path: str,
                      progress_callback: Optional[callable] = None) -> FusionResult:
        """
        Analyze video file for deepfakes.
        """
        start_time = time.time()
        
        # Extract and process frames
        if progress_callback:
            progress_callback(0.1, "Extracting frames...")
        
        frames = self.video_processor.extract_all_frames(
            video_path,
            sample_rate=self.config.frame_sample_rate,
            max_frames=300  # Limit for memory
        )
        
        if not frames:
            return self._empty_result("No frames extracted from video")
        
        # Run global anti-spoofing checks (replay, frozen)
        replay_result = self.anti_spoof.detect_replay_attack(frames)
        frozen_result = self.anti_spoof.detect_frozen_frames(frames)
        
        # Analyze frames
        if progress_callback:
            progress_callback(0.3, "Analyzing faces...")
        
        frame_results = self._analyze_frames(frames, progress_callback)
        
        # Inject global replay scores into frame results
        for res in frame_results:
            res.replay_score = max(replay_result['score'], frozen_result['score'])
        
        # Analyze audio
        audio_result = None
        lip_sync_score = 0.0
        
        if self.config.analyze_audio and self.audio_analyzer:
            if progress_callback:
                progress_callback(0.7, "Analyzing audio...")
            
            try:
                audio, sr = self.video_processor.extract_audio(video_path)
                if len(audio) > 0:
                    audio_analysis = self.audio_analyzer.analyze_audio(audio, sr)
                    
                    # Run lip sync check if faces detected
                    if frame_results:
                         # Placeholder for advanced lip sync (requires landmarks history)
                         # Simple version: check rhythmic consistency
                         pass

                    audio_result = AudioResult(
                        fake_probability=audio_analysis['audio_fake_probability'],
                        lip_sync_score=lip_sync_score,
                        rhythm_anomaly=audio_analysis['rhythm_analysis']['synthetic_score'] if audio_analysis['rhythm_analysis'] else 0,
                        artifacts=audio_analysis['artifacts_detected'],
                        confidence=audio_analysis['confidence']
                    )
            except Exception as e:
                warnings.warn(f"Audio analysis failed: {e}")
        
        # Fuse results
        if progress_callback:
            progress_callback(0.9, "Computing final score...")
        
        result = self.fusion_engine.fuse(
            frame_results,
            audio_result,
            lip_sync_score,
            mode=self.config.mode
        )
        
        elapsed = time.time() - start_time
        
        if progress_callback:
            progress_callback(1.0, f"Analysis complete in {elapsed:.1f}s")
        
        return result
    
    def _analyze_frames(self, frames: List[np.ndarray],
                        progress_callback: Optional[callable] = None) -> List[FrameResult]:
        """Analyze all frames for deepfake indicators."""
        frame_results = []
        total = len(frames)
        
        # Process in batches
        for batch_start in range(0, total, self.config.batch_size):
            batch_end = min(batch_start + self.config.batch_size, total)
            batch_frames = frames[batch_start:batch_end]
            
            # Detect faces and analyze each frame
            for i, frame in enumerate(batch_frames):
                frame_idx = batch_start + i
                result = self._analyze_single_frame(frame, frame_idx)
                frame_results.append(result)
                
                if progress_callback:
                    progress = 0.3 + (0.4 * (frame_idx + 1) / total)
                    progress_callback(progress, f"Processing frame {frame_idx + 1}/{total}")
        
        return frame_results
    
    def _analyze_single_frame(self, frame: np.ndarray,
                               frame_idx: int) -> FrameResult:
        """Analyze single frame."""
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        if not faces:
            return FrameResult(
                frame_idx=frame_idx,
                fake_probability=0.0,
                face_detected=False,
                confidence=0.0,
                bbox=None,
                texture_anomaly=0.0,
                replay_score=0.0
            )
        
        # Use primary (largest) face
        primary_face = max(faces, key=lambda f: 
                          (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        
        # Preprocess face for model
        face_img = primary_face['aligned_face']
        face_tensor = self._preprocess_face(face_img)
        
        # Run inference
        with torch.no_grad():
            if self.onnx_session:
                fake_prob = self._onnx_inference(face_tensor)
            else:
                fake_prob = self._pytorch_inference(face_tensor)
        
        # Heuristic Checks via Anti-Spoof
        texture_result = self.anti_spoof.detect_texture_anomaly(face_img)
        
        return FrameResult(
            frame_idx=frame_idx,
            fake_probability=fake_prob,
            face_detected=True,
            confidence=primary_face['confidence'],
            bbox=primary_face['bbox'],
            texture_anomaly=texture_result['score'],
            replay_score=0.0 # Will be populated by batch checks
        )
    
    def _preprocess_face(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess face image for inference."""
        # Resize if needed
        if face_img.shape[0] != 224 or face_img.shape[1] != 224:
            import cv2
            face_img = cv2.resize(face_img, (224, 224))
        
        # Normalize to [0, 1]
        face_tensor = torch.FloatTensor(face_img).permute(2, 0, 1) / 255.0
        
        # Add batch dimension
        face_tensor = face_tensor.unsqueeze(0).to(self.device)
        
        return face_tensor
    
    def _pytorch_inference(self, face_tensor: torch.Tensor) -> float:
        """Run PyTorch inference."""
        logits = self.model(face_tensor)
        probs = F.softmax(logits, dim=1)
        fake_prob = probs[0, 1].item()  # Assuming class 1 is fake
        return fake_prob
    
    def _onnx_inference(self, face_tensor: torch.Tensor) -> float:
        """Run ONNX inference."""
        input_name = self.onnx_session.get_inputs()[0].name
        input_data = face_tensor.cpu().numpy()
        
        outputs = self.onnx_session.run(None, {input_name: input_data})
        logits = outputs[0]
        
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        
        return float(probs[0, 1])
    
    def analyze_webcam_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze single webcam frame.
        """
        result = self._analyze_single_frame(frame, 0)
        
        # If no face, we return safe 0s, but frontend handles the "Scanning" UI.
        # If face detected, we mix heuristics.
        
        final_score = result.fake_probability
        
        # Boost if texture anomaly is high (since webcam model is likely untrained)
        if result.face_detected:
            if result.texture_anomaly > 0.6:
                final_score = max(final_score, result.texture_anomaly)
            elif result.fake_probability < 0.5:
                # If model is uncertain, bias towards safe for webcam unless anomalies found
                 final_score = final_score * 0.8 # Reduce noise
        
        return {
            'fake_probability': float(final_score),
            'face_detected': result.face_detected,
            'bbox': result.bbox,
            'risk_level': 'low' if final_score < 0.3 
                         else 'medium' if final_score < 0.6
                         else 'high'
        }
    
    def set_mode(self, mode: Union[str, DetectionMode]):
        """
        Change detection mode.
        
        Args:
            mode: 'fast', 'balanced', or 'accuracy'
        """
        if isinstance(mode, str):
            mode = DetectionMode(mode)
        
        if mode != self.config.mode:
            self.config.mode = mode
            self.model = self._load_model()
            self.model.eval()
    
    def get_model_info(self) -> Dict:
        """Get current model information."""
        return {
            'mode': self.config.mode.value,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.model.parameters()),
            'using_onnx': self.onnx_session is not None
        }
    
    def benchmark(self, num_iterations: int = 100) -> Dict:
        """
        Benchmark inference speed.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = self.model(dummy_input)
        
        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.model(dummy_input)
        elapsed = time.time() - start
        
        fps = num_iterations / elapsed
        
        return {
            'mode': self.config.mode.value,
            'device': str(self.device),
            'iterations': num_iterations,
            'total_time': elapsed,
            'fps': fps,
            'ms_per_frame': 1000 / fps
        }
    
    def _empty_result(self, message: str) -> FusionResult:
        """Create empty result with message."""
        return FusionResult(
            overall_fake_probability=0.0,
            video_score=0.0,
            audio_score=0.0,
            temporal_consistency_score=1.0,
            lip_sync_score=0.0,
            confidence=0.0,
            risk_level='low',
            explanation=message,
            suspicious_frames=[],
            frame_results=[],
            audio_result=None
        )


class RealTimeDetector:
    """
    Real-time deepfake detection for webcam feeds.
    Optimized for low latency.
    """
    
    def __init__(self, camera_id: int = 0,
                 mode: DetectionMode = DetectionMode.FAST):
        """
        Initialize real-time detector.
        
        Args:
            camera_id: Webcam device ID
            mode: Detection mode (FAST recommended for real-time)
        """
        self.camera_id = camera_id
        
        config = InferenceConfig(
            mode=mode,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            analyze_audio=False  # Audio not available in real-time
        )
        
        self.engine = InferenceEngine(config)
        self.webcam = WebcamProcessor(camera_id=camera_id)
        
        self._running = False
        self._last_result = None
    
    def start(self, callback: Optional[callable] = None):
        """
        Start real-time detection.
        
        Args:
            callback: Function called with each result (frame, result)
        """
        if not self.webcam.start():
            raise RuntimeError("Could not start webcam")
        
        self._running = True
        
        try:
            for frame in self.webcam.stream_frames():
                if not self._running:
                    break
                
                result = self.engine.analyze_webcam_frame(frame)
                self._last_result = result
                
                if callback:
                    callback(frame, result)
                    
        finally:
            self.stop()
    
    def stop(self):
        """Stop real-time detection."""
        self._running = False
        self.webcam.stop()
    
    def get_last_result(self) -> Optional[Dict]:
        """Get most recent detection result."""
        return self._last_result
