"""
Multi-Modal Fusion Engine
Combines video and audio analysis for robust deepfake detection.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DetectionMode(Enum):
    """Detection mode settings."""
    FAST = "fast"
    BALANCED = "balanced"
    ACCURACY = "accuracy"


@dataclass
class FrameResult:
    """Result for a single frame."""
    frame_idx: int
    fake_probability: float
    face_detected: bool
    confidence: float
    bbox: Optional[List[int]] = None
    attention_regions: Optional[np.ndarray] = None
    texture_anomaly: float = 0.0
    replay_score: float = 0.0


@dataclass
class AudioResult:
    """Result for audio analysis."""
    fake_probability: float
    lip_sync_score: float
    rhythm_anomaly: float
    artifacts: List[str]
    confidence: float


@dataclass
class FusionResult:
    """Combined analysis result."""
    overall_fake_probability: float
    video_score: float
    audio_score: float
    temporal_consistency_score: float
    lip_sync_score: float
    confidence: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    explanation: str
    suspicious_frames: List[int]
    frame_results: List[FrameResult]
    audio_result: Optional[AudioResult]
    mode: str = "balanced"


class TemporalConsistencyAnalyzer:
    """
    Analyze temporal consistency across video frames.
    Detects frame-to-frame anomalies that indicate manipulation.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize analyzer.
        
        Args:
            window_size: Number of frames to analyze together
        """
        self.window_size = window_size
        
    def analyze(self, frame_scores: List[float], 
                frame_features: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Analyze temporal consistency of detection scores.
        
        Args:
            frame_scores: Fake probability per frame
            frame_features: Optional feature vectors per frame
            
        Returns:
            Temporal analysis results
        """
        if len(frame_scores) < 2:
            return {
                'consistency_score': 1.0,
                'jitter_score': 0.0,
                'anomaly_frames': [],
                'variance': 0.0
            }
        
        scores = np.array(frame_scores)
        
        # Calculate score variance (high variance = suspicious)
        variance = np.var(scores)
        
        # Calculate jitter (sudden changes)
        diffs = np.abs(np.diff(scores))
        jitter = np.mean(diffs)
        
        # Identify anomaly frames (sudden spikes)
        threshold = np.mean(diffs) + 2 * np.std(diffs)
        anomaly_indices = np.where(diffs > threshold)[0].tolist()
        
        # Consistency score (0 = inconsistent, 1 = consistent)
        consistency = 1 - min(1, variance * 10)
        
        # Jitter score (0 = smooth, 1 = jerky - suspicious)
        jitter_score = min(1, jitter * 5)
        
        return {
            'consistency_score': float(consistency),
            'jitter_score': float(jitter_score),
            'variance': float(variance),
            'anomaly_frames': anomaly_indices
        }
    
    def detect_frozen_frames(self, frame_features: List[np.ndarray],
                             threshold: float = 0.99) -> List[int]:
        """
        Detect frozen/repeated frames (replay attack indicator).
        
        Args:
            frame_features: Feature vectors per frame
            threshold: Similarity threshold for identical frames
            
        Returns:
            Indices of frozen frame sequences
        """
        if len(frame_features) < 2:
            return []
        
        frozen_indices = []
        
        for i in range(1, len(frame_features)):
            similarity = self._cosine_similarity(
                frame_features[i-1].flatten(),
                frame_features[i].flatten()
            )
            if similarity > threshold:
                frozen_indices.append(i)
        
        return frozen_indices
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)


class FusionEngine:
    """
    Multi-modal fusion engine combining video and audio analysis.
    """
    
    def __init__(self, video_weight: float = 0.6, audio_weight: float = 0.4,
                 temporal_weight: float = 0.2, lip_sync_weight: float = 0.2):
        """
        Initialize fusion engine.
        
        Args:
            video_weight: Weight for video analysis
            audio_weight: Weight for audio analysis
            temporal_weight: Weight for temporal consistency
            lip_sync_weight: Weight for lip sync analysis
        """
        self.video_weight = video_weight
        self.audio_weight = audio_weight
        self.temporal_weight = temporal_weight
        self.lip_sync_weight = lip_sync_weight
        self.temporal_analyzer = TemporalConsistencyAnalyzer()
        
    def fuse(self, frame_results: List[FrameResult],
             audio_result: Optional[AudioResult] = None,
             lip_sync_score: float = 0.0,
             mode: DetectionMode = DetectionMode.BALANCED) -> FusionResult:
        """
        Fuse multi-modal analysis results.
        
        Args:
            frame_results: Per-frame video analysis results
            audio_result: Audio analysis results
            lip_sync_score: Lip-audio synchronization score
            mode: Detection mode (affects weighting)
            
        Returns:
            Combined fusion result
        """
        if not frame_results:
            return self._empty_result()
        
        # Calculate base video score from CNN model
        frame_scores = [r.fake_probability for r in frame_results]
        cnn_video_score = self._aggregate_frame_scores(frame_scores)
        
        # Calculate Heuristic Scores (Anti-Spoofing & Artifacts)
        texture_scores = [r.texture_anomaly for r in frame_results]
        replay_scores = [r.replay_score for r in frame_results]
        
        avg_texture = np.mean(texture_scores) if texture_scores else 0.0
        avg_replay = np.mean(replay_scores) if replay_scores else 0.0
        
        # Heuristic combined score (0 to 1)
        heuristic_score = max(avg_texture, avg_replay)
        
        # Mode-Specific Fusion Logic
        if mode == DetectionMode.FAST:
            # Fast mode: Trust heuristics heavily if CNN is uncertain (~0.5)
            if 0.4 < cnn_video_score < 0.6:
                video_score = (cnn_video_score * 0.3) + (heuristic_score * 0.7)
            else:
                video_score = (cnn_video_score * 0.7) + (heuristic_score * 0.3)
                
        elif mode == DetectionMode.BALANCED:
            # Balanced: Weighted average
            video_score = (cnn_video_score * 0.5) + (heuristic_score * 0.5)
            
        else: # ACCURACY
            # Accuracy: Penalize ANY anomaly heavily (Max-pooling logic)
            video_score = max(cnn_video_score, heuristic_score)
            
            # Boost score if multiple independent indicators are present
            if cnn_video_score > 0.5 and heuristic_score > 0.5:
                video_score = min(1.0, video_score + 0.1)

        # Temporal consistency
        temporal = self.temporal_analyzer.analyze(frame_scores)
        temporal_score = temporal['jitter_score'] * 0.5 + (1 - temporal['consistency_score']) * 0.5
        
        # Audio score
        audio_score = audio_result.fake_probability if audio_result else 0.0
        has_audio = audio_result is not None
        
        # Calculate weighted fusion
        if has_audio:
            # Full multi-modal fusion
            base_score = (
                self.video_weight * video_score +
                self.audio_weight * audio_score
            )
            
            # Add temporal and lip-sync penalties
            fusion_score = base_score + (
                self.temporal_weight * temporal_score +
                self.lip_sync_weight * lip_sync_score
            )
        else:
            # Video-only fusion
            fusion_score = video_score + self.temporal_weight * temporal_score
        
        # Normalize to [0, 1]
        overall_fake_prob = float(np.clip(fusion_score, 0, 1))
        
        # Disagreement bonus: if audio and video strongly disagree, increase suspicion
        if has_audio:
            disagreement = abs(video_score - audio_score)
            if disagreement > 0.4 and mode == DetectionMode.ACCURACY:
                overall_fake_prob = min(1.0, overall_fake_prob + 0.1)

        # Calculate confidence
        confidence = self._calculate_confidence(
            frame_results, audio_result, temporal
        )
        
        # Get suspicious frames
        suspicious_threshold = 0.55 if mode == DetectionMode.ACCURACY else 0.65
        suspicious_frames = [
            r.frame_idx for r in frame_results 
            if r.fake_probability > suspicious_threshold or r.texture_anomaly > 0.6
        ]
        
        suspicious_frames.extend(temporal['anomaly_frames'])
        suspicious_frames = sorted(set(suspicious_frames))
        
        # Risk Level
        risk_level = self._determine_risk_level(overall_fake_prob)
        
        # Generate explanation
        explanation = self._generate_explanation(
            overall_fake_prob, video_score, audio_score,
            temporal, lip_sync_score, has_audio,
            heuristic_score, avg_texture, avg_replay
        )
        
        return FusionResult(
            overall_fake_probability=overall_fake_prob,
            video_score=float(video_score),
            audio_score=float(audio_score),
            temporal_consistency_score=float(temporal['consistency_score']),
            lip_sync_score=float(lip_sync_score),
            confidence=float(confidence),
            risk_level=risk_level,
            explanation=explanation,
            suspicious_frames=suspicious_frames,
            frame_results=frame_results,
            audio_result=audio_result,
            mode=mode.value
        )
    
    def _aggregate_frame_scores(self, scores: List[float]) -> float:
        """
        Aggregate frame-level scores into a single video score.
        Uses weighted combination of mean, max, and percentile.
        """
        if not scores:
            return 0.0
        
        scores_arr = np.array(scores)
        
        mean_score = np.mean(scores_arr)
        max_score = np.max(scores_arr)
        p90_score = np.percentile(scores_arr, 90)
        
        # Weighted combination favoring high values
        return 0.4 * mean_score + 0.3 * max_score + 0.3 * p90_score
    
    def _calculate_confidence(self, frame_results: List[FrameResult],
                               audio_result: Optional[AudioResult],
                               temporal: Dict) -> float:
        """Calculate overall confidence in the analysis."""
        # Face detection rate
        faces_detected = sum(1 for r in frame_results if r.face_detected)
        detection_rate = faces_detected / len(frame_results) if frame_results else 0
        
        # Frame coverage
        frame_confidence = np.mean([r.confidence for r in frame_results])
        
        # Audio confidence
        audio_confidence = audio_result.confidence if audio_result else 0.5
        
        # Temporal stability
        temporal_confidence = temporal['consistency_score']
        
        # Weighted confidence
        confidence = (
            0.4 * detection_rate +
            0.3 * frame_confidence +
            0.2 * audio_confidence +
            0.1 * temporal_confidence
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def _determine_risk_level(self, fake_prob: float) -> str:
        """Determine risk level from fake probability."""
        if fake_prob < 0.3:
            return 'low'
        elif fake_prob < 0.5:
            return 'medium'
        elif fake_prob < 0.75:
            return 'high'
        else:
            return 'critical'
    
    def _generate_explanation(self, overall: float, video: float, 
                               audio: float, temporal: Dict,
                               lip_sync: float, has_audio: bool,
                               heuristic_score: float = 0.0,
                               texture_score: float = 0.0,
                               replay_score: float = 0.0) -> str:
        """Generate human-readable explanation of results."""
        explanations = []
        
        # Overall assessment
        if overall < 0.3:
            explanations.append("This content appears to be authentic/safe.")
        elif overall < 0.5:
            explanations.append("Some minor anomalies detected, but likely authentic.")
        elif overall < 0.75:
            explanations.append("Significant manipulation indicators detected.")
        else:
            explanations.append("Strong evidence of deepfake/manipulation detected.")
        
        # Video-specific
        if video > 0.6:
            explanations.append(f"Visual analysis indicates {video*100:.0f}% likelihood of face manipulation.")
        
        # Heuristics
        if texture_score > 0.6:
            explanations.append("Skin texture analysis reveals unnatural smoothness or noise patterns typical of digital generation.")
        if replay_score > 0.6:
            explanations.append("Potential replay attack or screen capture artifacts detected.")
            
        # Audio-specific
        if has_audio and audio > 0.5:
            explanations.append(f"Audio shows synthetic characteristics ({audio*100:.0f}% synthetic likelihood).")
        
        # Lip sync
        if lip_sync > 0.5:
            explanations.append("Lip movements appear out of sync with audio track.")
        
        # Temporal
        if temporal['jitter_score'] > 0.4:
            explanations.append("Temporal inconsistencies detected between frames (jitter).")
        
        if temporal['anomaly_frames']:
            explanations.append(f"Sudden quality changes at frames: {temporal['anomaly_frames'][:5]}")
        
        return " ".join(explanations)
    
    def _empty_result(self) -> FusionResult:
        """Return empty result when no frames available."""
        return FusionResult(
            overall_fake_probability=0.0,
            video_score=0.0,
            audio_score=0.0,
            temporal_consistency_score=1.0,
            lip_sync_score=0.0,
            confidence=0.0,
            risk_level='low',
            explanation="No video frames analyzed.",
            suspicious_frames=[],
            frame_results=[],
            audio_result=None
        )


class AdaptiveFusion(nn.Module):
    """
    Neural network-based adaptive fusion.
    Learns optimal weights for combining modalities.
    """
    
    def __init__(self, video_feat_dim: int = 256, 
                 audio_feat_dim: int = 128,
                 hidden_dim: int = 128):
        super().__init__()
        
        # Modality encoders
        self.video_encoder = nn.Sequential(
            nn.Linear(video_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            hidden_dim, num_heads=4, batch_first=True
        )
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
        
    def forward(self, video_features: torch.Tensor,
                audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-modal attention.
        
        Args:
            video_features: (B, video_feat_dim)
            audio_features: (B, audio_feat_dim)
            
        Returns:
            Logits (B, 2)
        """
        # Encode modalities
        video_encoded = self.video_encoder(video_features)  # (B, hidden)
        audio_encoded = self.audio_encoder(audio_features)  # (B, hidden)
        
        # Add sequence dimension for attention
        video_seq = video_encoded.unsqueeze(1)  # (B, 1, hidden)
        audio_seq = audio_encoded.unsqueeze(1)  # (B, 1, hidden)
        
        # Cross-modal attention
        attended, _ = self.cross_attention(video_seq, audio_seq, audio_seq)
        attended = attended.squeeze(1)  # (B, hidden)
        
        # Concatenate and classify
        fused = torch.cat([video_encoded, attended], dim=1)
        logits = self.classifier(fused)
        
        return logits
