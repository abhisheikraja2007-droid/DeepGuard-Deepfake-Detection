"""
Anti-Spoofing Detection Module
Detects replay attacks, frozen frames, and presentation attacks.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class AntiSpoofResult:
    """Anti-spoofing detection result."""
    is_spoof: bool
    spoof_type: Optional[str]
    confidence: float
    details: Dict


class AntiSpoofDetector:
    """Detect various spoofing attacks."""
    
    def __init__(self, blink_threshold: float = 0.25,
                 texture_threshold: float = 0.4):
        self.blink_threshold = blink_threshold
        self.texture_threshold = texture_threshold
        self._blink_history = []
    
    def analyze(self, frames: List[np.ndarray],
                face_data: Optional[List[Dict]] = None) -> AntiSpoofResult:
        """Run all anti-spoofing checks."""
        checks = {
            'replay': self.detect_replay_attack(frames),
            'frozen': self.detect_frozen_frames(frames),
            'texture': self.detect_texture_anomaly(frames[0] if frames else None),
        }
        
        # Find highest confidence spoof
        max_score = 0
        spoof_type = None
        
        for check_type, result in checks.items():
            if result['score'] > max_score:
                max_score = result['score']
                spoof_type = check_type
        
        is_spoof = max_score > 0.6
        
        return AntiSpoofResult(
            is_spoof=is_spoof,
            spoof_type=spoof_type if is_spoof else None,
            confidence=max_score,
            details=checks
        )
    
    def detect_replay_attack(self, frames: List[np.ndarray]) -> Dict:
        """Detect screen replay attacks via moire patterns."""
        if not frames or len(frames) < 2:
            return {'score': 0, 'detail': 'Insufficient frames'}
        
        frame = frames[len(frames)//2]
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # FFT to detect periodic patterns (moire)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Check for periodic spikes indicating screen capture
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        
        # Mask out center (DC component)
        mask = np.ones_like(magnitude)
        mask[center_h-10:center_h+10, center_w-10:center_w+10] = 0
        
        masked_mag = magnitude * mask
        max_val = np.max(masked_mag)
        mean_val = np.mean(masked_mag)
        
        ratio = max_val / (mean_val + 1e-8)
        score = min(1, ratio / 100)
        
        return {
            'score': float(score),
            'detail': 'Moire pattern detected' if score > 0.5 else 'No replay indicators'
        }
    
    def detect_frozen_frames(self, frames: List[np.ndarray]) -> Dict:
        """Detect frozen/repeated frames."""
        if len(frames) < 3:
            return {'score': 0, 'detail': 'Insufficient frames'}
        
        frozen_count = 0
        
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i-1], frames[i])
            if np.mean(diff) < 1.0:  # Nearly identical
                frozen_count += 1
        
        score = frozen_count / (len(frames) - 1)
        
        return {
            'score': float(score),
            'frozen_frames': frozen_count,
            'detail': f'{frozen_count} frozen frames detected'
        }
    
    def detect_texture_anomaly(self, frame: Optional[np.ndarray]) -> Dict:
        """Detect unnatural skin texture (print/screen)."""
        if frame is None:
            return {'score': 0, 'detail': 'No frame'}
        
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # LBP-like texture analysis
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        texture = cv2.Laplacian(blur, cv2.CV_64F)
        texture_var = np.var(texture)
        
        # Very low or very high texture variance is suspicious
        if texture_var < 50:
            score = 0.7  # Too smooth (printed/screen)
            detail = 'Unnaturally smooth texture'
        elif texture_var > 2000:
            score = 0.5  # Too noisy
            detail = 'Excessive noise detected'
        else:
            score = 0.1
            detail = 'Normal texture'
        
        return {'score': float(score), 'texture_variance': float(texture_var), 'detail': detail}
    
    def detect_blink_irregularity(self, eye_aspect_ratios: List[float]) -> Dict:
        """Detect unnatural blinking patterns."""
        if len(eye_aspect_ratios) < 30:
            return {'score': 0, 'detail': 'Insufficient data'}
        
        ears = np.array(eye_aspect_ratios)
        
        # Detect blinks (EAR drops below threshold)
        blinks = []
        in_blink = False
        
        for i, ear in enumerate(ears):
            if ear < self.blink_threshold and not in_blink:
                blinks.append(i)
                in_blink = True
            elif ear >= self.blink_threshold:
                in_blink = False
        
        # Normal blink rate: 15-20 per minute
        duration_sec = len(ears) / 30  # Assume 30 fps
        blink_rate = len(blinks) / (duration_sec / 60) if duration_sec > 0 else 0
        
        # Check for abnormal blink rate
        if blink_rate < 5 or blink_rate > 40:
            score = 0.7
            detail = f'Abnormal blink rate: {blink_rate:.1f}/min'
        elif len(blinks) > 1:
            # Check blink regularity (too regular = fake)
            intervals = np.diff(blinks)
            interval_std = np.std(intervals) / (np.mean(intervals) + 1e-8)
            
            if interval_std < 0.2:  # Too regular
                score = 0.5
                detail = 'Unnaturally regular blinking'
            else:
                score = 0.1
                detail = 'Normal blinking pattern'
        else:
            score = 0.3
            detail = 'Insufficient blinks detected'
        
        return {'score': float(score), 'blink_rate': blink_rate, 'detail': detail}
    
    def detect_audio_video_desync(self, audio_features: np.ndarray,
                                  lip_movements: np.ndarray) -> Dict:
        """Detect audio-video synchronization issues."""
        if len(audio_features) < 10 or len(lip_movements) < 10:
            return {'score': 0, 'detail': 'Insufficient data'}
        
        # Align lengths
        min_len = min(len(audio_features), len(lip_movements))
        audio = audio_features[:min_len]
        lips = lip_movements[:min_len]
        
        # Cross-correlation
        correlation = np.corrcoef(audio, lips)[0, 1]
        if np.isnan(correlation):
            correlation = 0
        
        # Low correlation = desync
        score = max(0, 0.8 - correlation)
        detail = 'Audio-video mismatch' if score > 0.5 else 'Audio-video synced'
        
        return {'score': float(score), 'correlation': float(correlation), 'detail': detail}
