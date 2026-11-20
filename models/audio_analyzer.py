"""
Audio Analysis Module for Deepfake Detection
Analyzes audio for deepfake artifacts and lip-sync issues.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import warnings


class MelSpectrogramExtractor:
    """Extract Mel spectrograms from audio for analysis."""
    
    def __init__(self, sample_rate: int = 16000, n_fft: int = 1024,
                 hop_length: int = 256, n_mels: int = 80,
                 f_min: float = 0.0, f_max: float = 8000.0):
        """
        Initialize Mel spectrogram extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length between frames
            n_mels: Number of mel bands
            f_min: Minimum frequency
            f_max: Maximum frequency
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        
        self._librosa = None
        
    def _load_librosa(self):
        """Lazy load librosa."""
        if self._librosa is None:
            try:
                import librosa
                self._librosa = librosa
            except ImportError:
                raise ImportError("librosa is required. Install with: pip install librosa")
        return self._librosa
    
    def extract(self, audio: np.ndarray, sr: Optional[int] = None) -> np.ndarray:
        """
        Extract Mel spectrogram from audio.
        
        Args:
            audio: Audio waveform as numpy array
            sr: Sample rate (if different from default)
            
        Returns:
            Mel spectrogram (n_mels, time_frames)
        """
        librosa = self._load_librosa()
        
        if sr is not None and sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_from_file(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract Mel spectrogram from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (mel_spectrogram, sample_rate)
        """
        librosa = self._load_librosa()
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        mel_spec = self.extract(audio, sr)
        return mel_spec, sr


class AudioCNN(nn.Module):
    """CNN for audio deepfake detection using Mel spectrograms."""
    
    def __init__(self, n_mels: int = 80, num_classes: int = 2):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram tensor (B, 1, n_mels, time)
            
        Returns:
            Logits (B, num_classes)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class LipSyncAnalyzer:
    """
    Analyze lip movement synchronization with audio.
    Detects mismatches between visual lip movements and audio speech.
    """
    
    def __init__(self, window_size: int = 5, threshold: float = 0.5):
        """
        Initialize Lip Sync Analyzer.
        
        Args:
            window_size: Number of frames to analyze together
            threshold: Mismatch threshold
        """
        self.window_size = window_size
        self.threshold = threshold
        self._librosa = None
        
    def _load_librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def extract_audio_features(self, audio: np.ndarray, 
                                sr: int = 16000) -> np.ndarray:
        """
        Extract audio features for lip sync analysis.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Audio features (energy envelope, spectral features)
        """
        librosa = self._load_librosa()
        
        # Energy envelope
        hop_length = int(sr / 30)  # Match video frame rate
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        
        # Spectral centroid (voice frequency indicator)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=sr, hop_length=hop_length
        )[0]
        
        # Voice activity detection
        voice_activity = rms > np.percentile(rms, 20)
        
        return np.stack([
            rms / (rms.max() + 1e-8),
            spectral_centroid / (spectral_centroid.max() + 1e-8),
            voice_activity.astype(float)
        ], axis=0)
    
    def extract_lip_features(self, face_landmarks: List[np.ndarray]) -> np.ndarray:
        """
        Extract lip movement features from facial landmarks.
        
        Args:
            face_landmarks: List of landmark arrays per frame
            
        Returns:
            Lip movement features
        """
        lip_features = []
        
        for landmarks in face_landmarks:
            if landmarks is None or len(landmarks) < 68:
                lip_features.append([0, 0, 0])
                continue
            
            # Lip landmarks (48-67 in 68-point model)
            upper_lip = landmarks[48:55]
            lower_lip = landmarks[55:60]
            
            # Lip opening (vertical distance)
            lip_opening = np.mean(lower_lip[:, 1]) - np.mean(upper_lip[:, 1])
            
            # Lip width
            lip_width = np.max(landmarks[48:55, 0]) - np.min(landmarks[48:55, 0])
            
            # Lip shape ratio
            lip_ratio = lip_opening / (lip_width + 1e-8)
            
            lip_features.append([lip_opening, lip_width, lip_ratio])
        
        return np.array(lip_features)
    
    def analyze_sync(self, audio_features: np.ndarray,
                     lip_features: np.ndarray) -> Dict[str, float]:
        """
        Analyze synchronization between audio and lip movements.
        
        Args:
            audio_features: Audio features (3, num_frames)
            lip_features: Lip features (num_frames, 3)
            
        Returns:
            Sync analysis results
        """
        # Align lengths
        min_len = min(audio_features.shape[1], len(lip_features))
        audio_features = audio_features[:, :min_len]
        lip_features = lip_features[:min_len]
        
        # Calculate correlation between voice activity and lip opening
        voice_activity = audio_features[2]
        lip_opening = lip_features[:, 0]
        
        # Normalize
        lip_opening = (lip_opening - lip_opening.mean()) / (lip_opening.std() + 1e-8)
        
        # Cross-correlation
        correlation = np.corrcoef(voice_activity, lip_opening)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # Check for temporal alignment
        max_lag = 5  # frames
        best_corr = correlation
        best_lag = 0
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            shifted_lip = np.roll(lip_opening, lag)
            corr = np.corrcoef(voice_activity, shifted_lip)[0, 1]
            if not np.isnan(corr) and corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        # Mismatch score (higher = more suspicious)
        if best_corr < 0.3:
            mismatch_score = 0.8
        elif best_corr < 0.5:
            mismatch_score = 0.5
        elif abs(best_lag) > 2:
            mismatch_score = 0.4
        else:
            mismatch_score = max(0, 0.5 - best_corr)
        
        return {
            'correlation': float(best_corr),
            'optimal_lag': best_lag,
            'mismatch_score': mismatch_score,
            'is_synced': mismatch_score < self.threshold
        }


class SpeechRhythmAnalyzer:
    """Analyze speech rhythm for anomalies in synthesized speech."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._librosa = None
        
    def _load_librosa(self):
        if self._librosa is None:
            import librosa
            self._librosa = librosa
        return self._librosa
    
    def analyze(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze speech rhythm for synthetic characteristics.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Rhythm analysis results
        """
        librosa = self._load_librosa()
        
        # Onset detection
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sample_rate
        )
        
        if len(onsets) < 2:
            return {
                'rhythm_regularity': 0.5,
                'tempo_stability': 0.5,
                'synthetic_score': 0.5
            }
        
        # Inter-onset intervals
        onset_times = librosa.frames_to_time(onsets, sr=self.sample_rate)
        intervals = np.diff(onset_times)
        
        # Rhythm regularity (synthetic speech often has unnaturally regular rhythm)
        rhythm_regularity = 1 - (np.std(intervals) / (np.mean(intervals) + 1e-8))
        rhythm_regularity = np.clip(rhythm_regularity, 0, 1)
        
        # Tempo estimation stability
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sample_rate)
        tempo_stability = 1 if len(tempo) == 1 else 0.5
        
        # Synthetic speech score (high regularity is suspicious)
        synthetic_score = 0.0
        if rhythm_regularity > 0.8:  # Too regular
            synthetic_score += 0.3
        if rhythm_regularity < 0.2:  # Too irregular (glitchy)
            synthetic_score += 0.2
        
        return {
            'rhythm_regularity': float(rhythm_regularity),
            'tempo_stability': float(tempo_stability),
            'synthetic_score': float(synthetic_score)
        }


class AudioAnalyzer:
    """
    Main audio analyzer combining all audio analysis components.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize Audio Analyzer.
        
        Args:
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        self.mel_extractor = MelSpectrogramExtractor()
        self.audio_cnn = AudioCNN().to(self.device)
        self.audio_cnn.eval()
        self.lip_sync_analyzer = LipSyncAnalyzer()
        self.rhythm_analyzer = SpeechRhythmAnalyzer()
        
    def analyze_audio(self, audio: np.ndarray, 
                      sr: int = 16000) -> Dict[str, any]:
        """
        Comprehensive audio analysis.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            
        Returns:
            Analysis results dictionary
        """
        results = {
            'audio_fake_probability': 0.5,
            'rhythm_analysis': None,
            'artifacts_detected': [],
            'confidence': 0.5
        }
        
        if audio is None or len(audio) == 0:
            return results
        
        try:
            # Extract Mel spectrogram
            mel_spec = self.mel_extractor.extract(audio, sr)
            
            # CNN prediction
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
            mel_tensor = mel_tensor.to(self.device)
            
            with torch.no_grad():
                proba = self.audio_cnn.predict_proba(mel_tensor)
                fake_prob = proba[0, 1].item()
            
            results['audio_fake_probability'] = fake_prob
            
            # Rhythm analysis
            rhythm = self.rhythm_analyzer.analyze(audio)
            results['rhythm_analysis'] = rhythm
            
            # Detect artifacts
            artifacts = self._detect_audio_artifacts(mel_spec, audio)
            results['artifacts_detected'] = artifacts
            
            # Overall confidence
            results['confidence'] = min(0.95, fake_prob + rhythm['synthetic_score'] * 0.2)
            
        except Exception as e:
            warnings.warn(f"Audio analysis error: {e}")
            
        return results
    
    def _detect_audio_artifacts(self, mel_spec: np.ndarray,
                                 audio: np.ndarray) -> List[str]:
        """Detect common audio deepfake artifacts."""
        artifacts = []
        
        # Check for spectral discontinuities
        spec_diff = np.abs(np.diff(mel_spec, axis=1))
        if np.max(spec_diff) > 50:  # Threshold for sudden changes
            artifacts.append('spectral_discontinuity')
        
        # Check for silence gaps (common in concatenative synthesis)
        silence_threshold = np.percentile(np.abs(audio), 5)
        silence_regions = np.abs(audio) < silence_threshold
        silence_runs = self._count_runs(silence_regions)
        if len(silence_runs) > 10 and np.mean(silence_runs) < 100:
            artifacts.append('unusual_silence_pattern')
        
        # Check for frequency cutoff (common in compressed TTS)
        high_freq_energy = np.mean(mel_spec[-10:])
        low_freq_energy = np.mean(mel_spec[:10])
        if high_freq_energy < low_freq_energy * 0.1:
            artifacts.append('high_frequency_cutoff')
        
        return artifacts
    
    def _count_runs(self, binary_array: np.ndarray) -> List[int]:
        """Count consecutive runs of True values."""
        runs = []
        count = 0
        for val in binary_array:
            if val:
                count += 1
            elif count > 0:
                runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        return runs
    
    def analyze_lip_sync(self, audio: np.ndarray, sr: int,
                         face_landmarks: List[np.ndarray]) -> Dict[str, any]:
        """
        Analyze lip sync between audio and video.
        
        Args:
            audio: Audio waveform
            sr: Sample rate
            face_landmarks: Facial landmarks per frame
            
        Returns:
            Lip sync analysis results
        """
        audio_features = self.lip_sync_analyzer.extract_audio_features(audio, sr)
        lip_features = self.lip_sync_analyzer.extract_lip_features(face_landmarks)
        sync_results = self.lip_sync_analyzer.analyze_sync(audio_features, lip_features)
        
        return sync_results
    
    def load_weights(self, checkpoint_path: str):
        """Load pretrained weights for audio CNN."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.audio_cnn.load_state_dict(checkpoint['model_state_dict'])
        self.audio_cnn.eval()


def extract_audio_from_video(video_path: str, 
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
        
        # Try to load audio directly from video
        audio, sr = librosa.load(video_path, sr=16000, mono=True)
        
        if output_path:
            import soundfile as sf
            sf.write(output_path, audio, sr)
        
        return audio, sr
        
    except Exception as e:
        warnings.warn(f"Could not extract audio: {e}")
        return np.array([]), 16000
