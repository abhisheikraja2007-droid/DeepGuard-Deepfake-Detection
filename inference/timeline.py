"""
Timeline Analysis Module
Analyzes video timeline for suspicious frames.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TimelineMarker:
    """Marker for suspicious segment."""
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    severity: str
    description: str
    avg_score: float
    peak_score: float
    marker_type: str


@dataclass  
class TimelineResult:
    """Timeline analysis result."""
    total_frames: int
    duration: float
    fps: float
    frame_scores: List[float]
    markers: List[TimelineMarker]
    suspicious_segments: List[Tuple[int, int]]
    overall_risk: str
    timeline_summary: str


class TimelineAnalyzer:
    """Analyze video timeline for deepfake detection."""
    
    def __init__(self, fps: float = 30.0, spike_threshold: float = 0.7,
                 sustained_threshold: float = 0.5, min_segment_frames: int = 3):
        self.fps = fps
        self.spike_threshold = spike_threshold
        self.sustained_threshold = sustained_threshold
        self.min_segment_frames = min_segment_frames
    
    def analyze(self, frame_scores: List[float]) -> TimelineResult:
        """Analyze frame scores to generate timeline."""
        if not frame_scores:
            return TimelineResult(0, 0, self.fps, [], [], [], 'unknown', "No frames")
        
        total_frames = len(frame_scores)
        duration = total_frames / self.fps
        segments = self._find_segments(frame_scores)
        markers = self._create_markers(segments, frame_scores)
        overall_risk = self._calculate_risk(frame_scores, markers)
        summary = self._generate_summary(markers, frame_scores, duration)
        
        return TimelineResult(total_frames, duration, self.fps, frame_scores,
                            markers, segments, overall_risk, summary)
    
    def _find_segments(self, scores: List[float]) -> List[Tuple[int, int]]:
        """Find continuous suspicious segments."""
        segments = []
        in_segment = False
        start = 0
        
        for i, score in enumerate(scores):
            if score >= self.sustained_threshold:
                if not in_segment:
                    start = i
                    in_segment = True
            elif in_segment:
                if i - start >= self.min_segment_frames:
                    segments.append((start, i - 1))
                in_segment = False
        
        if in_segment and len(scores) - start >= self.min_segment_frames:
            segments.append((start, len(scores) - 1))
        
        return segments
    
    def _create_markers(self, segments: List[Tuple[int, int]],
                       scores: List[float]) -> List[TimelineMarker]:
        """Create timeline markers."""
        markers = []
        for start, end in segments:
            seg_scores = scores[start:end+1]
            avg = float(np.mean(seg_scores))
            peak = float(np.max(seg_scores))
            severity = 'critical' if peak >= 0.9 else 'high' if peak >= 0.75 else 'medium'
            
            markers.append(TimelineMarker(
                start_frame=start, end_frame=end,
                start_time=start/self.fps, end_time=end/self.fps,
                severity=severity,
                description=f"Manipulation detected ({peak*100:.0f}%)",
                avg_score=avg, peak_score=peak, marker_type='sustained'
            ))
        return markers
    
    def _calculate_risk(self, scores: List[float], markers: List[TimelineMarker]) -> str:
        """Calculate overall risk level."""
        if not scores: return 'unknown'
        max_score = np.max(scores)
        if max_score >= 0.9: return 'critical'
        if max_score >= 0.7: return 'high'
        if max_score >= 0.5: return 'medium'
        return 'low' if markers else 'safe'
    
    def _generate_summary(self, markers: List[TimelineMarker],
                         scores: List[float], duration: float) -> str:
        """Generate summary."""
        if not markers:
            return "No suspicious segments detected."
        avg = np.mean(scores)
        return f"Found {len(markers)} suspicious segments. Avg score: {avg*100:.1f}%"
    
    def get_key_frames(self, scores: List[float], n: int = 5) -> List[int]:
        """Get most suspicious frame indices."""
        if not scores: return []
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)
        return sorted([i for i, _ in indexed[:n]])
    
    def generate_timeline_data(self, result: TimelineResult) -> Dict:
        """Generate JSON-serializable timeline data."""
        return {
            'totalFrames': result.total_frames,
            'duration': result.duration,
            'scores': result.frame_scores,
            'markers': [{'startFrame': m.start_frame, 'endFrame': m.end_frame,
                        'severity': m.severity, 'description': m.description}
                       for m in result.markers],
            'overallRisk': result.overall_risk
        }
