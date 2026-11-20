"""
Face Detection Module using MTCNN
Provides face detection, alignment, and tracking for deepfake analysis.
"""

import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
import cv2


class FaceDetector:
    """
    MTCNN-based face detector with tracking capabilities.
    Optimized for deepfake detection pipeline.
    """
    
    def __init__(self, device: str = 'cpu', min_face_size: int = 50, 
                 thresholds: List[float] = None, keep_all: bool = True):
        """
        Initialize Face Detector.
        
        Args:
            device: 'cpu' or 'cuda' for GPU acceleration
            min_face_size: Minimum face size to detect
            thresholds: Detection thresholds for MTCNN stages
            keep_all: If True, return all detected faces
        """
        self.device = torch.device(device)
        self.min_face_size = min_face_size
        self.thresholds = thresholds or [0.6, 0.7, 0.7]
        self.keep_all = keep_all
        
        # Lazy load MTCNN to avoid import issues
        self._mtcnn = None
        self._face_tracker = {}
        self._next_face_id = 0
        
    def _load_mtcnn(self):
        """Lazy load MTCNN model."""
        if self._mtcnn is None:
            try:
                from facenet_pytorch import MTCNN
                self._mtcnn = MTCNN(
                    image_size=224,
                    margin=40,
                    min_face_size=self.min_face_size,
                    thresholds=self.thresholds,
                    factor=0.709,
                    post_process=True,
                    device=self.device,
                    keep_all=self.keep_all
                )
            except ImportError:
                raise ImportError("facenet-pytorch is required. Install with: pip install facenet-pytorch")
        return self._mtcnn
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a single frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            List of face dictionaries with keys:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection confidence
                - landmarks: facial landmarks (5 points)
                - aligned_face: cropped and aligned face image
        """
        mtcnn = self._load_mtcnn()
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Detect faces
        boxes, probs, landmarks = mtcnn.detect(pil_image, landmarks=True)
        
        faces = []
        if boxes is not None:
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks if landmarks is not None else [None]*len(boxes))):
                if prob < self.thresholds[2]:
                    continue
                    
                x1, y1, x2, y2 = map(int, box)
                
                # Extract aligned face
                aligned = self._align_face(rgb_frame, box, landmark)
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(prob),
                    'landmarks': landmark.tolist() if landmark is not None else None,
                    'aligned_face': aligned,
                    'face_id': None  # Will be set during tracking
                })
        
        return faces
    
    def detect_and_track(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Detect and track faces across frames.
        
        Args:
            frame: BGR image as numpy array
            frame_idx: Current frame index for tracking
            
        Returns:
            List of face dictionaries with tracking IDs
        """
        faces = self.detect_faces(frame)
        
        # Simple IoU-based tracking
        for face in faces:
            best_match_id = None
            best_iou = 0.3  # Minimum IoU threshold
            
            for face_id, prev_info in self._face_tracker.items():
                if frame_idx - prev_info['last_seen'] > 30:  # Lost track after 30 frames
                    continue
                    
                iou = self._calculate_iou(face['bbox'], prev_info['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_match_id = face_id
            
            if best_match_id is not None:
                face['face_id'] = best_match_id
                self._face_tracker[best_match_id] = {
                    'bbox': face['bbox'],
                    'last_seen': frame_idx
                }
            else:
                face['face_id'] = self._next_face_id
                self._face_tracker[self._next_face_id] = {
                    'bbox': face['bbox'],
                    'last_seen': frame_idx
                }
                self._next_face_id += 1
        
        return faces
    
    def _align_face(self, rgb_frame: np.ndarray, box: np.ndarray, 
                    landmarks: Optional[np.ndarray] = None, 
                    target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Align and crop face from frame.
        
        Args:
            rgb_frame: RGB image
            box: Face bounding box [x1, y1, x2, y2]
            landmarks: Facial landmarks for alignment
            target_size: Output size
            
        Returns:
            Aligned face as numpy array
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Add margin
        h, w = rgb_frame.shape[:2]
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(w, x2 + margin_x)
        y2 = min(h, y2 + margin_y)
        
        # Crop face
        face = rgb_frame[y1:y2, x1:x2]
        
        # Resize to target size
        if face.size > 0:
            face = cv2.resize(face, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            face = np.zeros((*target_size, 3), dtype=np.uint8)
        
        return face
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reset_tracking(self):
        """Reset face tracking state."""
        self._face_tracker = {}
        self._next_face_id = 0
    
    def extract_face_sequence(self, frames: List[np.ndarray], 
                               sample_rate: int = 1) -> Dict[int, List[np.ndarray]]:
        """
        Extract face sequences from a list of frames.
        
        Args:
            frames: List of BGR video frames
            sample_rate: Process every Nth frame
            
        Returns:
            Dictionary mapping face_id to list of aligned face images
        """
        self.reset_tracking()
        face_sequences = {}
        
        for idx, frame in enumerate(frames):
            if idx % sample_rate != 0:
                continue
                
            faces = self.detect_and_track(frame, idx)
            
            for face in faces:
                face_id = face['face_id']
                if face_id not in face_sequences:
                    face_sequences[face_id] = []
                face_sequences[face_id].append(face['aligned_face'])
        
        return face_sequences
