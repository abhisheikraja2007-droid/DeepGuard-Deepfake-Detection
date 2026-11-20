"""
FastAPI Backend Server for Deepfake Detection Platform
Provides REST API and WebSocket endpoints for video analysis.
"""

import os
import sys
import uuid
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.engine import InferenceEngine, InferenceConfig
from models.fusion import DetectionMode


# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Real-time deepfake detection with explainability",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for jobs
jobs: Dict[str, Dict] = {}
UPLOAD_DIR = Path(tempfile.gettempdir()) / "deepfake_detection"
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize inference engine (lazy load)
_engine: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """Get or initialize inference engine."""
    global _engine
    if _engine is None:
        config = InferenceConfig(
            mode=DetectionMode.BALANCED,
            device='cpu',
            analyze_audio=True
        )
        _engine = InferenceEngine(config)
    return _engine


# Request/Response Models
class AnalysisRequest(BaseModel):
    mode: str = "balanced"  # fast, balanced, accuracy
    analyze_audio: bool = True


class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    progress: float
    message: str
    created_at: str
    completed_at: Optional[str] = None


class DetectionResult(BaseModel):
    job_id: str
    fake_probability: float
    video_score: float
    audio_score: float
    risk_level: str
    confidence: float
    explanation: str
    suspicious_frames: List[int]
    timeline: Dict


# API Endpoints
@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Deepfake Detection API", "version": "1.0.0"}


@app.get("/api/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "engine_loaded": _engine is not None
    }


@app.post("/api/upload", response_model=AnalysisResponse)
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """Upload video file for analysis."""
    # Validate file type
    allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    ext = Path(file.filename).suffix.lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}{ext}"
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    
    # Create job entry
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "File uploaded, waiting for analysis",
        "file_path": str(file_path),
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "result": None
    }
    
    return AnalysisResponse(
        job_id=job_id,
        status="pending",
        message="Video uploaded successfully. Call /api/analyze to start."
    )


@app.post("/api/analyze/{job_id}", response_model=AnalysisResponse)
async def start_analysis(
    job_id: str,
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """Start analysis on uploaded video."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] == "processing":
        raise HTTPException(status_code=400, detail="Analysis already in progress")
    
    if job["status"] == "completed":
        return AnalysisResponse(
            job_id=job_id,
            status="completed",
            message="Analysis already completed"
        )
    
    # Set mode
    mode = DetectionMode(request.mode)
    engine = get_engine()
    engine.set_mode(mode)
    
    # Start background analysis
    job["status"] = "processing"
    job["message"] = "Analysis started"
    
    background_tasks.add_task(run_analysis, job_id, request.analyze_audio)
    
    return AnalysisResponse(
        job_id=job_id,
        status="processing",
        message="Analysis started"
    )


async def run_analysis(job_id: str, analyze_audio: bool):
    """Background task for video analysis."""
    job = jobs[job_id]
    
    try:
        engine = get_engine()
        
        def progress_callback(progress: float, message: str):
            job["progress"] = progress
            job["message"] = message
        
        # Run analysis
        result = engine.analyze_video(
            job["file_path"],
            progress_callback=progress_callback
        )
        
        # Store result
        job["result"] = {
            "fake_probability": result.overall_fake_probability,
            "video_score": result.video_score,
            "audio_score": result.audio_score,
            "risk_level": result.risk_level,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "suspicious_frames": result.suspicious_frames,
            "timeline": {
                "frame_scores": [r.fake_probability for r in result.frame_results],
                "total_frames": len(result.frame_results)
            }
        }
        
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["message"] = "Analysis completed"
        job["progress"] = 1.0
        
    except Exception as e:
        job["status"] = "failed"
        job["message"] = f"Analysis failed: {str(e)}"
    
    finally:
        # Cleanup uploaded file
        try:
            os.remove(job["file_path"])
        except:
            pass


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at")
    )


@app.get("/api/results/{job_id}")
async def get_results(job_id: str):
    """Get analysis results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Analysis not completed. Status: {job['status']}"
        )
    
    return job["result"]


@app.get("/api/modes")
async def get_detection_modes():
    """Get available detection modes."""
    return {
        "modes": [
            {"id": "fast", "name": "Fast Mode", "description": "60+ FPS, ~2M params"},
            {"id": "balanced", "name": "Balanced Mode", "description": "15-30 FPS, ~5M params"},
            {"id": "accuracy", "name": "Accuracy Mode", "description": "5-10 FPS, ~22M params"}
        ]
    }


# WebSocket for real-time webcam analysis
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket for real-time webcam analysis."""
    await websocket.accept()
    engine = get_engine()
    engine.set_mode(DetectionMode.FAST)  # Use fast mode for real-time
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            
            # Decode frame
            import numpy as np
            import cv2
            
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Analyze frame
                result = engine.analyze_webcam_frame(frame)
                
                # Send result
                await websocket.send_json({
                    "fake_probability": result["fake_probability"],
                    "face_detected": result["face_detected"],
                    "bbox": result["bbox"],
                    "risk_level": result["risk_level"]
                })
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=1001)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
