# 🛡️ DeepGuard - Real-Time Deepfake Detection Platform

> AI-powered deepfake detection with video + audio analysis, explainable AI, and 100% offline functionality.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9+-green.svg)
![React](https://img.shields.io/badge/react-18.2-blue.svg)

## ✨ Features

- **Multi-Modal Analysis**: Video + Audio fusion detection
- **3 Detection Modes**: Fast (60+ FPS), Balanced (15-30 FPS), Accuracy (5-10 FPS)
- **Explainable AI**: GRAD-CAM heatmaps and human-readable explanations
- **Anti-Spoofing**: Replay attack, blink irregularity, texture analysis
- **Offline-First**: Zero cloud uploads, 100% local inference
- **Beautiful UI**: Dark mode, glassmorphism, Hindi + English
- **Real-Time**: Webcam live detection with WebSocket streaming

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- 8GB RAM minimum

### Installation

```bash
# Clone repository
cd "d:\projects\Deepfake detection"

# Install Python dependencies
pip install -r requirements.txt

# Install UI dependencies
cd ui
npm install
cd ..
```

### Running

```bash
# Terminal 1: Start backend
python scripts/server.py

# Terminal 2: Start frontend
cd ui
npm run dev
```

Open http://localhost:5173 in your browser.

## 📁 Project Structure

```
├── models/              # Neural network architectures
│   ├── face_detector.py    # MTCNN face detection
│   ├── fast_cnn.py         # Fast mode (~2M params)
│   ├── efficient_net.py    # Balanced mode (~5M params)
│   ├── xception.py         # Accuracy mode (~22M params)
│   ├── audio_analyzer.py   # Audio deepfake detection
│   └── fusion.py           # Multi-modal fusion
├── preprocessing/       # Data preprocessing
│   ├── augmentations.py    # Training augmentations
│   └── video_processor.py  # Video/webcam processing
├── inference/           # Detection pipeline
│   ├── engine.py           # Main inference engine
│   ├── explainability.py   # GRAD-CAM heatmaps
│   ├── timeline.py         # Temporal analysis
│   └── anti_spoof.py       # Anti-spoofing detection
├── scripts/             # CLI and server
│   ├── server.py           # FastAPI backend
│   └── cli.py              # Command-line tools
├── ui/                  # React + Tailwind frontend
│   └── src/
│       ├── components/     # UI components
│       └── i18n/           # Translations
└── training/            # Model training
```

## 🎮 CLI Usage

```bash
# Analyze a video
python scripts/cli.py analyze path/to/video.mp4 --mode balanced

# Run benchmarks
python scripts/cli.py benchmark --mode all

# Export to ONNX
python scripts/cli.py export-onnx --model all --verify
```

## 🔬 Detection Modes

| Mode | FPS (CPU) | Parameters | Use Case |
|------|-----------|------------|----------|
| Fast | 60+ | ~2M | Real-time webcam |
| Balanced | 15-30 | ~5M | General analysis |
| Accuracy | 5-10 | ~22M | Forensic analysis |

## 🧠 Supported Detections

- **Face Manipulations**: Face swaps, morphing, lip-sync
- **Audio Fakes**: Synthetic speech, cloned voices
- **Anti-Spoofing**: Replay attacks, printed photos, screens
- **Artifacts**: Blending boundaries, texture anomalies
- **Temporal**: Frame jitter, frozen frames, inconsistencies

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload video file |
| `/api/analyze/{id}` | POST | Start analysis |
| `/api/status/{id}` | GET | Check job status |
| `/api/results/{id}` | GET | Get results |
| `/ws/stream` | WebSocket | Real-time webcam |

## 🔐 Privacy & Ethics

- ✅ **100% Local**: No cloud uploads ever
- ✅ **Auto-Delete**: Temp files removed after analysis
- ✅ **Open Source**: Fully auditable code
- ✅ **Educational**: For safety and research purposes

## 📊 Datasets (Free)

- [FaceForensics++](https://github.com/ondyari/FaceForensics) (Academic)
- [DFDC Preview](https://ai.facebook.com/datasets/dfdc/)
- DeeperForensics
- Celeb-DF

## 🛠️ Technology Stack

- **Backend**: Python, PyTorch, FastAPI, ONNX Runtime
- **Frontend**: React 18, Tailwind CSS, Vite
- **Models**: Custom CNN, EfficientNet, Xception, ViT
- **Audio**: Librosa, Mel-spectrograms

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

**Built for hackathons and real-world impact** 🏆
