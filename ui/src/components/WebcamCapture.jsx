import { useState, useRef, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import Webcam from 'react-webcam';
import { Camera, CameraOff, AlertTriangle, CheckCircle } from 'lucide-react';

export default function WebcamCapture({ mode, onResult }) {
    const { t } = useTranslation();
    const webcamRef = useRef(null);
    const [isActive, setIsActive] = useState(false);
    const [currentResult, setCurrentResult] = useState(null);
    const wsRef = useRef(null);

    const startCamera = useCallback(() => {
        setIsActive(true);
        // Connect WebSocket
        wsRef.current = new WebSocket(`ws://${window.location.host}/ws/stream`);

        wsRef.current.onmessage = (event) => {
            const result = JSON.parse(event.data);
            setCurrentResult(result);
            onResult(result);
        };
    }, [onResult]);

    const stopCamera = useCallback(() => {
        setIsActive(false);
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        setCurrentResult(null);
    }, []);

    // Send frames to server
    useEffect(() => {
        if (!isActive || !webcamRef.current || !wsRef.current) return;

        const interval = setInterval(() => {
            if (wsRef.current?.readyState === WebSocket.OPEN) {
                const canvas = webcamRef.current.getCanvas();
                if (canvas) {
                    canvas.toBlob((blob) => {
                        if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
                            wsRef.current.send(blob);
                        }
                    }, 'image/jpeg', 0.8);
                }
            }
        }, 100); // 10 FPS

        return () => clearInterval(interval);
    }, [isActive]);

    const getBorderColor = () => {
        if (!currentResult) return 'border-dark-600';
        if (currentResult.risk_level === 'high' || currentResult.risk_level === 'critical')
            return 'border-danger-500 shadow-lg shadow-danger-500/30';
        if (currentResult.risk_level === 'medium')
            return 'border-yellow-500 shadow-lg shadow-yellow-500/30';
        return 'border-success-500 shadow-lg shadow-success-500/30';
    };

    return (
        <div className="space-y-4">
            <div className={`glass-card p-4 border-2 ${getBorderColor()} transition-all duration-300`}>
                <div className="relative aspect-video bg-dark-900 rounded-xl overflow-hidden">
                    {isActive ? (
                        <>
                            <Webcam
                                ref={webcamRef}
                                audio={false}
                                screenshotFormat="image/jpeg"
                                className="w-full h-full object-cover"
                            />

                            {/* Overlay with detection info */}
                            {currentResult?.face_detected && currentResult?.bbox && (
                                <div
                                    className="absolute border-2 border-primary-400 rounded-lg pointer-events-none"
                                    style={{
                                        left: `${(currentResult.bbox[0] / 224) * 100}%`,
                                        top: `${(currentResult.bbox[1] / 224) * 100}%`,
                                        width: `${((currentResult.bbox[2] - currentResult.bbox[0]) / 224) * 100}%`,
                                        height: `${((currentResult.bbox[3] - currentResult.bbox[1]) / 224) * 100}%`,
                                    }}
                                />
                            )}

                            {/* Status indicator */}
                            <div className="absolute top-4 right-4">
                                {currentResult && (
                                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full backdrop-blur-sm ${currentResult.risk_level === 'high' || currentResult.risk_level === 'critical'
                                            ? 'bg-danger-500/80 text-white'
                                            : currentResult.risk_level === 'medium'
                                                ? 'bg-yellow-500/80 text-dark-900'
                                                : 'bg-success-500/80 text-white'
                                        }`}>
                                        {(currentResult.risk_level === 'high' || currentResult.risk_level === 'critical') ? (
                                            <AlertTriangle className="w-4 h-4" />
                                        ) : (
                                            <CheckCircle className="w-4 h-4" />
                                        )}
                                        <span className="font-medium text-sm">
                                            {(currentResult.fake_probability * 100).toFixed(0)}%
                                        </span>
                                    </div>
                                )}
                            </div>

                            {/* Recording indicator */}
                            <div className="absolute top-4 left-4 flex items-center gap-2">
                                <div className="w-3 h-3 bg-danger-500 rounded-full animate-pulse" />
                                <span className="text-sm font-medium text-white drop-shadow-lg">LIVE</span>
                            </div>
                        </>
                    ) : (
                        <div className="flex flex-col items-center justify-center h-full text-dark-500">
                            <CameraOff className="w-16 h-16 mb-4" />
                            <p>{t('webcam.start')}</p>
                        </div>
                    )}
                </div>
            </div>

            <button
                onClick={isActive ? stopCamera : startCamera}
                className={`w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${isActive
                        ? 'bg-danger-600 hover:bg-danger-500 text-white'
                        : 'btn-primary'
                    }`}
            >
                {isActive ? (
                    <>
                        <CameraOff className="w-5 h-5" />
                        {t('webcam.stop')}
                    </>
                ) : (
                    <>
                        <Camera className="w-5 h-5" />
                        {t('webcam.start')}
                    </>
                )}
            </button>

            {/* Live Stats */}
            {isActive && currentResult && (
                <div className="glass-card p-4 grid grid-cols-2 gap-4">
                    <div className="text-center">
                        <p className="text-sm text-dark-400">Fake Probability</p>
                        <p className={`text-2xl font-bold ${currentResult.fake_probability > 0.7 ? 'text-danger-400' :
                                currentResult.fake_probability > 0.4 ? 'text-yellow-400' :
                                    'text-success-400'
                            }`}>
                            {(currentResult.fake_probability * 100).toFixed(1)}%
                        </p>
                    </div>
                    <div className="text-center">
                        <p className="text-sm text-dark-400">Face Detected</p>
                        <p className="text-2xl font-bold">
                            {currentResult.face_detected ? '✓' : '✗'}
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
}
