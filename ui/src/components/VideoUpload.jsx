import { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Upload, FileVideo, X, Loader2 } from 'lucide-react';

export default function VideoUpload({ mode, onAnalysisStart, onAnalysisComplete, onProgress, isAnalyzing, progress }) {
    const { t } = useTranslation();
    const [dragActive, setDragActive] = useState(false);
    const [file, setFile] = useState(null);
    const [error, setError] = useState(null);

    const handleDrag = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(e.type === 'dragenter' || e.type === 'dragover');
    }, []);

    const handleDrop = useCallback((e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        const droppedFile = e.dataTransfer.files?.[0];
        if (droppedFile && isValidVideo(droppedFile)) {
            setFile(droppedFile);
            setError(null);
        } else {
            setError('Invalid file type. Please upload MP4, AVI, MOV, or MKV.');
        }
    }, []);

    const handleChange = useCallback((e) => {
        const selectedFile = e.target.files?.[0];
        if (selectedFile && isValidVideo(selectedFile)) {
            setFile(selectedFile);
            setError(null);
        }
    }, []);

    const isValidVideo = (file) => {
        const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/webm'];
        return validTypes.includes(file.type) || file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i);
    };

    const removeFile = () => {
        setFile(null);
        setError(null);
    };

    const startAnalysis = async () => {
        if (!file) return;

        onAnalysisStart();
        setError(null);

        try {
            // Upload file
            const formData = new FormData();
            formData.append('file', file);

            const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
            if (!uploadRes.ok) throw new Error('Upload failed');

            const { job_id } = await uploadRes.json();

            // Start analysis
            await fetch(`/api/analyze/${job_id}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode, analyze_audio: true })
            });

            // Poll for results
            let status = 'processing';
            while (status === 'processing' || status === 'pending') {
                await new Promise(r => setTimeout(r, 1000));

                const statusRes = await fetch(`/api/status/${job_id}`);
                const statusData = await statusRes.json();
                status = statusData.status;
                onProgress(statusData.progress * 100);

                if (status === 'failed') throw new Error(statusData.message);
            }

            // Get results
            const resultsRes = await fetch(`/api/results/${job_id}`);
            const results = await resultsRes.json();
            onAnalysisComplete(results);

        } catch (err) {
            setError(err.message || 'Analysis failed');
            onAnalysisComplete(null);
        }
    };

    return (
        <div className="space-y-4">
            {/* Drop Zone */}
            <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                className={`upload-zone flex flex-col items-center justify-center min-h-[200px] ${dragActive ? 'active' : ''
                    }`}
            >
                <input
                    type="file"
                    accept="video/*"
                    onChange={handleChange}
                    className="hidden"
                    id="video-upload"
                />

                {!file ? (
                    <>
                        <Upload className="w-12 h-12 text-dark-500 mb-4" />
                        <p className="text-lg font-medium mb-2">{t('upload.dragDrop')}</p>
                        <p className="text-dark-500 mb-4">{t('upload.or')}</p>
                        <label htmlFor="video-upload" className="btn-primary cursor-pointer">
                            {t('upload.browse')}
                        </label>
                        <p className="text-sm text-dark-500 mt-4">{t('upload.formats')}</p>
                    </>
                ) : (
                    <div className="flex items-center gap-4 px-4">
                        <FileVideo className="w-10 h-10 text-primary-400" />
                        <div className="flex-1">
                            <p className="font-medium truncate max-w-[300px]">{file.name}</p>
                            <p className="text-sm text-dark-500">{(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                        </div>
                        <button onClick={removeFile} className="p-2 hover:bg-dark-700 rounded-lg transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                )}
            </div>

            {/* Error Message */}
            {error && (
                <div className="p-4 bg-danger-500/10 border border-danger-500/30 rounded-xl text-danger-400">
                    {error}
                </div>
            )}

            {/* Progress Bar */}
            {isAnalyzing && (
                <div className="glass-card p-4">
                    <div className="flex items-center gap-3 mb-3">
                        <Loader2 className="w-5 h-5 animate-spin text-primary-400" />
                        <span>{t('analysis.processing')}... {progress.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-dark-700 rounded-full h-2">
                        <div
                            className="bg-gradient-to-r from-primary-500 to-cyan-400 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${progress}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Analyze Button */}
            {file && !isAnalyzing && (
                <button onClick={startAnalysis} className="btn-primary w-full flex items-center justify-center gap-2">
                    <Upload className="w-5 h-5" />
                    Analyze Video
                </button>
            )}
        </div>
    );
}
