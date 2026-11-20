import { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import { Shield, Globe, Upload, Camera, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import VideoUpload from './components/VideoUpload';
import WebcamCapture from './components/WebcamCapture';
import RiskMeter from './components/RiskMeter';
import Timeline from './components/Timeline';
import ExplainPanel from './components/ExplainPanel';
import ModeSelector from './components/ModeSelector';
import EthicsDisclaimer from './components/EthicsDisclaimer';

function App() {
    const { t, i18n } = useTranslation();
    const [darkMode, setDarkMode] = useState(true); // Default to true and keep it that way
    const [activeTab, setActiveTab] = useState('upload');
    const [selectedMode, setSelectedMode] = useState('balanced');
    const [analysisResult, setAnalysisResult] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [showEthics, setShowEthics] = useState(false);

    const toggleLanguage = () => {
        i18n.changeLanguage(i18n.language === 'en' ? 'hi' : 'en');
    };

    const handleAnalysisComplete = useCallback((result) => {
        setAnalysisResult(result);
        setIsAnalyzing(false);
        setProgress(100);
    }, []);

    const handleAnalysisStart = useCallback(() => {
        setIsAnalyzing(true);
        setAnalysisResult(null);
        setProgress(0);
    }, []);

    const handleProgress = useCallback((value) => {
        setProgress(value);
    }, []);

    const getRiskColor = (risk) => {
        const colors = {
            safe: 'text-success-400',
            low: 'text-success-400',
            medium: 'text-yellow-400',
            high: 'text-danger-400',
            critical: 'text-danger-500'
        };
        return colors[risk] || 'text-gray-400';
    };

    const getRiskIcon = (risk) => {
        if (risk === 'safe' || risk === 'low') return <CheckCircle className="w-6 h-6" />;
        if (risk === 'medium') return <Info className="w-6 h-6" />;
        return <AlertTriangle className="w-6 h-6" />;
    };

    // Force Dark Mode for Security Theme
    useEffect(() => {
        document.documentElement.classList.add('dark');
    }, []);

    return (
        <div className="min-h-screen bg-dark-950 text-gray-100 font-sans selection:bg-primary-500/30 selection:text-white">
            <div className="scanlines"></div>

            {/* Header */}
            <header className="border-b border-white/5 bg-dark-900/80 backdrop-blur-md sticky top-0 z-50">
                <div className="max-w-[1600px] mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="relative group">
                                <div className="absolute inset-0 bg-primary-500 blur-lg opacity-20 group-hover:opacity-40 transition-opacity"></div>
                                <Shield className="w-10 h-10 text-primary-400 relative z-10" />
                            </div>
                            <div>
                                <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                                    DEEPGUARD <span className="text-primary-500 text-xs px-2 py-0.5 bg-primary-500/10 border border-primary-500/30 rounded-sm">PRO v2.0</span>
                                </h1>
                                <p className="text-xs text-primary-400/80 font-mono tracking-widest uppercase">Deepfake Detection System</p>
                            </div>
                        </div>

                        <div className="flex items-center gap-4">
                            {/* System Status Indicators */}
                            <div className="hidden md:flex items-center gap-6 mr-6 border-r border-white/10 pr-6">
                                <div className="flex flex-col items-end">
                                    <span className="text-[10px] uppercase text-gray-500 tracking-wider font-bold">System Status</span>
                                    <span className="text-xs font-mono text-success-400 flex items-center gap-1.5">
                                        <span className="w-1.5 h-1.5 bg-success-500 rounded-full animate-pulse"></span>
                                        ONLINE
                                    </span>
                                </div>
                                <div className="flex flex-col items-end">
                                    <span className="text-[10px] uppercase text-gray-500 tracking-wider font-bold">Engine</span>
                                    <span className="text-xs font-mono text-primary-400">
                                        {selectedMode.toUpperCase()}
                                    </span>
                                </div>
                            </div>

                            {/* Language Toggle */}
                            <button
                                onClick={toggleLanguage}
                                className="flex items-center gap-2 px-3 py-2 bg-dark-800 border border-white/10 hover:border-primary-500/50 text-gray-400 hover:text-white transition-all text-xs uppercase tracking-wider"
                            >
                                <Globe className="w-3.5 h-3.5" />
                                {i18n.language === 'en' ? 'HI' : 'EN'}
                            </button>

                            {/* Ethics Button */}
                            <button
                                onClick={() => setShowEthics(true)}
                                className="px-4 py-2 bg-primary-500/10 hover:bg-primary-500/20 text-primary-400 border border-primary-500/30 hover:border-primary-500/50 transition-all text-xs font-bold uppercase tracking-wider flex items-center gap-2"
                            >
                                <Info className="w-3.5 h-3.5" />
                                Protocol
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <main className="max-w-[1600px] mx-auto px-6 py-8">
                {/* Mode Selector */}
                <div className="mb-8">
                    <ModeSelector
                        selectedMode={selectedMode}
                        onModeChange={setSelectedMode}
                    />
                </div>

                <div className="grid grid-cols-12 gap-8">
                    {/* Main Content Area */}
                    <div className="col-span-12 lg:col-span-8 space-y-6">
                        {/* Tab Navigation */}
                        <div className="flex gap-1 bg-dark-900/50 p-1 border border-white/5 w-fit">
                            <button
                                onClick={() => setActiveTab('upload')}
                                className={`flex items-center gap-2 px-6 py-2.5 text-sm font-bold uppercase tracking-wider transition-all ${activeTab === 'upload'
                                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-900/20'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                <Upload className="w-4 h-4" />
                                {t('upload.title')}
                            </button>
                            <button
                                onClick={() => setActiveTab('webcam')}
                                className={`flex items-center gap-2 px-6 py-2.5 text-sm font-bold uppercase tracking-wider transition-all ${activeTab === 'webcam'
                                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-900/20'
                                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                                    }`}
                            >
                                <Camera className="w-4 h-4" />
                                {t('webcam.title')}
                            </button>
                        </div>

                        {/* Input Area */}
                        <div className="relative">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-primary-500 to-cyan-500 opacity-20 blur opacity-0 transition duration-500 pointer-events-none"></div>
                            {activeTab === 'upload' ? (
                                <VideoUpload
                                    mode={selectedMode}
                                    onAnalysisStart={handleAnalysisStart}
                                    onAnalysisComplete={handleAnalysisComplete}
                                    onProgress={handleProgress}
                                    isAnalyzing={isAnalyzing}
                                    progress={progress}
                                />
                            ) : (
                                <WebcamCapture
                                    mode={selectedMode}
                                    onResult={handleAnalysisComplete}
                                />
                            )}
                        </div>

                        {/* Timeline */}
                        {analysisResult && analysisResult.timeline && (
                            <Timeline
                                data={analysisResult.timeline}
                                suspiciousFrames={analysisResult.suspicious_frames || []}
                            />
                        )}

                        {/* Explanation Panel */}
                        {analysisResult && (
                            <ExplainPanel result={analysisResult} />
                        )}
                    </div>

                    {/* Right Panel - Analysis Results */}
                    <div className="col-span-12 lg:col-span-4 space-y-6">
                        {/* Risk Assessment Card */}
                        <div className="glass-card p-6">
                            <div className="flex items-center justify-between mb-6">
                                <h3 className="text-sm font-bold text-white uppercase tracking-widest flex items-center gap-2">
                                    <div className="w-1 h-4 bg-primary-500"></div>
                                    Threat Assessment
                                </h3>
                                {analysisResult && (
                                    <span className={`px-3 py-1 text-xs font-mono font-bold border ${(analysisResult.risk_level === 'high' || analysisResult.risk_level === 'critical')
                                        ? 'border-danger-500/50 text-danger-400 bg-danger-500/10'
                                        : 'border-success-500/50 text-success-400 bg-success-500/10'
                                        }`}>
                                        {(analysisResult.risk_level || 'safe').toUpperCase()}
                                    </span>
                                )}
                            </div>

                            <div className="mb-8">
                                <RiskMeter
                                    value={analysisResult?.fake_probability || 0}
                                    riskLevel={analysisResult?.risk_level || 'safe'}
                                />
                            </div>

                            {analysisResult && (
                                <div className="grid grid-cols-2 gap-4 mt-6">
                                    <div className="p-4 bg-dark-800/50 border border-white/5 text-center group hover:border-primary-500/30 transition-colors">
                                        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Authenticity</div>
                                        <div className="text-2xl font-mono font-bold text-white">
                                            {((1 - (analysisResult.fake_probability || 0)) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                    <div className="p-4 bg-dark-800/50 border border-white/5 text-center group hover:border-danger-500/30 transition-colors">
                                        <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">Fake Prob</div>
                                        <div className={`text-2xl font-mono font-bold ${(analysisResult.fake_probability || 0) > 0.5 ? 'text-danger-400' : 'text-success-400'
                                            }`}>
                                            {((analysisResult.fake_probability || 0) * 100).toFixed(1)}%
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Detailed Metrics */}
                        {analysisResult && (
                            <div className="glass-card p-6">
                                <h3 className="text-sm font-bold text-white uppercase tracking-widest flex items-center gap-2 mb-6">
                                    <div className="w-1 h-4 bg-cyan-500"></div>
                                    Signal Analysis
                                </h3>

                                <div className="space-y-6">
                                    {/* Video Score */}
                                    <div className="space-y-2">
                                        <div className="flex justify-between items-end">
                                            <span className="text-xs text-gray-400 uppercase tracking-wider">Visual Artifacts</span>
                                            <span className="font-mono text-sm text-primary-400">
                                                {Math.max(0, (analysisResult.video_score || 0) * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="h-1.5 bg-dark-800 w-full overflow-hidden">
                                            <div
                                                className="h-full bg-primary-500 transition-all duration-1000 ease-out"
                                                style={{ width: `${Math.max(0, (analysisResult.video_score || 0) * 100)}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Audio Score */}
                                    <div className="space-y-2">
                                        <div className="flex justify-between items-end">
                                            <span className="text-xs text-gray-400 uppercase tracking-wider">Audio Anomalies</span>
                                            <span className="font-mono text-sm text-cyan-400">
                                                {Math.max(0, (analysisResult.audio_score || 0) * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="h-1.5 bg-dark-800 w-full overflow-hidden">
                                            <div
                                                className="h-full bg-cyan-500 transition-all duration-1000 ease-out"
                                                style={{ width: `${Math.max(0, (analysisResult.audio_score || 0) * 100)}%` }}
                                            />
                                        </div>
                                    </div>

                                    {/* Temporal Score */}
                                    <div className="space-y-2">
                                        <div className="flex justify-between items-end">
                                            <span className="text-xs text-gray-400 uppercase tracking-wider">Temporal Jitter</span>
                                            <span className="font-mono text-sm text-purple-400">
                                                {Math.max(0, (analysisResult.temporal_consistency_score || 0) * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <div className="h-1.5 bg-dark-800 w-full overflow-hidden">
                                            <div
                                                className="h-full bg-purple-500 transition-all duration-1000 ease-out"
                                                style={{ width: `${Math.max(0, (analysisResult.temporal_consistency_score || 0) * 100)}%` }}
                                            />
                                        </div>
                                    </div>
                                </div>

                                <div className="mt-8 pt-6 border-t border-white/5">
                                    <div className="flex items-center justify-between text-xs text-gray-500 font-mono">
                                        <span>CONFIDENCE SCORE</span>
                                        <span className="text-white">
                                            {Math.max(0, (analysisResult.confidence || 0) * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {!analysisResult && !isAnalyzing && (
                            <div className="glass-card p-8 text-center opacity-50 border-dashed border-white/10">
                                <div className="text-4xl mb-4">🛡️</div>
                                <h4 className="text-sm font-bold uppercase tracking-widest text-gray-400 mb-2">System Idle</h4>
                                <p className="text-xs text-gray-600">Upload media or start webcam to initialize detection protocols.</p>
                            </div>
                        )}
                    </div>
                </div>
            </main>

            <EthicsDisclaimer isOpen={showEthics} onClose={() => setShowEthics(false)} />
        </div>
    );
}

export default App;
