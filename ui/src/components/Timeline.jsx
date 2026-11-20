import { useMemo, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { AlertTriangle } from 'lucide-react';

export default function Timeline({ data, suspiciousFrames }) {
    const { t } = useTranslation();
    const [hoveredFrame, setHoveredFrame] = useState(null);

    const markers = useMemo(() => {
        if (!data?.markers) return [];
        return data.markers.map(m => ({
            ...m,
            position: (m.startFrame / (data.totalFrames || 1)) * 100
        }));
    }, [data]);

    if (!data?.scores?.length) {
        return (
            <div className="glass-card p-6 text-center text-dark-500">
                {t('timeline.noIssues')}
            </div>
        );
    }

    return (
        <div className="glass-card p-6">
            <h3 className="text-lg font-semibold mb-4">{t('timeline.title')}</h3>

            {/* Timeline visualization */}
            <div className="relative h-16 bg-dark-800 rounded-lg overflow-hidden mb-4">
                {/* Score graph */}
                <svg className="w-full h-full" preserveAspectRatio="none" viewBox={`0 0 ${data.scores.length} 100`}>
                    <defs>
                        <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#ef4444" stopOpacity="0.8" />
                            <stop offset="100%" stopColor="#ef4444" stopOpacity="0.1" />
                        </linearGradient>
                    </defs>

                    {/* Area chart */}
                    <path
                        d={`M 0 100 ${data.scores.map((s, i) => `L ${i} ${100 - s * 100}`).join(' ')} L ${data.scores.length} 100 Z`}
                        fill="url(#scoreGradient)"
                    />

                    {/* Line */}
                    <path
                        d={`M ${data.scores.map((s, i) => `${i} ${100 - s * 100}`).join(' L ')}`}
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="2"
                    />
                </svg>

                {/* Markers */}
                {markers.map((marker, idx) => (
                    <div
                        key={idx}
                        className={`timeline-marker ${marker.severity}`}
                        style={{ left: `${marker.position}%` }}
                        onMouseEnter={() => setHoveredFrame(marker)}
                        onMouseLeave={() => setHoveredFrame(null)}
                    />
                ))}

                {/* Hover tooltip */}
                {hoveredFrame && (
                    <div
                        className="absolute bottom-full mb-2 px-3 py-2 bg-dark-800 border border-dark-600 rounded-lg text-sm whitespace-nowrap z-10"
                        style={{ left: `${hoveredFrame.position}%`, transform: 'translateX(-50%)' }}
                    >
                        <p className="font-medium">{hoveredFrame.description}</p>
                        <p className="text-dark-400">Frame {hoveredFrame.startFrame}</p>
                    </div>
                )}
            </div>

            {/* Suspicious frames list */}
            {suspiciousFrames?.length > 0 && (
                <div>
                    <h4 className="text-sm font-medium mb-2 flex items-center gap-2">
                        <AlertTriangle className="w-4 h-4 text-danger-400" />
                        {t('timeline.suspicious')}
                    </h4>
                    <div className="flex flex-wrap gap-2">
                        {suspiciousFrames.slice(0, 10).map((frame) => (
                            <span
                                key={frame}
                                className="px-2 py-1 bg-danger-500/20 text-danger-400 rounded text-sm font-mono"
                            >
                                #{frame}
                            </span>
                        ))}
                        {suspiciousFrames.length > 10 && (
                            <span className="px-2 py-1 text-dark-500 text-sm">
                                +{suspiciousFrames.length - 10} more
                            </span>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
