import { useMemo } from 'react';

export default function RiskMeter({ value, riskLevel }) {
    const percentage = Math.min(100, Math.max(0, value * 100));

    const needlePosition = useMemo(() => {
        return `${percentage}%`;
    }, [percentage]);

    const getGlowColor = () => {
        if (riskLevel === 'critical') return 'rgba(239, 68, 68, 0.5)';
        if (riskLevel === 'high') return 'rgba(239, 68, 68, 0.3)';
        if (riskLevel === 'medium') return 'rgba(234, 179, 8, 0.3)';
        return 'rgba(34, 197, 94, 0.3)';
    };

    return (
        <div className="space-y-4">
            {/* Circular Gauge */}
            <div className="relative w-48 h-48 mx-auto">
                <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                    {/* Background arc */}
                    <circle
                        cx="50"
                        cy="50"
                        r="42"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="12"
                        className="text-dark-700"
                        strokeDasharray="198 66"
                        strokeLinecap="round"
                    />

                    {/* Gradient arc */}
                    <defs>
                        <linearGradient id="riskGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                            <stop offset="0%" stopColor="#22c55e" />
                            <stop offset="50%" stopColor="#eab308" />
                            <stop offset="100%" stopColor="#ef4444" />
                        </linearGradient>
                    </defs>

                    <circle
                        cx="50"
                        cy="50"
                        r="42"
                        fill="none"
                        stroke="url(#riskGradient)"
                        strokeWidth="12"
                        strokeDasharray={`${(percentage / 100) * 198} 264`}
                        strokeLinecap="round"
                        className="transition-all duration-700 ease-out"
                        style={{
                            filter: `drop-shadow(0 0 10px ${getGlowColor()})`
                        }}
                    />
                </svg>

                {/* Center display */}
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className={`text-4xl font-bold ${percentage > 70 ? 'text-danger-400' :
                            percentage > 40 ? 'text-yellow-400' :
                                'text-success-400'
                        }`}>
                        {percentage.toFixed(0)}%
                    </span>
                    <span className="text-dark-500 text-sm">Risk Score</span>
                </div>
            </div>

            {/* Linear meter */}
            <div className="relative">
                <div className="risk-meter">
                    {/* Needle */}
                    <div
                        className="risk-needle"
                        style={{ left: needlePosition }}
                    />
                </div>

                {/* Labels */}
                <div className="flex justify-between mt-2 text-xs text-dark-500">
                    <span>Safe</span>
                    <span>Medium</span>
                    <span>Critical</span>
                </div>
            </div>
        </div>
    );
}
