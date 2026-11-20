import { useTranslation } from 'react-i18next';
import { Zap, Scale, Target } from 'lucide-react';

export default function ModeSelector({ selectedMode, onModeChange }) {
    const { t } = useTranslation();

    const modes = [
        { id: 'fast', icon: Zap, color: 'text-green-400', bg: 'bg-green-500/10 border-green-500/30' },
        { id: 'balanced', icon: Scale, color: 'text-primary-400', bg: 'bg-primary-500/10 border-primary-500/30' },
        { id: 'accuracy', icon: Target, color: 'text-purple-400', bg: 'bg-purple-500/10 border-purple-500/30' },
    ];

    return (
        <div className="glass-card p-4 mb-6">
            <h3 className="text-sm font-medium text-dark-400 mb-3">{t('modes.title')}</h3>
            <div className="grid grid-cols-3 gap-3">
                {modes.map((mode) => (
                    <button
                        key={mode.id}
                        onClick={() => onModeChange(mode.id)}
                        className={`p-4 rounded-xl border transition-all ${selectedMode === mode.id
                                ? `${mode.bg} border-2`
                                : 'bg-dark-800/30 border-dark-700/50 hover:bg-dark-700/30'
                            }`}
                    >
                        <mode.icon className={`w-6 h-6 mx-auto mb-2 ${mode.color}`} />
                        <p className="font-medium text-sm">{t(`modes.${mode.id}`)}</p>
                        <p className="text-xs text-dark-500 mt-1">{t(`modes.${mode.id}Desc`)}</p>
                    </button>
                ))}
            </div>
        </div>
    );
}
