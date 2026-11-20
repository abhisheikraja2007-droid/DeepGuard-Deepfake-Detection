import { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Lightbulb, ChevronDown, ChevronUp } from 'lucide-react';

export default function ExplainPanel({ explanation }) {
    const { t } = useTranslation();
    const [expanded, setExpanded] = useState(false);

    if (!explanation) return null;

    return (
        <div className="glass-card p-6">
            <button
                onClick={() => setExpanded(!expanded)}
                className="w-full flex items-center justify-between"
            >
                <h3 className="text-lg font-semibold flex items-center gap-2">
                    <Lightbulb className="w-5 h-5 text-yellow-400" />
                    {t('explain.title')}
                </h3>
                {expanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
            </button>

            {expanded && (
                <div className="mt-4 space-y-4">
                    <div className="p-4 bg-dark-800/50 rounded-xl">
                        <h4 className="text-sm font-medium text-dark-400 mb-2">{t('explain.simple')}</h4>
                        <p className="text-dark-200">{explanation}</p>
                    </div>
                </div>
            )}
        </div>
    );
}
