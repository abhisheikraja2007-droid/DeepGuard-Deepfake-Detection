import { useTranslation } from 'react-i18next';
import { X, Shield, AlertTriangle } from 'lucide-react';

export default function EthicsDisclaimer({ isOpen, onClose }) {
    const { t } = useTranslation();

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-dark-950/80 backdrop-blur-sm flex items-center justify-center z-[100] p-4">
            <div className="glass-card max-w-lg w-full p-6 relative">
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 p-2 hover:bg-dark-700 rounded-lg transition-colors"
                >
                    <X className="w-5 h-5" />
                </button>

                <div className="text-center mb-6">
                    <Shield className="w-16 h-16 mx-auto text-primary-400 mb-4" />
                    <h2 className="text-2xl font-bold">{t('ethics.title')}</h2>
                </div>

                <div className="space-y-4">
                    <div className="p-4 bg-primary-500/10 border border-primary-500/30 rounded-xl">
                        <p className="text-dark-200">{t('ethics.purpose')}</p>
                    </div>

                    <div className="p-4 bg-danger-500/10 border border-danger-500/30 rounded-xl flex items-start gap-3">
                        <AlertTriangle className="w-5 h-5 text-danger-400 flex-shrink-0 mt-0.5" />
                        <p className="text-dark-200">{t('ethics.misuse')}</p>
                    </div>
                </div>

                <button
                    onClick={onClose}
                    className="btn-primary w-full mt-6"
                >
                    I Understand
                </button>
            </div>
        </div>
    );
}
