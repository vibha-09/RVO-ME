import React from 'react';

const PredictionCard = ({ prediction }) => {
  
  const getSeverityStyles = (severity) => {
    switch (severity.toLowerCase()) {
        case 'severe':
            return {
                bg: 'bg-red-500',
                border: 'border-red-200',
                text: 'text-red-700',
                lightBg: 'bg-red-50',
                icon: (
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                )
            };
        case 'moderate':
            return {
                bg: 'bg-orange-500',
                border: 'border-orange-200',
                text: 'text-orange-700',
                lightBg: 'bg-orange-50',
                icon: (
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                )
            };
        case 'mild':
            return {
                bg: 'bg-yellow-500',
                border: 'border-yellow-200',
                text: 'text-yellow-700',
                lightBg: 'bg-yellow-50',
                icon: (
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                )
            };
        case 'normal':
        default:
            return {
                bg: 'bg-emerald-500',
                border: 'border-emerald-200',
                text: 'text-emerald-700',
                lightBg: 'bg-emerald-50',
                icon: (
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                )
            };
    }
  };

  const style = getSeverityStyles(prediction.severity);

  return (
    <div className={`glass-panel overflow-hidden rounded-2xl border ${style.border}`}>
        {/* Header Block */}
        <div className={`${style.bg} p-6 text-white flex flex-col items-center justify-center space-y-3`}>
            <div className="p-3 bg-white/20 rounded-full backdrop-blur-sm">
                {style.icon}
            </div>
            <div className="text-center">
                <p className="text-white/80 font-medium text-sm tracking-wide uppercase">Severity Level</p>
                <h3 className="text-3xl font-bold tracking-tight">{prediction.severity}</h3>
            </div>
        </div>

        {/* Content Block */}
        <div className={`p-6 ${style.lightBg} h-full`}>
            <div className="space-y-6">
                <div>
                    <h4 className={`text-sm font-bold uppercase tracking-wider ${style.text} mb-2`}>Recommendation</h4>
                    <p className="text-slate-800 font-medium text-lg border-l-4 border-current pl-3 py-1">
                        {prediction.recommendation}
                    </p>
                </div>

                <div className="h-px bg-slate-200/50 w-full rounded-full"></div>

                <div>
                    <h4 className="text-sm font-bold uppercase tracking-wider text-slate-500 mb-2">Clinical Reasoning</h4>
                    <p className="text-slate-600 text-sm leading-relaxed bg-white/50 p-4 rounded-xl border border-white/50 shadow-sm inline-block">
                        {prediction.reasoning}
                    </p>
                </div>
            </div>
        </div>
    </div>
  );
};

export default PredictionCard;
