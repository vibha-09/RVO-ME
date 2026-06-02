import React from 'react';

const MetricCard = ({ title, value, unit, highlight = false, colorClass = "text-medical-blue" }) => (
    <div className={`p-4 rounded-xl border transition-all ${highlight ? 'bg-blue-50/50 border-blue-200' : 'bg-white border-slate-100 shadow-sm'}`}>
        <p className="text-sm font-medium text-slate-500 mb-1">{title}</p>
        <div className="flex items-baseline space-x-1">
            <span className={`text-2xl font-bold ${colorClass}`}>{value}</span>
            {unit && <span className="text-sm text-slate-400 font-medium">{unit}</span>}
        </div>
    </div>
);

const MetricsDisplay = ({ metrics }) => {
  return (
    <div className="glass-panel p-6 rounded-2xl">
      <h3 className="text-xl font-semibold mb-4 text-slate-800">Biomarker Quantification</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard 
              title="SRF Area" 
              value={metrics.srf_area_percent} 
              unit="%" 
              colorClass="text-blue-600"
          />
          <MetricCard 
              title="IRF Area" 
              value={metrics.irf_area_percent} 
              unit="%" 
              colorClass="text-red-500"
          />
          <MetricCard 
              title="HF Count" 
              value={metrics.hf_count} 
              unit="spots" 
              highlight={metrics.hf_count > 0}
              colorClass={metrics.hf_count > 0 ? "text-cyan-600" : "text-slate-700"}
          />
          <MetricCard 
              title="Total Fluid" 
              value={metrics.total_fluid_area_percent} 
              unit="%" 
              colorClass="text-purple-600"
          />
          <MetricCard 
              title="ELM Integrity" 
              value={metrics.elm_integrity} 
              unit="ratio" 
              colorClass={(metrics.elm_integrity < 0.8) ? "text-orange-500" : "text-emerald-500"}
          />
          <MetricCard 
              title="EZ Integrity" 
              value={metrics.ez_integrity} 
              unit="ratio" 
              colorClass={(metrics.ez_integrity < 0.8) ? "text-orange-500" : "text-emerald-500"}
          />
      </div>
    </div>
  );
};

export default MetricsDisplay;
