import React, { useState } from 'react';

// ── Color palettes per cluster ────────────────────────────────────────────────
const OVERALL_STYLE = {
  0: { header: 'bg-rose-600',    border: 'border-rose-200',    text: 'text-rose-700',    bg: 'bg-rose-50',    bar: 'bg-rose-500',    badge: 'bg-rose-100 text-rose-800' },
  1: { header: 'bg-amber-500',   border: 'border-amber-200',   text: 'text-amber-700',   bg: 'bg-amber-50',   bar: 'bg-amber-400',   badge: 'bg-amber-100 text-amber-800' },
  2: { header: 'bg-emerald-600', border: 'border-emerald-200', text: 'text-emerald-700', bg: 'bg-emerald-50', bar: 'bg-emerald-500', badge: 'bg-emerald-100 text-emerald-800' },
};

const FLUID_STYLE = {
  0: { border: 'border-blue-200',   text: 'text-blue-700',   bg: 'bg-blue-50',   badge: 'bg-blue-100 text-blue-800',   dot: 'bg-blue-500' },
  1: { border: 'border-indigo-200', text: 'text-indigo-700', bg: 'bg-indigo-50', badge: 'bg-indigo-100 text-indigo-800', dot: 'bg-indigo-500' },
  2: { border: 'border-amber-200',  text: 'text-amber-700',  bg: 'bg-amber-50',  badge: 'bg-amber-100 text-amber-800',  dot: 'bg-amber-500' },
  3: { border: 'border-red-200',    text: 'text-red-700',    bg: 'bg-red-50',    badge: 'bg-red-100 text-red-800',      dot: 'bg-red-500' },
};

const PROGNOSIS_STYLE = {
  0: { border: 'border-red-200',     text: 'text-red-700',     bg: 'bg-red-50',     badge: 'bg-red-100 text-red-800',      bar: 'bg-red-400' },
  1: { border: 'border-amber-200',   text: 'text-amber-700',   bg: 'bg-amber-50',   badge: 'bg-amber-100 text-amber-800',  bar: 'bg-amber-400' },
  2: { border: 'border-emerald-200', text: 'text-emerald-700', bg: 'bg-emerald-50', badge: 'bg-emerald-100 text-emerald-800', bar: 'bg-emerald-500' },
};

const INFLAMMATION_STYLE = {
  0: { border: 'border-teal-200',   text: 'text-teal-700',   bg: 'bg-teal-50',   badge: 'bg-teal-100 text-teal-800',   dot: 'bg-teal-500' },
  1: { border: 'border-amber-200',  text: 'text-amber-700',  bg: 'bg-amber-50',  badge: 'bg-amber-100 text-amber-800',  dot: 'bg-amber-500' },
  2: { border: 'border-red-200',    text: 'text-red-700',    bg: 'bg-red-50',    badge: 'bg-red-100 text-red-800',      dot: 'bg-red-500' },
};

// ── Small reusable pieces ─────────────────────────────────────────────────────
const SectionLabel = ({ children }) => (
  <p className="text-[10px] font-bold uppercase tracking-widest text-slate-400 mb-2">
    {children}
  </p>
);

const Badge = ({ children, className }) => (
  <span className={`inline-block text-xs font-bold px-2.5 py-1 rounded-full ${className}`}>
    {children}
  </span>
);

const IntegrityBar = ({ label, value, barColor }) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <span className="text-xs text-slate-500">{label}</span>
      <span className="text-xs font-semibold text-slate-700 tabular-nums">
        {Math.round(value * 100)}%
      </span>
    </div>
    <div className="w-full bg-slate-200 rounded-full h-1.5">
      <div
        className={`h-1.5 rounded-full transition-all duration-700 ${barColor}`}
        style={{ width: `${Math.round(value * 100)}%` }}
      />
    </div>
  </div>
);

const ProbabilityBadge = ({ value }) => {
  if (value == null) return null;
  const pct = Math.round(value * 100);
  return (
    <span className="text-[10px] font-semibold text-slate-400 tabular-nums ml-1">
      {pct}% conf.
    </span>
  );
};

// ── Sub-panels ────────────────────────────────────────────────────────────────

const FluidPanel = ({ fluid }) => {
  if (!fluid) return null;
  const s = FLUID_STYLE[fluid.cluster] ?? FLUID_STYLE[0];
  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} p-4`}>
      <SectionLabel>Fluid Pattern — Anti-VEGF Guidance</SectionLabel>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${s.dot}`} />
        <Badge className={s.badge}>{fluid.severity}</Badge>
        <ProbabilityBadge value={fluid.probability} />
      </div>
      <div className="grid grid-cols-2 gap-2 mb-3">
        <div className="bg-white/70 rounded-lg p-2.5 text-center border border-white">
          <p className="text-[10px] text-slate-400 uppercase tracking-wide mb-0.5">IRF</p>
          <p className={`text-lg font-bold tabular-nums ${s.text}`}>
            {fluid.irf_percent.toFixed(2)}
            <span className="text-xs font-normal ml-0.5">%</span>
          </p>
        </div>
        <div className="bg-white/70 rounded-lg p-2.5 text-center border border-white">
          <p className="text-[10px] text-slate-400 uppercase tracking-wide mb-0.5">SRF</p>
          <p className={`text-lg font-bold tabular-nums ${s.text}`}>
            {fluid.srf_percent.toFixed(2)}
            <span className="text-xs font-normal ml-0.5">%</span>
          </p>
        </div>
      </div>
      <p className="text-xs text-slate-600 leading-relaxed">{fluid.recommendation}</p>
    </div>
  );
};

const PrognosisPanel = ({ prognosis }) => {
  if (!prognosis) return null;
  const s = PROGNOSIS_STYLE[prognosis.cluster] ?? PROGNOSIS_STYLE[1];
  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} p-4`}>
      <SectionLabel>Prognosis — Visual Recovery</SectionLabel>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <Badge className={s.badge}>{prognosis.tier}</Badge>
        <ProbabilityBadge value={prognosis.probability} />
      </div>
      <div className="space-y-2 mb-3">
        <IntegrityBar
          label="ELM Integrity"
          value={prognosis.elm_integrity}
          barColor={s.bar}
        />
        <IntegrityBar
          label="EZ Integrity"
          value={prognosis.ez_integrity}
          barColor={s.bar}
        />
      </div>
      <p className="text-xs text-slate-600 leading-relaxed">{prognosis.outlook}</p>
    </div>
  );
};

const InflammationPanel = ({ inflammation }) => {
  if (!inflammation) return null;
  const s = INFLAMMATION_STYLE[inflammation.cluster] ?? INFLAMMATION_STYLE[0];
  return (
    <div className={`rounded-xl border ${s.border} ${s.bg} p-4`}>
      <SectionLabel>Inflammation — HF Monitoring</SectionLabel>
      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className={`w-2.5 h-2.5 rounded-full shrink-0 ${s.dot}`} />
        <Badge className={s.badge}>{inflammation.level}</Badge>
        <ProbabilityBadge value={inflammation.probability} />
      </div>
      <div className="bg-white/70 rounded-lg px-4 py-3 border border-white mb-3 text-center">
        <p className="text-[10px] text-slate-400 uppercase tracking-wide mb-0.5">
          Hyperreflective Foci
        </p>
        <p className={`text-2xl font-bold tabular-nums ${s.text}`}>
          {inflammation.hf_count}
          <span className="text-xs font-normal ml-1 text-slate-400">detected</span>
        </p>
      </div>
      <p className="text-xs text-slate-600 leading-relaxed">{inflammation.recommendation}</p>
    </div>
  );
};

// ── Main card ─────────────────────────────────────────────────────────────────
const PhenotypingCard = ({ phenotyping }) => {
  const [showExplanation, setShowExplanation] = useState(false);

  if (!phenotyping) return null;

  const { overall, fluid, prognosis, inflammation, explanation, ml_mode } = phenotyping;
  const os = OVERALL_STYLE[overall?.cluster ?? 2];
  const pct = Math.round((overall?.severity_score ?? 0) * 100);

  return (
    <div className={`glass-panel overflow-hidden rounded-2xl border ${os.border}`}>

      {/* ── Overall phenotype header ─────────────────────────────────────── */}
      <div className={`${os.header} p-5 text-white`}>
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <p className="text-white/70 text-[10px] font-bold uppercase tracking-widest mb-1">
              Overall Disease Phenotype
            </p>
            <h3 className="text-xl font-bold leading-tight">{overall?.label}</h3>
            <p className="text-white/75 text-xs mt-1.5 leading-relaxed">
              {overall?.description}
            </p>
          </div>
          <div className="flex flex-col items-end gap-1.5 shrink-0">
            {overall?.probability != null && (
              <span className="text-[10px] font-semibold bg-white/20 px-2.5 py-1 rounded-full border border-white/30">
                {Math.round(overall.probability * 100)}% conf.
              </span>
            )}
            <span className={`text-[10px] font-bold px-2.5 py-1 rounded-full border ${
              ml_mode
                ? 'bg-indigo-900/40 border-indigo-300/40 text-indigo-100'
                : 'bg-white/10 border-white/20 text-white/70'
            }`}>
              {ml_mode ? 'GMM — Data-driven' : 'Rule-based'}
            </span>
          </div>
        </div>

        {/* Severity score bar */}
        <div className="mt-4">
          <div className="flex justify-between items-center mb-1.5">
            <span className="text-white/70 text-[10px] font-bold uppercase tracking-wide">
              Severity Score
            </span>
            <span className="text-white font-bold tabular-nums text-sm">{pct} / 100</span>
          </div>
          <div className="w-full bg-white/20 rounded-full h-2">
            <div
              className="bg-white h-2 rounded-full transition-all duration-700"
              style={{ width: `${pct}%` }}
            />
          </div>
          <div className="flex justify-between text-white/40 text-[10px] mt-1">
            <span>Low</span><span>Moderate</span><span>High</span>
          </div>
        </div>
      </div>

      {/* ── Three independent predictors ─────────────────────────────────── */}
      <div className="p-4 space-y-3 bg-slate-50">
        <FluidPanel fluid={fluid} />
        <PrognosisPanel prognosis={prognosis} />
        <InflammationPanel inflammation={inflammation} />
      </div>

      {/* ── Explanation (collapsible) ─────────────────────────────────────── */}
      <div className="border-t border-slate-100 bg-white">
        <button
          onClick={() => setShowExplanation(v => !v)}
          className="w-full flex items-center justify-between px-4 py-3 text-left"
        >
          <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
            {ml_mode ? 'GMM Analysis Details' : 'Rule-based Reasoning'}
          </span>
          <svg
            className={`w-4 h-4 text-slate-400 transition-transform ${showExplanation ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showExplanation && (
          <div className="px-4 pb-4">
            <pre className="text-xs text-slate-600 leading-relaxed whitespace-pre-wrap font-mono bg-slate-50 p-3 rounded-lg border border-slate-100">
              {explanation}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};

export default PhenotypingCard;
