// notebook.jsx — Lab notebook / insights page
const { useState, useMemo } = React;

// ─────────────────────────────────────────────────────────────────────────────
// Mock data — shaped like what the API would actually return
const SUMMARY = {
  total_graded: 412,
  this_period: 87,
  delta_pct: +14,
  avg_final: 8.3,
  high_grade: 9.5,
  agreement_rate: 0.83, // % of grades within ±0.5 of operator correction
  drift_7d: -0.04,
  pending_calibration: 12,
};

// Grade distribution histogram (PSA scale 1–10, half-grade buckets)
const GRADE_DIST = [
  { g: '5.0', n: 4 }, { g: '5.5', n: 3 },
  { g: '6.0', n: 8 }, { g: '6.5', n: 11 },
  { g: '7.0', n: 22 }, { g: '7.5', n: 31 },
  { g: '8.0', n: 58 }, { g: '8.5', n: 74 },
  { g: '9.0', n: 96 }, { g: '9.5', n: 71 },
  { g: '10', n: 34 },
];

// Calibration drift (model − operator) over time, biweekly
const DRIFT_SERIES = [
  { d: 'Feb 12', v: +0.18 },{ d: 'Feb 26', v: +0.14 },{ d: 'Mar 12', v: +0.11 },
  { d: 'Mar 26', v: +0.07 },{ d: 'Apr 09', v: +0.03 },{ d: 'Apr 23', v: -0.01 },
  { d: 'May 07', v: -0.04 },
];

// Per-criterion confidence + agreement
const CRITERIA = [
  { k: 'Centering', model: 'GAP-regressor v0.6', conf: 0.94, agreement: 0.91, n: 412, drift: -0.02, status: 'stable' },
  { k: 'Edges',     model: 'edge-defect v0.3',   conf: 0.81, agreement: 0.78, n: 412, drift: -0.04, status: 'stable' },
  { k: 'Corners',   model: 'corner-cls v0.2',    conf: 0.71, agreement: 0.66, n: 184, drift: -0.11, status: 'shaky' },
  { k: 'Surface',   model: 'surface-mp v0.2',    conf: 0.74, agreement: 0.69, n: 412, drift: -0.06, status: 'shaky' },
];

// Counterfeit detector reliability over the period
const DETECTORS = [
  { k: 'rosette',        l: 'Rosette pattern',  fires: 14, tp: 13, fp: 1,  pr: 0.93, anchor: 'always' },
  { k: 'color',          l: 'Color delta E',    fires: 22, tp: 18, fp: 4,  pr: 0.82, anchor: 'always' },
  { k: 'embedding_anom', l: 'Embedding anom.',  fires: 9,  tp: 9,  fp: 0,  pr: 1.00, anchor: 'identify' },
  { k: 'typography',     l: 'Typography OCR',   fires: 6,  tp: 5,  fp: 1,  pr: 0.83, anchor: 'name' },
  { k: 'holographic',    l: 'Holo parallax',    fires: 11, tp: 10, fp: 1,  pr: 0.91, anchor: 'tilt_30' },
  { k: 'knn_reference',  l: 'kNN reference',    fires: 18, tp: 15, fp: 3,  pr: 0.83, anchor: '≥k refs' },
  { k: 'substrate',      l: 'Substrate b*',     fires: 4,  tp: 4,  fp: 0,  pr: 1.00, anchor: 'flash' },
];

// Recent regrade events (where operator overrode the model)
const REGRADES = [
  { ts: 'May 06 14:22', card: 'Charizard · Base · 4/102 1st Ed.', from: 8.5, to: 8.0, reason: 'Surface scratch missed · diagonal under holo', criterion: 'surface' },
  { ts: 'May 04 09:48', card: 'Pikachu Illustrator · promo',      from: 9.0, to: 9.5, reason: 'Centering re-measured, model under-rotated dewarp', criterion: 'centering' },
  { ts: 'May 02 19:11', card: 'Lugia · Neo Genesis 9/111',         from: 7.5, to: 7.0, reason: 'Corner softening on BR not flagged', criterion: 'corners' },
  { ts: 'Apr 30 11:03', card: 'Black Lotus · Beta',                from: 6.0, to: 6.5, reason: 'Edge whitening overstated · hue threshold tweaked', criterion: 'edges' },
];

const PIPELINE_PERF = {
  median_s: 4.7, p95_s: 8.9, n: 412,
  stages: [
    { k: 'preflight',   median: 0.4, p95: 0.6 },
    { k: 'detection',   median: 1.4, p95: 2.2 },
    { k: 'identify',    median: 0.7, p95: 1.4 },
    { k: 'counterfeit', median: 1.2, p95: 2.6 },
    { k: 'grade',       median: 0.9, p95: 1.8 },
    { k: 'cert',        median: 0.1, p95: 0.3 },
  ],
};

const FLAG_PATTERNS = [
  { l: 'Holo parallax low confidence', n: 9,  trend: '+3', why: 'Tilt shots under-illuminated for foil cards' },
  { l: 'Corner subgrade missing',       n: 47, trend: '+12', why: 'Macro corner shots optional · 47 subs lacked all 4' },
  { l: 'Substrate b* needs flash',      n: 38, trend: '−4', why: 'Flash shot only captured on 9% of subs' },
  { l: 'Typography below threshold',    n: 6,  trend: '+1', why: '5 different sets · OCR struggles on multilingual prints' },
];

// ─────────────────────────────────────────────────────────────────────────────
function App() {
  const DEFAULTS = window.__NOTEBOOK_TWEAK_DEFAULTS_RAW || { timeRange: '90d', view: 'overview', showAnnotations: true, density: 'comfortable' };
  const [t, setTweak] = useTweaks(DEFAULTS);

  return (
    <main className="nb-main">
      <Header t={t} setTweak={setTweak} />

      <Section
        eyebrow="01"
        title="Where the system stands"
        body="A 90-day read on how the bench performs against your own corrections. Numbers from completed submissions only."
      >
        <SummaryStrip />
      </Section>

      <Section
        eyebrow="02"
        title="Grade distribution"
        body="Half-grade buckets across 412 graded cards. Brightness scales with count. Shape is the truth — no normalization."
      >
        <GradeDistribution />
      </Section>

      <Section
        eyebrow="03"
        title="Calibration drift"
        body="Mean signed gap between model output and operator correction, biweekly. Trending toward zero is the goal; under-grading is preferred to over."
      >
        <CalibrationDrift annotations={t.showAnnotations} />
      </Section>

      <Section
        eyebrow="04"
        title="Per-criterion reliability"
        body="Each subgrade model judged on its own. Confidence is what the model claims; agreement is whether you accepted it."
      >
        <CriteriaTable />
      </Section>

      <Section
        eyebrow="05"
        title="Counterfeit ensemble"
        body="Seven detectors. True / false positive accounting against post-hoc operator review. Precision targets ≥ 0.85 before promotion."
      >
        <DetectorTable />
      </Section>

      <Section
        eyebrow="06"
        title="Persistent flags"
        body="Patterns that keep showing up in the audit log. Each one is a hint at the next training cycle's priority."
      >
        <FlagPatterns />
      </Section>

      <Section
        eyebrow="07"
        title="Pipeline performance"
        body="Median + p95 stage durations across the period. Useful when latency drifts; less so when it doesn't."
      >
        <PipelinePerf />
      </Section>

      <Section
        eyebrow="08"
        title="Recent regrade events"
        body="Cards where you overrode the model. These feed the calibration loop and shape the next training run."
      >
        <RegradeLog />
      </Section>

      <Footnote />
      <NotebookTweaks t={t} setTweak={setTweak} />

      <style>{`
        .nb-main {
          max-width: 1280px; margin: 24px auto; padding: 0 24px 100px;
          display: flex; flex-direction: column; gap: 36px;
        }
        .nb-section { display: grid; grid-template-columns: 280px 1fr; gap: 32px; }
        @media (max-width: 1080px) { .nb-section { grid-template-columns: 1fr; gap: 14px; } }
        .nb-eyebrow {
          font-family: 'Geist Mono', monospace; font-size: 10px;
          color: var(--ink-3); letter-spacing: 0.22em; text-transform: uppercase;
        }
        .nb-title {
          font-family: 'Instrument Serif', serif; font-size: 28px;
          color: var(--ink-0); margin: 6px 0 10px; letter-spacing: -0.012em;
          font-weight: 400; line-height: 1.15;
        }
        .nb-body { font-size: 13px; color: var(--ink-2); line-height: 1.55; max-width: 26em; }
      `}</style>
    </main>
  );
}

function Header({ t, setTweak }) {
  return (
    <header style={{
      padding: '20px 22px', background: 'var(--bg-1)',
      border: '1px solid var(--line)', borderRadius: 8,
      display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 24, flexWrap: 'wrap',
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
          Lab notebook · operator view
        </span>
        <h1 style={{ fontFamily: 'Instrument Serif, serif', fontSize: 36, fontWeight: 400, color: 'var(--ink-0)', margin: 0, letterSpacing: '-0.018em', lineHeight: 1 }}>
          Is the system getting better?
        </h1>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', marginTop: 4 }}>
          Window <span style={{ color: 'var(--lime)' }}>{t.timeRange}</span>
          <span style={{ color: 'var(--ink-5)', margin: '0 8px' }}>·</span>
          412 submissions · 87 in this period
          <span style={{ color: 'var(--ink-5)', margin: '0 8px' }}>·</span>
          last regrade <span style={{ color: 'var(--ink-1)' }}>14h ago</span>
        </span>
      </div>

      <div style={{ display: 'flex', gap: 6 }}>
        {['7d','30d','90d','all'].map(r => (
          <button key={r} onClick={() => setTweak('timeRange', r)} style={{
            all: 'unset', cursor: 'pointer',
            padding: '6px 12px', borderRadius: 3,
            background: t.timeRange === r ? 'var(--bg-3)' : 'transparent',
            border: '1px solid ' + (t.timeRange === r ? 'var(--line)' : 'var(--line-2)'),
            color: t.timeRange === r ? 'var(--ink-0)' : 'var(--ink-3)',
            fontFamily: 'Geist Mono', fontSize: 10, letterSpacing: '0.14em', textTransform: 'uppercase',
          }}>{r}</button>
        ))}
      </div>
    </header>
  );
}

function Section({ eyebrow, title, body, children }) {
  return (
    <section className="nb-section">
      <div>
        <div className="nb-eyebrow">§ {eyebrow}</div>
        <h2 className="nb-title">{title}</h2>
        <p className="nb-body">{body}</p>
      </div>
      <div>{children}</div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function SummaryStrip() {
  const items = [
    { k: 'Total graded',     v: '412',   sub: SUMMARY.this_period + ' this period · +' + SUMMARY.delta_pct + '%' },
    { k: 'Mean final',       v: '8.30',  sub: 'σ 1.21 · over 412 cards' },
    { k: 'Highest grade',    v: '9.50',  sub: 'Charizard · Base 1st Ed' },
    { k: 'Operator agree.',  v: '83%',   sub: 'within ±0.5 of correction' },
    { k: 'Drift (7d)',       v: '−0.04', sub: 'model under-grading slightly', tone: 'lime' },
    { k: 'Awaiting calib.',  v: '12',    sub: '7 corners · 4 surface · 1 edges', tone: 'amber' },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1, background: 'var(--line)', border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden' }}>
      {items.map(it => (
        <div key={it.k} style={{ padding: 16, background: 'var(--bg-1)', display: 'flex', flexDirection: 'column', gap: 4 }}>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{it.k}</span>
          <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 38, fontWeight: 400, color: it.tone === 'lime' ? 'var(--lime)' : it.tone === 'amber' ? 'var(--amber)' : 'var(--ink-0)', letterSpacing: '-0.01em', lineHeight: 1 }}>{it.v}</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>{it.sub}</span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function GradeDistribution() {
  const max = Math.max(...GRADE_DIST.map(d => d.n));
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 18 }}>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 6, height: 220 }}>
        {GRADE_DIST.map(d => {
          const h = (d.n / max) * 100;
          const bright = 0.25 + (d.n / max) * 0.75;
          return (
            <div key={d.g} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', alignItems: 'center', gap: 4, height: '100%' }}>
              <span className="mono" style={{ fontSize: 9, color: 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}>{d.n}</span>
              <div style={{
                width: '100%', height: h + '%',
                background: `rgba(190,242,100,${bright})`,
                borderTop: '2px solid var(--lime)',
                minHeight: 4,
              }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 6, marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--line-2)' }}>
        {GRADE_DIST.map(d => (
          <span key={d.g} className="mono" style={{ flex: 1, textAlign: 'center', fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>{d.g}</span>
        ))}
      </div>
      <div style={{ marginTop: 14, display: 'flex', gap: 12, alignItems: 'center', color: 'var(--ink-3)' }}>
        <span className="mono" style={{ fontSize: 10, letterSpacing: '0.16em', textTransform: 'uppercase' }}>Mode 9.0</span>
        <span style={{ color: 'var(--line)' }}>·</span>
        <span className="mono" style={{ fontSize: 10, letterSpacing: '0.16em', textTransform: 'uppercase' }}>Median 8.5</span>
        <span style={{ color: 'var(--line)' }}>·</span>
        <span className="mono" style={{ fontSize: 10, letterSpacing: '0.16em', textTransform: 'uppercase' }}>74% above 8.0</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CalibrationDrift({ annotations }) {
  const W = 720, H = 220, padL = 50, padR = 24, padT = 20, padB = 36;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const min = -0.25, max = 0.25;
  const x = i => padL + (i / (DRIFT_SERIES.length - 1)) * innerW;
  const y = v => padT + (1 - (v - min) / (max - min)) * innerH;
  const path = DRIFT_SERIES.map((d, i) => (i === 0 ? 'M' : 'L') + x(i) + ' ' + y(d.v)).join(' ');
  const zeroY = y(0);

  const annots = [
    { i: 1, label: 'corner-cls v0.2', desc: 'Stricter corner softening threshold' },
    { i: 4, label: 'centering refit', desc: 'GAP regressor retrained on 180 corrections' },
  ];

  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 18 }}>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
        {/* grid */}
        {[-0.2, -0.1, 0, 0.1, 0.2].map(g => (
          <g key={g}>
            <line x1={padL} x2={W - padR} y1={y(g)} y2={y(g)} stroke={g === 0 ? 'var(--line)' : 'var(--line-2)'} strokeDasharray={g === 0 ? '0' : '2 4'} />
            <text x={padL - 8} y={y(g) + 3} textAnchor="end" fontFamily="Geist Mono" fontSize="9" fill="var(--ink-3)">{g > 0 ? '+' : ''}{g.toFixed(2)}</text>
          </g>
        ))}
        {/* zero label */}
        <text x={W - padR + 4} y={zeroY + 3} fontFamily="Geist Mono" fontSize="8" fill="var(--ink-3)" letterSpacing="0.12em">0</text>
        {/* line */}
        <path d={path} stroke="var(--lime)" strokeWidth="1.6" fill="none" />
        {/* fill */}
        <path d={path + ` L ${x(DRIFT_SERIES.length-1)} ${zeroY} L ${x(0)} ${zeroY} Z`} fill="rgba(190,242,100,0.07)" />
        {/* points + xlabels */}
        {DRIFT_SERIES.map((d, i) => (
          <g key={d.d}>
            <circle cx={x(i)} cy={y(d.v)} r="3" fill="var(--bg-1)" stroke="var(--lime)" strokeWidth="1.4" />
            <text x={x(i)} y={H - padB + 18} textAnchor="middle" fontFamily="Geist Mono" fontSize="9" fill="var(--ink-3)" letterSpacing="0.06em">{d.d}</text>
          </g>
        ))}
        {/* annotations */}
        {annotations && annots.map(a => (
          <g key={a.i}>
            <line x1={x(a.i)} y1={y(DRIFT_SERIES[a.i].v) - 6} x2={x(a.i)} y2={padT + 6} stroke="var(--ink-3)" strokeDasharray="2 3" />
            <rect x={x(a.i) - 4} y={padT - 2} width={Math.max(120, a.label.length * 6.4)} height={28} fill="var(--bg-2)" stroke="var(--line)" rx="2" />
            <text x={x(a.i) + 4} y={padT + 9} fontFamily="Geist Mono" fontSize="9" fill="var(--lime)" letterSpacing="0.04em">{a.label}</text>
            <text x={x(a.i) + 4} y={padT + 21} fontFamily="Geist Mono" fontSize="8" fill="var(--ink-3)" letterSpacing="0.02em">{a.desc}</text>
          </g>
        ))}
      </svg>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 8, paddingTop: 8, borderTop: '1px solid var(--line-2)' }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>Direction · model − operator (signed)</span>
        <span className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>↘ from +0.18 to −0.04</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CriteriaTable() {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden' }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.4fr 1.4fr 0.8fr 0.8fr 0.6fr 0.8fr 0.6fr', padding: '10px 16px', background: 'var(--bg-2)', borderBottom: '1px solid var(--line)' }}>
        {['Criterion','Model','Conf.','Agree','n','Drift','State'].map(h => (
          <span key={h} className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{h}</span>
        ))}
      </div>
      {CRITERIA.map(c => (
        <div key={c.k} style={{ display: 'grid', gridTemplateColumns: '1.4fr 1.4fr 0.8fr 0.8fr 0.6fr 0.8fr 0.6fr', padding: '12px 16px', borderBottom: '1px solid var(--line-2)', alignItems: 'center' }}>
          <span style={{ fontSize: 13, color: 'var(--ink-0)', fontWeight: 500 }}>{c.k}</span>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-2)' }}>{c.model}</span>
          <Bar v={c.conf} label={c.conf.toFixed(2)} />
          <Bar v={c.agreement} label={(c.agreement*100).toFixed(0) + '%'} />
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>{c.n}</span>
          <span className="mono" style={{ fontSize: 11, color: c.drift < -0.05 ? 'var(--rose)' : c.drift < 0 ? 'var(--lime)' : 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}>
            {c.drift > 0 ? '+' : ''}{c.drift.toFixed(2)}
          </span>
          <StatusPill status={c.status} />
        </div>
      ))}
    </div>
  );
}

function Bar({ v, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 4, background: 'var(--bg-3)', borderRadius: 1, position: 'relative' }}>
        <div style={{ width: (v*100) + '%', height: '100%', background: v > 0.85 ? 'var(--lime)' : v > 0.7 ? 'var(--amber)' : 'var(--rose)', borderRadius: 1 }} />
      </div>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', minWidth: 28, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{label}</span>
    </div>
  );
}

function StatusPill({ status }) {
  const map = {
    stable: { c: 'var(--lime)', l: 'stable' },
    shaky:  { c: 'var(--amber)', l: 'shaky' },
    bad:    { c: 'var(--rose)', l: 'bad' },
  };
  const it = map[status] || map.stable;
  return (
    <span className="mono" style={{
      fontSize: 9, padding: '3px 8px', borderRadius: 99,
      background: it.c + '14', color: it.c, border: '1px solid ' + it.c + '40',
      letterSpacing: '0.14em', textTransform: 'uppercase', justifySelf: 'start',
    }}>{it.l}</span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function DetectorTable() {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 14 }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.6fr 0.7fr 1.4fr 0.8fr', gap: 10, padding: '6px 8px', borderBottom: '1px solid var(--line-2)', marginBottom: 4 }}>
        {['Detector','Anchor','Fires','TP / FP','Precision'].map(h => (
          <span key={h} className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{h}</span>
        ))}
      </div>
      {DETECTORS.map(d => (
        <div key={d.k} style={{ display: 'grid', gridTemplateColumns: '1.2fr 1.6fr 0.7fr 1.4fr 0.8fr', gap: 10, padding: '10px 8px', borderBottom: '1px solid var(--line-2)', alignItems: 'center' }}>
          <span style={{ fontSize: 12, color: 'var(--ink-1)' }}>{d.l}</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>needs {d.anchor}</span>
          <span className="mono" style={{ fontSize: 12, color: 'var(--ink-2)', fontVariantNumeric: 'tabular-nums' }}>{d.fires}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ flex: 1, display: 'flex', height: 4, borderRadius: 1, overflow: 'hidden', background: 'var(--bg-3)' }}>
              <div style={{ width: (d.tp / (d.tp+d.fp+0.001) * 100) + '%', background: 'var(--lime)' }} />
              <div style={{ width: (d.fp / (d.tp+d.fp+0.001) * 100) + '%', background: 'var(--rose)' }} />
            </div>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums', minWidth: 50, textAlign: 'right' }}>{d.tp} / {d.fp}</span>
          </div>
          <span className="mono" style={{ fontSize: 12, color: d.pr >= 0.85 ? 'var(--lime)' : 'var(--amber)', fontVariantNumeric: 'tabular-nums' }}>{d.pr.toFixed(2)}</span>
        </div>
      ))}
      <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', padding: '10px 8px 0', letterSpacing: '0.04em' }}>
        Promotion bar · precision ≥ 0.85 · n ≥ 30. Three detectors below the bar; combine into ensemble at lower weight.
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function FlagPatterns() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
      {FLAG_PATTERNS.map(f => (
        <div key={f.l} style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 14, display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
            <span style={{ fontSize: 13, color: 'var(--ink-0)', fontWeight: 500 }}>{f.l}</span>
            <span className="mono" style={{ fontSize: 10, color: f.trend.startsWith('+') ? 'var(--amber)' : 'var(--lime)', letterSpacing: '0.1em' }}>{f.trend}</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 32, color: 'var(--ink-0)', letterSpacing: '-0.01em', lineHeight: 1 }}>{f.n}</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>occurrences</span>
          </div>
          <span style={{ fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.45, marginTop: 2 }}>{f.why}</span>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function PipelinePerf() {
  const max = Math.max(...PIPELINE_PERF.stages.map(s => s.p95));
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 18 }}>
      <div style={{ display: 'flex', gap: 24, marginBottom: 14, paddingBottom: 12, borderBottom: '1px solid var(--line-2)' }}>
        <Stat l="Median total"  v={PIPELINE_PERF.median_s + 's'} />
        <Stat l="p95 total"     v={PIPELINE_PERF.p95_s + 's'} />
        <Stat l="Submissions"   v={PIPELINE_PERF.n} />
        <Stat l="Throughput"    v="≈ 87 / 90d" />
      </div>
      {PIPELINE_PERF.stages.map(s => (
        <div key={s.k} style={{ display: 'grid', gridTemplateColumns: '120px 1fr 60px 60px', gap: 12, padding: '8px 0', alignItems: 'center', borderBottom: '1px dashed var(--line-2)' }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', letterSpacing: '0.04em' }}>{s.k}</span>
          <div style={{ position: 'relative', height: 8, background: 'var(--bg-3)', borderRadius: 1 }}>
            <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: (s.p95 / max * 100) + '%', background: 'var(--bg-3)', borderRight: '1px solid var(--line)' }} />
            <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: (s.median / max * 100) + '%', background: 'var(--lime)', borderRadius: '1px 0 0 1px' }} />
          </div>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-2)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{s.median}s</span>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-4)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{s.p95}s</span>
        </div>
      ))}
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', marginTop: 10, letterSpacing: '0.16em', textTransform: 'uppercase', display: 'flex', gap: 14 }}>
        <span><span style={{ display: 'inline-block', width: 10, height: 4, background: 'var(--lime)', verticalAlign: 'middle', marginRight: 6 }} />median</span>
        <span><span style={{ display: 'inline-block', width: 10, height: 4, background: 'var(--bg-3)', border: '1px solid var(--line)', verticalAlign: 'middle', marginRight: 6 }} />p95</span>
      </div>
    </div>
  );
}

function Stat({ l, v }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{l}</span>
      <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 22, color: 'var(--ink-0)', letterSpacing: '-0.01em', lineHeight: 1 }}>{v}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function RegradeLog() {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden' }}>
      {REGRADES.map((r, i) => (
        <div key={i} style={{ padding: '14px 16px', borderBottom: i < REGRADES.length - 1 ? '1px solid var(--line-2)' : 'none', display: 'grid', gridTemplateColumns: '110px 1fr auto', gap: 14, alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.06em' }}>{r.ts}</span>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, minWidth: 0 }}>
            <span style={{ fontSize: 13, color: 'var(--ink-0)', fontWeight: 500 }}>{r.card}</span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', letterSpacing: '0.02em' }}>
              <span style={{ color: 'var(--amber)' }}>{r.criterion}</span>
              <span style={{ color: 'var(--ink-5)', margin: '0 6px' }}>·</span>
              {r.reason}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', fontVariantNumeric: 'tabular-nums' }}>{r.from.toFixed(1)}</span>
            <span style={{ color: 'var(--ink-4)' }}>→</span>
            <span className="mono" style={{ fontSize: 13, color: r.to > r.from ? 'var(--lime)' : 'var(--rose)', fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>{r.to.toFixed(1)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Footnote() {
  return (
    <footer style={{ borderTop: '1px solid var(--line)', paddingTop: 18, marginTop: 8, display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.14em', textTransform: 'uppercase', maxWidth: 600, lineHeight: 1.6 }}>
        Notebook entries are append-only. The audit log mirrors every event in the Outbox table; nothing here is recomputed retroactively. Disagreements feed the next training run via /apps/api/grader/services/calibration.
      </span>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)' }}>v0.6 · synced 12:14 UTC</span>
    </footer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function NotebookTweaks({ t, setTweak }) {
  return (
    <TweaksPanel>
      <TweakSection label="Window" />
      <TweakRadio label="Time range" value={t.timeRange} options={['7d','30d','90d','all']} onChange={v => setTweak('timeRange', v)} />
      <TweakSection label="Display" />
      <TweakToggle label="Show drift annotations" value={t.showAnnotations} onChange={v => setTweak('showAnnotations', v)} />
      <TweakRadio label="Density" value={t.density} options={['compact','comfortable']} onChange={v => setTweak('density', v)} />
    </TweaksPanel>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
