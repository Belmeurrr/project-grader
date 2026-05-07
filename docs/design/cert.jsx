// cert.jsx — public cert page
const { useState, useMemo, useEffect } = React;

// ─────────────────────────────────────────────────────────────────────────────
// Mock cert payload — shaped from apps/api/grader/schemas/submissions.py
const CERT = {
  cert_id: 'PG-2026-0F1A4C-9D71',
  completed_at: '2026-05-07T14:08:45Z',
  identified_card: {
    variant_id: '4d2f-base1-charizard-holo',
    name: 'Charizard',
    set_code: 'Base Set',
    card_number: '4/102',
    edition: 'Holo · Unlimited',
    confidence: 0.973,
  },
  grades: [{
    scheme: 'psa',
    centering: 9.0,
    corners: 8.5,
    edges: 9.0,
    surface: 9.5,
    final: 9.0,
    confidence: 0.892,
    front_centering_lr: 54.2,
    front_centering_tb: 51.6,
    back_centering_lr: 53.8,
    back_centering_tb: 50.2,
  }],
  authenticity: {
    verdict: 'authentic',
    confidence: 0.973,
    reasons: [
      'Print rosette FFT peak strength within manufacturer reference (0.81)',
      'Holographic parallax flow consistent with WOTC era foil substrate',
      'Substrate b* under flash matches 1999 Topps stock (-2.1 vs ref -2.3)',
      'Typography stroke-width distribution matches set reference (KS=0.04)',
      'No edge-glow or reprint-mask anomalies detected',
    ],
    model_versions: {
      detect: 'yolo11-seg-v0.4.1',
      identify: 'dinov2-pgvec-v1.2',
      centering: 'geometric-v0.3.0',
      edges: 'mad-v0.4.0',
      counterfeit_ensemble: 'v0.3.2',
    },
    detectors: [
      { detector: 'rosette_fft', score: 0.94, verdict: 'authentic', confidence: 0.97, peak: 0.81 },
      { detector: 'holo_parallax', score: 0.91, verdict: 'authentic', confidence: 0.93, flow: 0.78 },
      { detector: 'substrate_bstar', score: 0.96, verdict: 'authentic', confidence: 0.95, b_delta: 0.2 },
      { detector: 'cielab_profile', score: 0.92, verdict: 'authentic', confidence: 0.94, deltaE: 1.4 },
      { detector: 'typography_ocr', score: 0.97, verdict: 'authentic', confidence: 0.98, ks: 0.04 },
      { detector: 'edge_glow', score: 0.99, verdict: 'authentic', confidence: 0.99, anomalies: 0 },
      { detector: 'reprint_mask', score: 0.95, verdict: 'authentic', confidence: 0.92, residual: 0.07 },
    ],
  },
  regions: [
    { kind: 'centering', position: 'whole_card', score: 0.92, severity: 'ok', reasons: [] },
    { kind: 'corner', position: 'top_left',     score: 0.88, severity: 'ok',    reasons: [] },
    { kind: 'corner', position: 'top_right',    score: 0.79, severity: 'minor', reasons: ['Minor whitening on tip · ~0.4mm'] },
    { kind: 'corner', position: 'bottom_left',  score: 0.91, severity: 'ok',    reasons: [] },
    { kind: 'corner', position: 'bottom_right', score: 0.82, severity: 'minor', reasons: ['Slight rounding · within PSA 9 tolerance'] },
    { kind: 'edge',   position: 'top',          score: 0.93, severity: 'ok',    reasons: [] },
    { kind: 'edge',   position: 'bottom',       score: 0.85, severity: 'minor', reasons: ['Edge whitening · 12% of length'] },
    { kind: 'edge',   position: 'left',         score: 0.95, severity: 'ok',    reasons: [] },
    { kind: 'edge',   position: 'right',        score: 0.96, severity: 'ok',    reasons: [] },
    { kind: 'surface', position: 'whole_card',  score: 0.96, severity: 'ok',    reasons: [] },
  ],
  population: {
    total_graded: 47,
    this_rank: 3,
    max_grade: 10.0,
    chronological_index: 12,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
function App() {
  const DEFAULTS = window.__CERT_TWEAK_DEFAULTS_RAW || { layout: 'split', showCardVision: 'canonical', scheme: 'psa', verdict: 'authentic' };
  const [t, setTweak] = useTweaks(DEFAULTS);

  // overrides for tweak demo
  const cert = useMemo(() => ({
    ...CERT,
    authenticity: { ...CERT.authenticity, verdict: t.verdict },
  }), [t.verdict]);

  return (
    <>
      {t.layout === 'split' && <SplitLayout cert={cert} t={t} />}
      {t.layout === 'stacked' && <StackedLayout cert={cert} t={t} />}
      {t.layout === 'magazine' && <MagazineLayout cert={cert} t={t} />}
      <CertTweaks t={t} setTweak={setTweak} />
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout 1 · SPLIT — primary. Card vision left, grade + auth right.
function SplitLayout({ cert, t }) {
  return (
    <main className="cert-page">
      <CertHeader cert={cert} />
      {/* Top band: card vision + grade hero side by side */}
      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 460px) minmax(0, 1fr)', gap: 36, marginTop: 28 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 22, position: 'sticky', top: 80, alignSelf: 'start' }}>
          <CardVisionPanel cert={cert} mode={t.showCardVision} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 24, minWidth: 0 }}>
          <GradeHero cert={cert} />
          <SubgradeGrid grade={cert.grades[0]} />
          <PopulationPanel pop={cert.population} />
        </div>
      </div>
      {/* Below: full-width sections */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 36, marginTop: 8 }}>
        <CenteringDetail grade={cert.grades[0]} />
        <DefectsPanel regions={cert.regions} />
      </div>
      <AuthenticityPanel auth={cert.authenticity} />
      <Provenance cert={cert} />
    </main>
  );
}

// Layout 2 · STACKED — single column, mobile-style. Hero up top.
function StackedLayout({ cert, t }) {
  return (
    <main className="cert-page" style={{ maxWidth: 760 }}>
      <CertHeader cert={cert} />
      <GradeHero cert={cert} bigVariant />
      <SubgradeGrid grade={cert.grades[0]} />
      <CardVisionPanel cert={cert} mode={t.showCardVision} />
      <PopulationPanel pop={cert.population} />
      <CenteringDetail grade={cert.grades[0]} />
      <DefectsPanel regions={cert.regions} />
      <AuthenticityPanel auth={cert.authenticity} />
      <Provenance cert={cert} />
    </main>
  );
}

// Layout 3 · MAGAZINE — wide editorial; large card with grade overlaid.
function MagazineLayout({ cert, t }) {
  const grade = cert.grades[0];
  return (
    <main className="cert-page" style={{ maxWidth: 1280 }}>
      <CertHeader cert={cert} />
      {/* Hero band */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 36,
        padding: '32px 0', borderBottom: '1px solid var(--line)',
        alignItems: 'center',
      }}>
        <div style={{ position: 'relative' }}>
          <CardArt cert={cert} large />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
            Final grade · {grade.scheme.toUpperCase()}
          </span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 14 }}>
            <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 200, lineHeight: 0.9, color: 'var(--ink-0)', letterSpacing: '-0.04em' }}>
              {grade.final.toFixed(1)}
            </span>
            <span className="mono" style={{ fontSize: 13, color: 'var(--ink-3)' }}>/ 10.0</span>
          </div>
          <ConfidenceBand grade={grade} variant="wide" />
          <SubgradeGrid grade={grade} compact />
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 36, marginTop: 32 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 28, minWidth: 0 }}>
          <CenteringDetail grade={grade} />
          <DefectsPanel regions={cert.regions} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 28, minWidth: 0 }}>
          <AuthenticityPanel auth={cert.authenticity} />
          <PopulationPanel pop={cert.population} />
        </div>
      </div>
      <Provenance cert={cert} />
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CERT HEADER — id strip + card name + share row
function CertHeader({ cert }) {
  const [copied, setCopied] = useState(false);
  const completed = new Date(cert.completed_at);
  const card = cert.identified_card;
  const grade = cert.grades[0];
  const verdict = cert.authenticity?.verdict || 'unverified';

  const copyId = () => {
    if (navigator.clipboard) {
      navigator.clipboard.writeText(cert.cert_id);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <header style={{
      display: 'flex', flexDirection: 'column', gap: 16,
      paddingBottom: 24, borderBottom: '1px solid var(--line)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 24, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
              Project Grader · Certificate
            </span>
            <VerdictPill verdict={verdict} small />
          </div>
          <h1 style={{ fontFamily: 'Instrument Serif, serif', fontSize: 44, fontWeight: 400, color: 'var(--ink-0)', margin: 0, letterSpacing: '-0.01em', lineHeight: 1.05 }}>
            {card.name}
          </h1>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-2)', letterSpacing: '0.04em', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <span>{card.set_code}</span>
            <span style={{ color: 'var(--ink-5)' }}>·</span>
            <span>#{card.card_number}</span>
            <span style={{ color: 'var(--ink-5)' }}>·</span>
            <span>{card.edition}</span>
            <span style={{ color: 'var(--ink-5)' }}>·</span>
            <span style={{ color: 'var(--ink-3)' }}>identification {(card.confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 10 }}>
          <button onClick={copyId} className="cert-id-pill" title="Copy cert ID">
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', letterSpacing: '0.04em' }}>
              {cert.cert_id}
            </span>
            <span className="mono" style={{ fontSize: 9, color: copied ? 'var(--lime)' : 'var(--ink-4)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              {copied ? 'COPIED' : 'COPY'}
            </span>
          </button>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>
            graded {completed.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: '2-digit' })} · {completed.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false })} UTC
          </span>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        <button className="pg-btn pg-btn-ghost"><Icon kind="share" /> Share</button>
        <button className="pg-btn pg-btn-ghost"><Icon kind="link" /> Copy URL</button>
        <button className="pg-btn pg-btn-ghost"><Icon kind="print" /> Print / PDF</button>
        <button className="pg-btn pg-btn-ghost"><Icon kind="image" /> Save image</button>
        <div style={{ flex: 1 }} />
        <button className="pg-btn pg-btn-ghost"><Icon kind="qr" /> QR</button>
        <button className="pg-btn pg-btn-ghost">Request manual review</button>
      </div>
      <style>{`
        .cert-id-pill {
          display: inline-flex; align-items: center; gap: 8px;
          padding: 5px 10px; border-radius: 4px;
          background: var(--bg-2); border: 1px solid var(--line);
          color: inherit; font-family: inherit; cursor: pointer;
          transition: border-color 120ms;
        }
        .cert-id-pill:hover { border-color: var(--ink-5); }
      `}</style>
    </header>
  );
}

function Icon({ kind }) {
  const paths = {
    share: 'M4 12v7h12v-7 M10 4l3-3 3 3 M13 1v11',
    link: 'M9 13a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1 M11 7a4 4 0 0 0-5.66 0l-3 3a4 4 0 1 0 5.66 5.66l1-1',
    print: 'M5 7V3h10v4 M5 14h10v6H5z M3 7h14v7H3z M16 10h.01',
    image: 'M3 3h14v14H3z M3 13l4-4 4 4 3-3 3 3 M11 7a1 1 0 1 1-2 0 1 1 0 0 1 2 0z',
    qr: 'M3 3h5v5H3z M12 3h5v5h-5z M3 12h5v5H3z M12 12h2v2h-2z M16 12h1v1 M12 16h1v1 M16 16h1v1',
  };
  return (
    <svg width="13" height="13" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
      <path d={paths[kind]} />
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CARD VISION — image with defect overlay, mode toggle (canonical / flash / overlay)
function CardVisionPanel({ cert, mode }) {
  const [activeMode, setActiveMode] = useState(mode);
  useEffect(() => { setActiveMode(mode); }, [mode]);
  const [opacity, setOpacity] = useState(0.45);

  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <SectionLabel hint="card-vision · ml/canonicals">Card vision</SectionLabel>

      {/* mode tabs */}
      <div style={{ display: 'flex', gap: 4, padding: 3, background: 'var(--bg-2)', border: '1px solid var(--line)', borderRadius: 5 }}>
        {[
          { k: 'canonical', l: 'Canonical' },
          { k: 'flash',     l: 'Flash' },
          { k: 'overlay',   l: 'Overlay' },
          { k: 'tilt',      l: 'Tilt' },
        ].map(m => (
          <button key={m.k} onClick={() => setActiveMode(m.k)} style={{
            flex: 1, padding: '6px 10px',
            background: activeMode === m.k ? 'var(--bg-3)' : 'transparent',
            border: 0, borderRadius: 3, cursor: 'pointer',
            color: activeMode === m.k ? 'var(--ink-0)' : 'var(--ink-3)',
            fontFamily: 'inherit', fontSize: 11, fontWeight: 500,
            transition: 'all 120ms',
          }}>{m.l}</button>
        ))}
      </div>

      {/* card frame with overlay */}
      <div style={{ position: 'relative' }}>
        <CardArt cert={cert} mode={activeMode} opacity={opacity} />
        <RegionOverlay regions={cert.regions} />
      </div>

      {activeMode === 'overlay' && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, paddingTop: 4 }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>opacity</span>
          <input type="range" min="0" max="1" step="0.01" value={opacity}
            onChange={e => setOpacity(parseFloat(e.target.value))}
            style={{ flex: 1, accentColor: 'var(--lime)' }} />
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', minWidth: 32, textAlign: 'right' }}>{Math.round(opacity * 100)}%</span>
        </div>
      )}

      <SeverityLegend />
    </section>
  );
}

function CardArt({ cert, mode = 'canonical', opacity = 0.45, large = false }) {
  // Stylized card art — Charizard themed.
  const isFlash = mode === 'flash';
  const isOverlay = mode === 'overlay';
  const isTilt = mode === 'tilt';

  return (
    <div style={{
      aspectRatio: '5/7',
      width: large ? '100%' : '100%',
      maxWidth: large ? 540 : '100%',
      margin: '0 auto',
      position: 'relative', borderRadius: 9,
      background: '#0a0a0a',
      border: '1px solid var(--line-2)',
      boxShadow: '0 30px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.02) inset',
      overflow: 'hidden',
      transform: isTilt ? 'perspective(1200px) rotateY(-8deg) rotateX(4deg)' : 'none',
      transition: 'transform 400ms cubic-bezier(0.16, 1, 0.3, 1)',
    }}>
      {/* Card body — stylized Charizard holo */}
      <div style={{
        position: 'absolute', inset: '4%',
        borderRadius: 4,
        background: `
          linear-gradient(140deg, #4d2410 0%, #7a3812 35%, #c0531a 60%, #f97316 80%, #fbbf24 100%)
        `,
        filter: isFlash ? 'brightness(1.4) contrast(0.95)' : 'none',
      }} />

      {/* Inner art frame */}
      <div style={{
        position: 'absolute', top: '10%', left: '8%', right: '8%', height: '56%',
        background: `
          radial-gradient(ellipse at 50% 35%, #fde68a, transparent 60%),
          linear-gradient(180deg, #f97316 0%, #ea580c 50%, #7a2e0c 100%)
        `,
        border: '4px solid #fbbf24',
        borderRadius: 2,
        boxShadow: 'inset 0 0 30px rgba(0,0,0,0.4)',
      }}>
        {/* dragon silhouette */}
        <svg viewBox="0 0 100 80" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.85 }}>
          <path d="M 20 60 Q 30 30 50 25 Q 70 20 78 35 Q 84 25 88 30 Q 82 38 80 45 Q 88 50 82 58 Q 70 62 60 58 Q 55 70 45 70 Q 35 72 30 65 Q 25 68 20 60 Z"
            fill="#7a2e0c" stroke="#1a0a04" strokeWidth="0.8" />
          <path d="M 38 50 L 45 38 L 52 48 Z" fill="#fbbf24" />
          <circle cx="72" cy="35" r="1.8" fill="#fbbf24" />
        </svg>
      </div>

      {/* HP bar */}
      <div style={{
        position: 'absolute', top: '4%', right: '8%',
        fontFamily: 'Geist Mono, monospace', fontSize: 11, fontWeight: 700,
        color: '#1a0a04', display: 'flex', alignItems: 'baseline', gap: 4,
      }}>
        <span style={{ fontSize: 6 }}>HP</span><span>120</span>
      </div>

      {/* Name plate */}
      <div style={{
        position: 'absolute', top: '4%', left: '8%',
        fontFamily: 'Instrument Serif, serif', fontSize: large ? 16 : 12, fontWeight: 600,
        color: '#1a0a04',
      }}>Charizard</div>

      {/* Bottom info */}
      <div style={{
        position: 'absolute', bottom: '4%', left: '8%', right: '8%',
        fontFamily: 'Geist Mono, monospace', fontSize: 7,
        color: '#1a0a04', display: 'flex', justifyContent: 'space-between',
      }}>
        <span>4/102 ★</span>
        <span>©1995, 96, 98 Nintendo</span>
      </div>

      {/* Holo shine */}
      {!isFlash && (
        <div style={{
          position: 'absolute', inset: '4%', borderRadius: 4,
          background: 'linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.18) 45%, transparent 60%)',
          mixBlendMode: 'screen',
          pointerEvents: 'none',
        }} />
      )}

      {/* Flash mode: warm flat exposure */}
      {isFlash && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'radial-gradient(ellipse at 50% 30%, rgba(255,255,255,0.28), transparent 65%)',
          pointerEvents: 'none',
        }} />
      )}

      {/* Overlay mode: scanned regions visualization */}
      {isOverlay && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'rgba(10,10,10,' + opacity + ')',
          pointerEvents: 'none',
        }} />
      )}
    </div>
  );
}

function RegionOverlay({ regions }) {
  const SEVERITY = {
    ok: { fill: 'rgba(190,242,100,0.15)', stroke: 'rgba(190,242,100,0.6)' },
    minor: { fill: 'rgba(251,191,36,0.18)', stroke: 'rgba(251,191,36,0.7)' },
    major: { fill: 'rgba(251,113,133,0.22)', stroke: 'rgba(251,113,133,0.8)' },
    unknown: { fill: 'rgba(82,82,91,0.18)', stroke: 'rgba(82,82,91,0.5)' },
  };
  const centering = regions.find(r => r.kind === 'centering');
  const corners = regions.filter(r => r.kind === 'corner');
  const edges = regions.filter(r => r.kind === 'edge');

  return (
    <div style={{ position: 'absolute', inset: 0, pointerEvents: 'none' }}>
      {/* Centering frame */}
      {centering && (
        <div style={{
          position: 'absolute', top: '14%', bottom: '14%', left: '10%', right: '10%',
          border: '1.5px dashed ' + SEVERITY[centering.severity].stroke,
          borderRadius: 4,
        }} />
      )}
      {/* Corners */}
      {corners.map(c => {
        const pos = {
          top_left:     { top: '3%',    left: '3%' },
          top_right:    { top: '3%',    right: '3%' },
          bottom_left:  { bottom: '3%', left: '3%' },
          bottom_right: { bottom: '3%', right: '3%' },
        }[c.position];
        const sev = SEVERITY[c.severity];
        return (
          <div key={c.position} style={{
            position: 'absolute', ...pos,
            width: 24, height: 24, borderRadius: 12,
            border: '1.5px solid ' + sev.stroke,
            background: sev.fill,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}>
            {c.severity !== 'ok' && (
              <span style={{ width: 5, height: 5, borderRadius: 3, background: sev.stroke }} />
            )}
          </div>
        );
      })}
      {/* Edges */}
      {edges.map(e => {
        const pos = {
          top:    { top: '3%',    left: '14%', right: '14%', height: 4 },
          bottom: { bottom: '3%', left: '14%', right: '14%', height: 4 },
          left:   { left: '3%',   top: '14%', bottom: '14%', width: 4 },
          right:  { right: '3%',  top: '14%', bottom: '14%', width: 4 },
        }[e.position];
        const sev = SEVERITY[e.severity];
        return (
          <div key={e.position} style={{
            position: 'absolute', ...pos,
            background: sev.fill,
            border: '1px solid ' + sev.stroke,
            borderRadius: 2,
          }} />
        );
      })}
    </div>
  );
}

function SeverityLegend() {
  const items = [
    { k: 'ok', l: 'OK', c: 'var(--lime)' },
    { k: 'minor', l: 'Minor', c: 'var(--amber)' },
    { k: 'major', l: 'Major', c: 'var(--rose)' },
    { k: 'unknown', l: 'Unknown', c: 'var(--ink-5)' },
  ];
  return (
    <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap', paddingTop: 4 }}>
      {items.map(it => (
        <span key={it.k} style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 8, height: 8, borderRadius: 4, background: it.c, opacity: 0.5, border: '1px solid ' + it.c }} />
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>{it.l}</span>
        </span>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GRADE HERO — big number + verdict
function GradeHero({ cert, bigVariant = false }) {
  const grade = cert.grades[0];
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <SectionLabel hint={`scheme · ${grade.scheme}`}>Grade</SectionLabel>
      <div style={{
        display: 'grid', gridTemplateColumns: 'auto minmax(0, 1fr)', gap: 44,
        alignItems: 'center',
        padding: '28px 32px',
        background: 'linear-gradient(135deg, rgba(190,242,100,0.04) 0%, transparent 60%), var(--bg-1)',
        border: '1px solid var(--line)', borderRadius: 8,
        position: 'relative', overflow: 'hidden',
      }}>
        {/* big number */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4, flexShrink: 0 }}>
          <span style={{
            fontFamily: 'Instrument Serif, serif',
            fontSize: bigVariant ? 180 : 128,
            lineHeight: 0.9,
            color: 'var(--ink-0)',
            letterSpacing: '-0.04em',
            textShadow: '0 0 80px rgba(190,242,100,0.15)',
          }}>{grade.final.toFixed(1)}</span>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
            of 10.0 · {grade.scheme}
          </span>
        </div>

        {/* right side */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18, minWidth: 0 }}>
          <ConfidenceBand grade={grade} />
        </div>

        {/* faint backing pattern */}
        <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.05, pointerEvents: 'none' }}>
          <defs>
            <pattern id="grade-grid" width="20" height="20" patternUnits="userSpaceOnUse">
              <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--lime)" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grade-grid)" />
        </svg>
      </div>
    </section>
  );
}

function ConfidenceBand({ grade, variant = 'default' }) {
  // confidence band: width = (1 - confidence) * 2 grade units, clamped 1..10
  const width = (1 - grade.confidence) * 2;
  const lower = Math.max(1, Math.min(10, grade.final - width));
  const upper = Math.max(1, Math.min(10, grade.final + width));
  const pct = (grade.confidence * 100).toFixed(0);

  // visual scale 1..10 → 0..100%
  const minPx = ((lower - 1) / 9) * 100;
  const maxPx = ((upper - 1) / 9) * 100;
  const finalPx = ((grade.final - 1) / 9) * 100;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8, minWidth: 0 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 12 }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
          Confidence band · {pct}%
        </span>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)' }}>
          likely <span style={{ color: 'var(--ink-0)', fontWeight: 600 }}>{lower.toFixed(1)}–{upper.toFixed(1)}</span>
        </span>
      </div>
      <div style={{
        height: variant === 'wide' ? 14 : 10,
        background: 'var(--bg-3)', borderRadius: 2,
        position: 'relative', overflow: 'visible',
      }}>
        {/* tick marks */}
        {[1,2,3,4,5,6,7,8,9,10].map(n => (
          <div key={n} style={{
            position: 'absolute', top: -2, bottom: -2,
            left: `${((n - 1) / 9) * 100}%`,
            width: 1,
            background: n === Math.floor(grade.final) ? 'var(--ink-3)' : 'var(--line)',
          }} />
        ))}
        {/* range bar */}
        <div style={{
          position: 'absolute', top: 0, bottom: 0,
          left: `${minPx}%`, width: `${maxPx - minPx}%`,
          background: 'rgba(190,242,100,0.2)',
          borderTop: '1px solid var(--lime)', borderBottom: '1px solid var(--lime)',
          borderRadius: 1,
        }} />
        {/* point marker */}
        <div style={{
          position: 'absolute', top: -3, bottom: -3,
          left: `calc(${finalPx}% - 1px)`, width: 2,
          background: 'var(--lime)',
          boxShadow: '0 0 8px var(--lime)',
        }} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)' }}>1.0</span>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)' }}>5.0</span>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)' }}>10.0</span>
      </div>
    </div>
  );
}

function CalibrationNote({ grade }) {
  return (
    <p className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', margin: 0, lineHeight: 1.6, letterSpacing: '0.02em' }}>
      Subgrades are min-merged into the final grade. Confidence is a calibrated bound,
      not a probability — width = (1 − conf) × 2 grade units, clamped to [1, 10].
    </p>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function SubgradeGrid({ grade, compact = false }) {
  const subs = [
    { k: 'centering', l: 'Centering', v: grade.centering, hint: 'geometric · 99% conf' },
    { k: 'corners',   l: 'Corners',   v: grade.corners,   hint: 'EfficientNet-V2' },
    { k: 'edges',     l: 'Edges',     v: grade.edges,     hint: 'MAD anomaly' },
    { k: 'surface',   l: 'Surface',   v: grade.surface,   hint: 'paired-flash + rosette' },
  ];
  return (
    <section>
      {!compact && <SectionLabel hint="per-criterion · independent">Subgrades</SectionLabel>}
      <div style={{
        display: 'grid', gridTemplateColumns: compact ? 'repeat(4, 1fr)' : 'repeat(4, 1fr)',
        gap: compact ? 6 : 10, marginTop: compact ? 0 : 12,
      }}>
        {subs.map(s => (
          <div key={s.k} style={{
            background: 'var(--bg-1)', border: '1px solid var(--line)',
            borderRadius: 6, padding: compact ? '12px 14px' : '16px 18px',
            display: 'flex', flexDirection: 'column', gap: 4,
            transition: 'border-color 120ms',
          }}>
            <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>
              {s.l}
            </span>
            <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: compact ? 32 : 40, lineHeight: 1, color: 'var(--ink-0)', letterSpacing: '-0.02em' }}>
              {s.v.toFixed(1)}
            </span>
            <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>
              {s.hint}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CENTERING — visual breakdown with L/R T/B percentages
function CenteringDetail({ grade }) {
  const front = { lr: grade.front_centering_lr, tb: grade.front_centering_tb };
  const back = { lr: grade.back_centering_lr, tb: grade.back_centering_tb };

  return (
    <section>
      <SectionLabel hint="L/R · T/B · per face">Centering breakdown</SectionLabel>
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 12,
      }}>
        <CenteringCard label="Front" data={front} />
        <CenteringCard label="Back" data={back} />
      </div>
      <p className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', margin: '10px 0 0', letterSpacing: '0.02em', lineHeight: 1.5 }}>
        Worst face dominates the centering subgrade. Perfect = 50/50; PSA 10 tolerance is ±5%.
      </p>
    </section>
  );
}

function CenteringCard({ label, data }) {
  return (
    <div style={{
      background: 'var(--bg-1)', border: '1px solid var(--line)',
      borderRadius: 6, padding: 14,
      display: 'flex', flexDirection: 'column', gap: 12,
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>{label}</span>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>
          L/R <span style={{ color: 'var(--ink-1)' }}>{data.lr.toFixed(1)}</span> · T/B <span style={{ color: 'var(--ink-1)' }}>{data.tb.toFixed(1)}</span>
        </span>
      </div>
      <CenteringSchematic lr={data.lr} tb={data.tb} />
    </div>
  );
}

function CenteringSchematic({ lr, tb }) {
  // lr: 50 = perfect; tb: 50 = perfect
  // Show inner card with L gap = lr%, R gap = (100-lr)% etc
  const lrLeft = lr;
  const tbTop = tb;
  return (
    <div style={{
      aspectRatio: '5/7',
      background: 'var(--bg-2)', border: '1px solid var(--line-2)',
      borderRadius: 4, position: 'relative', padding: '10%',
      maxHeight: 140, alignSelf: 'center', width: 100,
    }}>
      {/* inner card */}
      <div style={{
        position: 'absolute', inset: 0,
        // outer gap proportional to lrLeft % vs 100-lrLeft and tbTop vs 100-tbTop
        // canonical card occupies 80% inside; offset shifted by (lr - 50) and (tb - 50)
      }}>
        <div style={{
          position: 'absolute',
          left: ((100 - 80) * (lrLeft / 100)) + '%',
          right: ((100 - 80) * ((100 - lrLeft) / 100)) + '%',
          top: ((100 - 80) * (tbTop / 100)) + '%',
          bottom: ((100 - 80) * ((100 - tbTop) / 100)) + '%',
          background: 'rgba(190,242,100,0.1)',
          border: '1px solid var(--lime)',
          borderRadius: 2,
        }} />
        {/* center crosshair */}
        <div style={{ position: 'absolute', top: '50%', left: '20%', right: '20%', height: 1, background: 'var(--ink-5)', opacity: 0.4 }} />
        <div style={{ position: 'absolute', left: '50%', top: '20%', bottom: '20%', width: 1, background: 'var(--ink-5)', opacity: 0.4 }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFECTS — DINGS-style itemized list
function DefectsPanel({ regions }) {
  const flagged = regions.filter(r => r.reasons.length > 0);
  const groups = { centering: [], corner: [], edge: [], surface: [] };
  flagged.forEach(r => groups[r.kind].push(r));
  const kinds = [
    { k: 'centering', l: 'Centering' },
    { k: 'corner',    l: 'Corners' },
    { k: 'edge',      l: 'Edges' },
    { k: 'surface',   l: 'Surface' },
  ];
  const positionLabel = {
    top_left: 'Top-left',  top_right: 'Top-right',
    bottom_left: 'Bottom-left', bottom_right: 'Bottom-right',
    top: 'Top edge', bottom: 'Bottom edge', left: 'Left edge', right: 'Right edge',
    whole_card: '',
  };
  const dot = {
    ok: 'var(--lime)', minor: 'var(--amber)', major: 'var(--rose)', unknown: 'var(--ink-5)',
  };

  return (
    <section>
      <SectionLabel hint="DINGS · plain-english rationale">Defects flagged</SectionLabel>
      {flagged.length === 0 ? (
        <p style={{ fontSize: 13, color: 'var(--ink-3)', margin: '12px 0 0', fontStyle: 'italic' }}>No defects flagged. The card grades clean across all evaluated regions.</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, marginTop: 14 }}>
          {kinds.map(({ k, l }) => {
            if (groups[k].length === 0) return null;
            return (
              <div key={k}>
                <h4 className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase', margin: '0 0 8px', fontWeight: 500 }}>{l}</h4>
                <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 7 }}>
                  {groups[k].flatMap(r => r.reasons.map((reason, i) => (
                    <li key={`${r.position}-${i}`} style={{ display: 'flex', gap: 10, alignItems: 'baseline', padding: '8px 12px', background: 'var(--bg-1)', borderLeft: '2px solid ' + dot[r.severity], borderRadius: '0 4px 4px 0' }}>
                      <span style={{ width: 6, height: 6, borderRadius: 3, background: dot[r.severity], flexShrink: 0, transform: 'translateY(-1px)' }} />
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                        <span style={{ fontSize: 13, color: 'var(--ink-1)' }}>{reason}</span>
                        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>
                          {positionLabel[r.position] || r.position} · severity {r.severity} · score {r.score?.toFixed(2) ?? '–'}
                        </span>
                      </div>
                    </li>
                  )))}
                </ul>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTHENTICITY — verdict + detector ensemble
function AuthenticityPanel({ auth }) {
  const [expanded, setExpanded] = useState(null);
  return (
    <section>
      <SectionLabel hint={`ensemble · ${auth.detectors.length} detectors`}>Authenticity</SectionLabel>
      <div style={{ marginTop: 12, padding: '18px 20px', background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 8, display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
          <VerdictPill verdict={auth.verdict} />
          <div style={{ textAlign: 'right' }}>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.16em', textTransform: 'uppercase' }}>
              ensemble confidence
            </span>
            <div style={{ fontFamily: 'Instrument Serif, serif', fontSize: 28, color: 'var(--ink-0)', lineHeight: 1, marginTop: 4 }}>
              {(auth.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {auth.reasons.length > 0 && (
          <ul style={{ listStyle: 'none', margin: 0, padding: 0, display: 'flex', flexDirection: 'column', gap: 5 }}>
            {auth.reasons.map((r, i) => (
              <li key={i} style={{ fontSize: 12, color: 'var(--ink-2)', display: 'flex', gap: 8, lineHeight: 1.5 }}>
                <span style={{ color: 'var(--lime)', flexShrink: 0 }}>✓</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        )}

        <div style={{ height: 1, background: 'var(--line-2)' }} />

        <div>
          <h4 className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase', margin: '0 0 10px', fontWeight: 500 }}>
            Detector breakdown · {auth.detectors.length} signals
          </h4>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {auth.detectors.map(d => (
              <DetectorRow key={d.detector} d={d}
                expanded={expanded === d.detector}
                onToggle={() => setExpanded(expanded === d.detector ? null : d.detector)} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

const VERDICT_META = {
  authentic: { label: 'Authentic', tone: 'lime', dot: 'var(--lime)' },
  suspicious: { label: 'Suspicious — manual review', tone: 'amber', dot: 'var(--amber)' },
  likely_counterfeit: { label: 'Likely counterfeit', tone: 'rose', dot: 'var(--rose)' },
  unverified: { label: 'Unverified — insufficient signal', tone: 'ink', dot: 'var(--ink-4)' },
};

function VerdictPill({ verdict, small = false }) {
  const meta = VERDICT_META[verdict] || VERDICT_META.unverified;
  const colorMap = {
    lime: { bg: 'rgba(190,242,100,0.06)', border: 'rgba(190,242,100,0.4)', fg: 'var(--lime)' },
    amber: { bg: 'rgba(251,191,36,0.06)', border: 'rgba(251,191,36,0.4)', fg: 'var(--amber)' },
    rose: { bg: 'rgba(251,113,133,0.06)', border: 'rgba(251,113,133,0.4)', fg: 'var(--rose)' },
    ink: { bg: 'var(--bg-2)', border: 'var(--line)', fg: 'var(--ink-3)' },
  };
  const c = colorMap[meta.tone];
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: small ? 5 : 8,
      padding: small ? '3px 8px' : '7px 14px',
      background: c.bg,
      border: '1px solid ' + c.border,
      borderRadius: 99,
      color: c.fg,
      fontWeight: 500,
      fontSize: small ? 10 : 13,
    }}>
      <span style={{
        width: small ? 4 : 6, height: small ? 4 : 6, borderRadius: 4,
        background: meta.dot, boxShadow: '0 0 6px ' + meta.dot,
        animation: verdict === 'authentic' ? 'none' : verdict === 'unverified' ? 'none' : 'pulse 1.6s ease-in-out infinite',
      }} />
      {meta.label}
      <style>{`@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }`}</style>
    </span>
  );
}

function DetectorRow({ d, expanded, onToggle }) {
  const meta = VERDICT_META[d.verdict] || VERDICT_META.unverified;
  const tone = { lime: 'var(--lime)', amber: 'var(--amber)', rose: 'var(--rose)', ink: 'var(--ink-3)' }[meta.tone];
  const detail = useMemo(() => detectorDetails(d), [d]);
  return (
    <div style={{
      background: 'var(--bg-2)', border: '1px solid var(--line-2)',
      borderRadius: 4, overflow: 'hidden',
      transition: 'background 120ms',
    }}>
      <button onClick={onToggle} style={{
        width: '100%', padding: '10px 14px',
        background: 'transparent', border: 0, cursor: 'pointer',
        display: 'grid', gridTemplateColumns: '180px 1fr 70px 60px 14px',
        gap: 12, alignItems: 'center',
        color: 'inherit', fontFamily: 'inherit', textAlign: 'left',
      }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', letterSpacing: '0.04em' }}>
          {d.detector}
        </span>
        <div style={{ height: 4, background: 'var(--bg-3)', borderRadius: 1, overflow: 'hidden' }}>
          <div style={{ width: (d.score * 100) + '%', height: '100%', background: tone }} />
        </div>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
          {(d.score * 100).toFixed(0)}%
        </span>
        <span className="mono" style={{ fontSize: 9, color: tone, textTransform: 'uppercase', letterSpacing: '0.1em', textAlign: 'right' }}>
          {meta.tone === 'ink' ? 'unv' : meta.tone === 'lime' ? 'auth' : meta.tone === 'amber' ? 'susp' : 'cnft'}
        </span>
        <span style={{ color: 'var(--ink-4)', fontSize: 10, transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 120ms' }}>›</span>
      </button>
      {expanded && (
        <div style={{ padding: '12px 14px', borderTop: '1px solid var(--line-2)', display: 'flex', flexDirection: 'column', gap: 8 }}>
          <p style={{ margin: 0, fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.5 }}>{detail.desc}</p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6 }}>
            {detail.metrics.map(([l, v]) => (
              <div key={l} style={{ background: 'var(--bg-1)', borderRadius: 3, padding: '6px 8px' }}>
                <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{l}</div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', fontVariantNumeric: 'tabular-nums', marginTop: 2 }}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function detectorDetails(d) {
  const map = {
    rosette_fft: {
      desc: 'FFT analysis of print rosette pattern. Counterfeits typically show degraded peak strength below 0.6 due to inkjet vs offset.',
      metrics: [['peak strength', d.peak], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['ref µ ± σ', '0.78 ± 0.05']],
    },
    holo_parallax: {
      desc: 'Farnebäck optical flow on the tilt-shot pair. Authentic foil substrates show characteristic parallax flow magnitude.',
      metrics: [['flow mag', d.flow], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['ref µ', '0.74']],
    },
    substrate_bstar: {
      desc: 'CIELAB b* channel under controlled flash. WOTC era stock has b* between -2.0 and -2.5; reprints drift positive.',
      metrics: [['b* delta', d.b_delta], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['ref window', '-2.0 to -2.5']],
    },
    cielab_profile: {
      desc: 'Full CIELAB color profile compared to manufacturer reference via mean ΔE76.',
      metrics: [['ΔE76', d.deltaE], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['threshold', '< 2.5']],
    },
    typography_ocr: {
      desc: 'RapidOCR stroke-width distribution. Reprints show significant Kolmogorov-Smirnov drift from reference.',
      metrics: [['KS stat', d.ks], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['threshold', '< 0.10']],
    },
    edge_glow: {
      desc: 'Edge detection for inkjet-printed reprint glow halo. Authentic offset shows sharp 0.0-0.1 glow.',
      metrics: [['anomalies', d.anomalies], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['threshold', '≤ 1']],
    },
    reprint_mask: {
      desc: 'Trained reprint-mask CNN. Detects fake-card characteristics: dot patterns, dye saturation, paper grain.',
      metrics: [['residual', d.residual], ['conf', (d.confidence * 100).toFixed(0) + '%'], ['threshold', '< 0.20']],
    },
  };
  return map[d.detector] || { desc: '', metrics: [] };
}

// ─────────────────────────────────────────────────────────────────────────────
// POPULATION — TAG-style scarcity counter
function PopulationPanel({ pop }) {
  const stats = [
    { l: 'Population', v: pop.total_graded, sub: 'graded by Project Grader' },
    { l: 'Rank', v: '#' + pop.this_rank, sub: pop.max_grade ? `top score ${pop.max_grade.toFixed(1)}` : 'highest scoring' },
    { l: 'Order', v: '#' + pop.chronological_index, sub: 'graded chronologically' },
  ];
  return (
    <section>
      <SectionLabel hint="among identified variants">Population</SectionLabel>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, marginTop: 12 }}>
        {stats.map(s => (
          <div key={s.l} style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 5, padding: '12px 14px' }}>
            <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{s.l}</div>
            <div style={{ fontFamily: 'Instrument Serif, serif', fontSize: 28, color: 'var(--ink-0)', lineHeight: 1, marginTop: 4, letterSpacing: '-0.02em' }}>{s.v}</div>
            <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', marginTop: 4, letterSpacing: '0.02em' }}>{s.sub}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Provenance({ cert }) {
  const versions = cert.authenticity?.model_versions || {};
  return (
    <footer style={{
      marginTop: 40, paddingTop: 24,
      borderTop: '1px solid var(--line-2)',
      display: 'flex', flexDirection: 'column', gap: 16,
    }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 6 }}>
        {Object.entries(versions).map(([k, v]) => (
          <div key={k} className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.02em', display: 'flex', justifyContent: 'space-between', padding: '5px 8px', background: 'var(--bg-1)', borderRadius: 3 }}>
            <span style={{ color: 'var(--ink-4)' }}>{k}</span>
            <span>{v}</span>
          </div>
        ))}
      </div>
      <p style={{ fontSize: 11, color: 'var(--ink-4)', margin: 0, lineHeight: 1.6, maxWidth: 720 }}>
        This certificate is generated by Project Grader's automated grading + counterfeit-detection pipeline.
        Not affiliated with PSA, BGS, SGC, or CGC. Grades are independent assessments and may differ from
        third-party graders. Disagreements are welcome — request a manual review above.
      </p>
    </footer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function SectionLabel({ children, hint }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', borderBottom: '1px solid var(--line-2)', paddingBottom: 8 }}>
      <h2 className="mono" style={{
        fontSize: 11, color: 'var(--ink-2)', letterSpacing: '0.18em', textTransform: 'uppercase',
        margin: 0, fontWeight: 500,
      }}>{children}</h2>
      {hint && <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>{hint}</span>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CertTweaks({ t, setTweak }) {
  return (
    <TweaksPanel>
      <TweakSection label="Layout" />
      <TweakRadio label="Variant"
        value={t.layout}
        options={['split','stacked','magazine']}
        onChange={v => setTweak('layout', v)} />
      <TweakSection label="Card vision" />
      <TweakRadio label="Default mode"
        value={t.showCardVision}
        options={['canonical','flash','overlay']}
        onChange={v => setTweak('showCardVision', v)} />
      <TweakSection label="Demo state" />
      <TweakSelect label="Verdict"
        value={t.verdict}
        options={['authentic','suspicious','likely_counterfeit','unverified']}
        onChange={v => setTweak('verdict', v)} />
    </TweaksPanel>
  );
}

// inline page styles
const styleTag = document.createElement('style');
styleTag.textContent = `
  .cert-page {
    max-width: 1180px;
    margin: 32px auto;
    padding: 0 32px 80px;
    display: flex; flex-direction: column; gap: 28px;
  }
  .cert-page h1 { text-wrap: balance; }
`;
document.head.appendChild(styleTag);

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
