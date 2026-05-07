// capture.jsx — 8-shot capture wizard
const { useState, useMemo, useEffect, useRef } = React;

// ─────────────────────────────────────────────────────────────────────────────
// Shot definitions — ShotKind enum from apps/api/grader/db/models.py
const SHOTS = [
  { k: 'front_full',       l: 'Front',         req: true,  pose: 'flat',   light: 'ambient', desc: 'Whole card, face up. No flash. Fill the guide rectangle edge-to-edge.' },
  { k: 'back_full',        l: 'Back',          req: true,  pose: 'flat',   light: 'ambient', desc: 'Whole card, face down. Same lighting as front.' },
  { k: 'front_full_flash', l: 'Front · flash', req: false, pose: 'flat',   light: 'flash',   desc: 'Same framing as front. Flash on. Used for substrate b* and glare-pair detectors.' },
  { k: 'tilt_30',          l: 'Tilt 30°',      req: false, pose: 'tilt',   light: 'ambient', desc: 'Front of card tilted ~30°. Used for holo parallax.' },
  { k: 'corner_tl',        l: 'Corner TL',     req: false, pose: 'corner-tl', light: 'macro',   desc: 'Macro shot of top-left corner. Fill the inner ring.' },
  { k: 'corner_tr',        l: 'Corner TR',     req: false, pose: 'corner-tr', light: 'macro',   desc: 'Macro shot of top-right corner.' },
  { k: 'corner_bl',        l: 'Corner BL',     req: false, pose: 'corner-bl', light: 'macro',   desc: 'Macro shot of bottom-left corner.' },
  { k: 'corner_br',        l: 'Corner BR',     req: false, pose: 'corner-br', light: 'macro',   desc: 'Macro shot of bottom-right corner.' },
];

// Mock captured-shot history per index — shape mirrors SubmissionShot rows
const MOCK_SHOT_RESULTS = [
  { kind: 'front_full',       passed: true,  blur: 0.91, glare: 0.04, perspective: 0.6,  width: 4032, height: 3024, ts: '14:06:12', kept: true },
  { kind: 'back_full',        passed: true,  blur: 0.88, glare: 0.06, perspective: 1.1,  width: 4032, height: 3024, ts: '14:06:31', kept: true },
  { kind: 'front_full_flash', passed: false, blur: 0.79, glare: 0.34, perspective: 0.4,  reasons: ['glare_band detected'], retake: true, ts: '14:06:51' },
  { kind: 'front_full_flash', passed: true,  blur: 0.84, glare: 0.09, perspective: 0.7,  width: 4032, height: 3024, ts: '14:07:09', kept: true },
];

// ─────────────────────────────────────────────────────────────────────────────
function App() {
  const DEFAULTS = window.__CAPTURE_TWEAK_DEFAULTS_RAW || { currentShot: 3, liveQuality: 'good', showShotList: true, deviceFraming: 'desktop' };
  const [t, setTweak] = useTweaks(DEFAULTS);
  const [framing, setFraming] = useState('flat');

  const shotsState = useMemo(() => SHOTS.map((s, i) => {
    let status;
    if (i < t.currentShot) status = 'captured';
    else if (i === t.currentShot) status = 'active';
    else status = 'pending';
    return { ...s, idx: i, status };
  }), [t.currentShot]);

  const activeShot = shotsState.find(s => s.status === 'active') || shotsState[0];
  const capturedCount = shotsState.filter(s => s.status === 'captured').length;
  const requiredCaptured = shotsState.filter(s => s.req && s.status === 'captured').length;
  const requiredTotal = SHOTS.filter(s => s.req).length;
  const canSubmit = requiredCaptured >= requiredTotal;

  return (
    <main className="cap-main">
      <CaptureHeader
        t={t} setTweak={setTweak}
        capturedCount={capturedCount}
        requiredCaptured={requiredCaptured}
        requiredTotal={requiredTotal}
        canSubmit={canSubmit}
      />

      <div className="cap-grid">
        {/* Left: shot list rail */}
        {t.showShotList && (
          <ShotRail shotsState={shotsState} setTweak={setTweak} />
        )}

        {/* Center: viewfinder */}
        <Viewfinder
          shot={activeShot}
          quality={t.liveQuality}
          deviceFraming={t.deviceFraming}
          onCapture={() => setTweak('currentShot', Math.min(SHOTS.length, t.currentShot + 1))}
          onSkip={() => setTweak('currentShot', Math.min(SHOTS.length, t.currentShot + 1))}
        />

        {/* Right: live coach + history */}
        <CoachPanel shot={activeShot} quality={t.liveQuality} />
      </div>

      <CaptureTweaks t={t} setTweak={setTweak} shotCount={SHOTS.length} />

      <style>{`
        .cap-main {
          max-width: 1380px; margin: 24px auto; padding: 0 24px 80px;
          display: flex; flex-direction: column; gap: 20px;
        }
        .cap-grid {
          display: grid;
          grid-template-columns: ${t.showShotList ? '300px' : ''} minmax(0, 1fr) 320px;
          gap: 18px;
        }
        @media (max-width: 1280px) {
          .cap-grid { grid-template-columns: 1fr; }
        }
      `}</style>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CaptureHeader({ t, setTweak, capturedCount, requiredCaptured, requiredTotal, canSubmit }) {
  return (
    <header style={{
      padding: '16px 20px', background: 'var(--bg-1)',
      border: '1px solid var(--line)', borderRadius: 8,
      display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 24, flexWrap: 'wrap',
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
          New submission · sub_a91f3c (capturing)
        </span>
        <h1 style={{ fontFamily: 'Instrument Serif, serif', fontSize: 28, fontWeight: 400, color: 'var(--ink-0)', margin: 0, lineHeight: 1.1, letterSpacing: '-0.01em' }}>
          Capture wizard
        </h1>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>
          <span style={{ color: requiredCaptured >= requiredTotal ? 'var(--lime)' : 'var(--amber)' }}>
            {requiredCaptured}/{requiredTotal} required
          </span>
          <span style={{ margin: '0 8px', color: 'var(--ink-5)' }}>·</span>
          <span style={{ color: 'var(--ink-2)' }}>{capturedCount} of 8 total</span>
          <span style={{ margin: '0 8px', color: 'var(--ink-5)' }}>·</span>
          <span style={{ color: 'var(--ink-3)' }}>2 min average · 1 retake so far</span>
        </span>
      </div>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <button className="pg-btn pg-btn-ghost">Save & resume later</button>
        <button className="pg-btn pg-btn-ghost">Reset wizard</button>
        {canSubmit ? (
          <a href="pipeline.html" className="pg-btn pg-btn-primary">Submit for grading →</a>
        ) : (
          <button className="pg-btn pg-btn-primary" style={{ opacity: 0.4, cursor: 'not-allowed' }} disabled>
            {requiredTotal - requiredCaptured} required remaining
          </button>
        )}
      </div>
    </header>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function ShotRail({ shotsState, setTweak }) {
  return (
    <aside style={{
      background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 8,
      padding: 14, display: 'flex', flexDirection: 'column', gap: 12,
      maxHeight: 'calc(100vh - 220px)', overflowY: 'auto',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', borderBottom: '1px solid var(--line-2)', paddingBottom: 8 }}>
        <h2 className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', letterSpacing: '0.18em', textTransform: 'uppercase', margin: 0, fontWeight: 500 }}>Shot list</h2>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)' }}>2 req · 6 opt</span>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {shotsState.map(s => (
          <ShotRailItem key={s.k} shot={s} onClick={() => setTweak('currentShot', s.idx)} />
        ))}
      </div>

      <div style={{
        marginTop: 4, padding: 10, background: 'var(--bg-2)', borderRadius: 4,
        borderLeft: '2px solid var(--lime)',
      }}>
        <div className="mono" style={{ fontSize: 9, color: 'var(--lime)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Optional shots boost</div>
        <div style={{ fontSize: 11, color: 'var(--ink-2)', marginTop: 6, lineHeight: 1.45 }}>
          <span style={{ color: 'var(--ink-1)' }}>+ Tilt 30°</span> unlocks holo parallax detector. <span style={{ color: 'var(--ink-1)' }}>+ Flash</span> unlocks substrate b*. <span style={{ color: 'var(--ink-1)' }}>+ Corners</span> sharpens corner subgrade.
        </div>
      </div>
    </aside>
  );
}

function ShotRailItem({ shot, onClick }) {
  const tone = shot.status === 'captured' ? 'var(--lime)' :
               shot.status === 'active' ? 'var(--lime)' : 'var(--ink-5)';

  return (
    <button onClick={onClick} style={{
      all: 'unset', cursor: 'pointer',
      display: 'grid', gridTemplateColumns: '24px 1fr auto',
      gap: 10, padding: '8px 10px', borderRadius: 4,
      background: shot.status === 'active' ? 'var(--bg-2)' : 'transparent',
      border: '1px solid ' + (shot.status === 'active' ? 'var(--line)' : 'transparent'),
      transition: 'all 150ms',
    }}>
      <ShotGlyph status={shot.status} tone={tone} />
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <span style={{ fontSize: 12, fontWeight: 500, color: shot.status === 'pending' ? 'var(--ink-3)' : 'var(--ink-0)' }}>{shot.l}</span>
          {shot.req && <span className="mono" style={{ fontSize: 8, color: 'var(--amber)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>req</span>}
          {!shot.req && <span className="mono" style={{ fontSize: 8, color: 'var(--ink-4)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>opt</span>}
        </div>
        <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', marginTop: 2 }}>{shot.k}</div>
      </div>
      <span className="mono" style={{ fontSize: 9, color: tone, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
        {shot.status === 'captured' ? '✓' : shot.status === 'active' ? '●' : '·'}
      </span>
    </button>
  );
}

function ShotGlyph({ status, tone }) {
  if (status === 'captured') {
    return (
      <div style={{ width: 22, height: 22, borderRadius: 11, background: 'rgba(190,242,100,0.12)', border: '1px solid rgba(190,242,100,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
          <path d="M2 6.5L5 9L10 3" stroke="var(--lime)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }
  if (status === 'active') {
    return (
      <div style={{ width: 22, height: 22, borderRadius: 11, background: 'rgba(190,242,100,0.08)', border: '1px solid rgba(190,242,100,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>
        <span style={{ width: 6, height: 6, borderRadius: 4, background: 'var(--lime)', boxShadow: '0 0 6px var(--lime)' }} />
      </div>
    );
  }
  return <div style={{ width: 22, height: 22, borderRadius: 11, border: '1px dashed var(--ink-5)' }} />;
}

// ─────────────────────────────────────────────────────────────────────────────
function Viewfinder({ shot, quality, deviceFraming, onCapture, onSkip }) {
  return (
    <section style={{
      background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 8,
      padding: 18, display: 'flex', flexDirection: 'column', gap: 14,
      minHeight: 720,
    }}>
      {/* header band */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16 }}>
        <div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.22em', textTransform: 'uppercase', display: 'flex', gap: 8, alignItems: 'center' }}>
            <span>shot {shot.idx + 1} / 8</span>
            <span style={{ color: 'var(--ink-5)' }}>·</span>
            <span className="mono" style={{ color: 'var(--lime)' }}>{shot.k}</span>
          </div>
          <h2 style={{ fontFamily: 'Instrument Serif, serif', fontSize: 26, fontWeight: 400, color: 'var(--ink-0)', margin: '4px 0 0', letterSpacing: '-0.01em' }}>
            {shot.l}
          </h2>
          <p style={{ fontSize: 13, color: 'var(--ink-2)', margin: '6px 0 0', maxWidth: 520, lineHeight: 1.5 }}>
            {shot.desc}
          </p>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'flex-end', minWidth: 180 }}>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Lighting</span>
          <div style={{ display: 'flex', gap: 4 }}>
            {['ambient', 'flash', 'macro'].map(l => (
              <span key={l} className="mono" style={{
                fontSize: 9, padding: '3px 7px', borderRadius: 99,
                background: shot.light === l ? 'rgba(190,242,100,0.12)' : 'var(--bg-2)',
                border: '1px solid ' + (shot.light === l ? 'rgba(190,242,100,0.45)' : 'var(--line-2)'),
                color: shot.light === l ? 'var(--lime)' : 'var(--ink-3)',
                letterSpacing: '0.08em', textTransform: 'uppercase',
              }}>{l}</span>
            ))}
          </div>
        </div>
      </div>

      {/* viewfinder canvas */}
      <ViewfinderCanvas shot={shot} quality={quality} deviceFraming={deviceFraming} />

      {/* live quality readout */}
      <QualityBand quality={quality} />

      {/* capture controls */}
      <div style={{ display: 'flex', gap: 10, alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Device</span>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', padding: '3px 8px', background: 'var(--bg-2)', borderRadius: 3, border: '1px solid var(--line-2)' }}>
            iPhone 15 Pro · main camera · 4032 × 3024
          </span>
        </div>

        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {!shot.req && <button className="pg-btn pg-btn-ghost" onClick={onSkip}>Skip optional</button>}
          <button className="pg-btn pg-btn-ghost">Show last capture</button>
          <CaptureButton quality={quality} onCapture={onCapture} />
        </div>
      </div>
    </section>
  );
}

function ViewfinderCanvas({ shot, quality, deviceFraming }) {
  // Rendered representation: a "live camera" stage with pose-specific guides
  // and a faux card silhouette at the right pose.

  const isOK = quality === 'good';
  const guideColor = isOK ? 'var(--lime)' : quality === 'warn' ? 'var(--amber)' : 'var(--rose)';

  return (
    <div style={{
      position: 'relative', aspectRatio: '4 / 3',
      background: 'radial-gradient(ellipse at center, #1a1a1a 0%, #0d0d0d 70%, #000 100%)',
      border: '1px solid var(--line)', borderRadius: 6, overflow: 'hidden',
    }}>
      {/* dotted scan grid */}
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.3 }}>
        <defs>
          <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r="0.6" fill="var(--ink-4)" />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#dots)" />
      </svg>

      {/* live noise / film grain feel */}
      <div style={{
        position: 'absolute', inset: 0, mixBlendMode: 'overlay', opacity: 0.4,
        background: 'repeating-linear-gradient(0deg, rgba(255,255,255,0.02), rgba(0,0,0,0.02) 2px)',
      }} />

      {/* Pose-specific card silhouette + guide */}
      <PoseGuide pose={shot.pose} guideColor={guideColor} />

      {/* corner brackets */}
      {[[8,8,1,1],[92,8,-1,1],[8,92,1,-1],[92,92,-1,-1]].map(([x,y,sx,sy], i) => (
        <svg key={i} style={{ position: 'absolute', left: x+'%', top: y+'%', width: 24, height: 24, transform: 'translate(-50%,-50%)' }}>
          <path d={`M ${sx<0?24:0} 12 L ${sx<0?12:12} 12 L 12 ${sy<0?24:0}`} stroke="var(--ink-2)" strokeWidth="1.5" fill="none" />
        </svg>
      ))}

      {/* live exposure / iso readout (top-left) */}
      <div style={{ position: 'absolute', top: 14, left: 14, display: 'flex', flexDirection: 'column', gap: 4 }}>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Exposure</span>
        <span className="mono" style={{ fontSize: 12, color: 'var(--ink-1)' }}>1/120 · f/1.78 · ISO 100</span>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 6, letterSpacing: '0.12em' }}>WB · 5200K (locked)</span>
      </div>

      {/* live status pill (top-right) */}
      <div style={{ position: 'absolute', top: 14, right: 14, display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
        <span style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          padding: '4px 10px', borderRadius: 99,
          background: 'rgba(190,242,100,0.1)', border: '1px solid rgba(190,242,100,0.4)',
          color: 'var(--lime)', fontSize: 10, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.12em',
        }}>
          <span style={{ width: 5, height: 5, borderRadius: 3, background: 'var(--lime)', boxShadow: '0 0 6px var(--lime)' }} />
          live · 30 fps
        </span>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.1em' }}>card detector locked</span>
      </div>

      {/* bottom centered hint */}
      <div style={{ position: 'absolute', bottom: 16, left: '50%', transform: 'translateX(-50%)', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
        <span className="mono" style={{ fontSize: 10, color: guideColor, letterSpacing: '0.18em', textTransform: 'uppercase', padding: '4px 10px', background: 'rgba(0,0,0,0.55)', borderRadius: 3, border: '1px solid ' + guideColor + '55' }}>
          {isOK ? 'Hold steady · ready to capture' : quality === 'warn' ? 'Adjust framing · 0.4° tilt' : 'Glare detected · diffuse the light'}
        </span>
      </div>
    </div>
  );
}

function PoseGuide({ pose, guideColor }) {
  if (pose === 'flat') {
    // Whole-card rectangle, centered with crosshair and centering ticks
    return (
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet">
        {/* card rect */}
        <rect x="130" y="65" width="140" height="190" rx="6"
          fill="rgba(190,242,100,0.04)"
          stroke={guideColor} strokeWidth="1.4" strokeDasharray="6 4" />
        {/* card silhouette inside */}
        <rect x="138" y="73" width="124" height="174" rx="4"
          fill="url(#cardSilhouette)" stroke="var(--ink-3)" strokeWidth="0.5" opacity="0.6" />
        <defs>
          <linearGradient id="cardSilhouette" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3a1a08" />
            <stop offset="50%" stopColor="#7a3a14" />
            <stop offset="100%" stopColor="#f97316" stopOpacity="0.6" />
          </linearGradient>
        </defs>
        {/* center cross */}
        <line x1="200" y1="155" x2="200" y2="165" stroke={guideColor} strokeWidth="1" />
        <line x1="195" y1="160" x2="205" y2="160" stroke={guideColor} strokeWidth="1" />
        {/* centering ticks (L/R) */}
        <line x1="130" y1="160" x2="125" y2="160" stroke={guideColor} strokeWidth="1.5" />
        <line x1="270" y1="160" x2="275" y2="160" stroke={guideColor} strokeWidth="1.5" />
        <line x1="200" y1="65" x2="200" y2="60" stroke={guideColor} strokeWidth="1.5" />
        <line x1="200" y1="255" x2="200" y2="260" stroke={guideColor} strokeWidth="1.5" />
        <text x="200" y="50" textAnchor="middle" fontFamily="Geist Mono" fontSize="7" fill={guideColor} letterSpacing="0.18em">FILL THE GUIDE</text>
        <text x="100" y="160" textAnchor="middle" fontFamily="Geist Mono" fontSize="6" fill="var(--ink-4)" letterSpacing="0.1em">L</text>
        <text x="300" y="160" textAnchor="middle" fontFamily="Geist Mono" fontSize="6" fill="var(--ink-4)" letterSpacing="0.1em">R</text>
      </svg>
    );
  }
  if (pose === 'tilt') {
    // perspective-warped card showing 30° tilt
    return (
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet">
        {/* tilted card shape */}
        <polygon points="148,95 260,85 270,225 138,235"
          fill="rgba(190,242,100,0.04)" stroke={guideColor} strokeWidth="1.4" strokeDasharray="6 4" />
        <polygon points="156,103 252,93 262,217 146,227"
          fill="url(#tiltSilhouette)" opacity="0.6" />
        <defs>
          <linearGradient id="tiltSilhouette" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3a1a08" />
            <stop offset="60%" stopColor="#7a3a14" />
            <stop offset="100%" stopColor="#f97316" stopOpacity="0.5" />
          </linearGradient>
        </defs>
        {/* tilt protractor */}
        <g transform="translate(200,260)">
          <line x1="-44" y1="0" x2="44" y2="0" stroke="var(--ink-4)" strokeWidth="0.8" />
          <line x1="0" y1="0" x2="38" y2="-22" stroke={guideColor} strokeWidth="1.2" />
          <text x="22" y="-6" fontFamily="Geist Mono" fontSize="8" fill={guideColor}>30°</text>
        </g>
        <text x="200" y="74" textAnchor="middle" fontFamily="Geist Mono" fontSize="7" fill={guideColor} letterSpacing="0.18em">TILT TOP TOWARD CAMERA</text>
      </svg>
    );
  }
  if (pose && pose.startsWith('corner-')) {
    const which = pose.split('-')[1]; // tl, tr, bl, br
    // macro-corner crosshair w/ huge zoomed-in card-corner silhouette
    const isLeft = which.endsWith('l');
    const isTop = which.startsWith('t');
    return (
      <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet">
        {/* zoomed card corner */}
        <rect
          x={isLeft ? 130 : 60}
          y={isTop ? 80 : 30}
          width="210"
          height="220"
          rx="14"
          fill="url(#cornerSilhouette)"
          opacity="0.55"
        />
        <defs>
          <radialGradient id="cornerSilhouette">
            <stop offset="0%" stopColor="#f97316" stopOpacity="0.7" />
            <stop offset="80%" stopColor="#3a1a08" />
          </radialGradient>
        </defs>
        {/* inner ring guide */}
        <circle cx="200" cy="150" r="62" fill="none" stroke={guideColor} strokeWidth="1.4" strokeDasharray="6 4" />
        <circle cx="200" cy="150" r="76" fill="none" stroke={guideColor} strokeWidth="0.6" opacity="0.5" />
        <line x1="200" y1="135" x2="200" y2="165" stroke={guideColor} strokeWidth="1" />
        <line x1="185" y1="150" x2="215" y2="150" stroke={guideColor} strokeWidth="1" />
        <text x="200" y="245" textAnchor="middle" fontFamily="Geist Mono" fontSize="7" fill={guideColor} letterSpacing="0.18em">FILL INNER RING · {which.toUpperCase()}</text>
      </svg>
    );
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────────
function QualityBand({ quality }) {
  // Live readout: blur, glare, perspective, distance, lighting
  // Demo values driven by `quality` tweak: 'good', 'warn', 'fail'
  const metrics = {
    good: [
      { l: 'Blur',        v: 0.91, ok: true,  unit: 'B', desc: 'Sharp · Laplacian variance > 220' },
      { l: 'Glare',       v: 0.04, ok: true,  unit: 'G', desc: 'No bloom regions detected' },
      { l: 'Perspective', v: 0.6,  ok: true,  unit: '°', desc: 'Within ±2° tolerance' },
      { l: 'Distance',    v: 0.92, ok: true,  unit: 'D', desc: 'Card fills 78% of frame' },
      { l: 'Lighting',    v: 0.87, ok: true,  unit: 'L', desc: 'Diffuse · 5200K · evenly lit' },
    ],
    warn: [
      { l: 'Blur',        v: 0.84, ok: true,  unit: 'B', desc: 'Acceptable · steady the camera' },
      { l: 'Glare',       v: 0.18, ok: true,  unit: 'G', desc: 'Faint hotspot top-right' },
      { l: 'Perspective', v: 2.4,  ok: false, unit: '°', desc: 'Tilt 2.4° · level the card' },
      { l: 'Distance',    v: 0.71, ok: true,  unit: 'D', desc: 'Slightly too close' },
      { l: 'Lighting',    v: 0.79, ok: true,  unit: 'L', desc: 'Mildly uneven · move 5cm' },
    ],
    fail: [
      { l: 'Blur',        v: 0.61, ok: false, unit: 'B', desc: 'Blurry · brace device' },
      { l: 'Glare',       v: 0.42, ok: false, unit: 'G', desc: 'Glare band over text · diffuse light' },
      { l: 'Perspective', v: 4.8,  ok: false, unit: '°', desc: 'Heavy tilt · re-level' },
      { l: 'Distance',    v: 0.55, ok: true,  unit: 'D', desc: 'Card fills 55% · move closer' },
      { l: 'Lighting',    v: 0.62, ok: false, unit: 'L', desc: 'One-sided light · rotate' },
    ],
  }[quality];

  return (
    <div style={{
      background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 5,
      padding: 10, display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 8,
    }}>
      {metrics.map(m => <QualityMetric key={m.l} m={m} />)}
    </div>
  );
}

function QualityMetric({ m }) {
  const tone = m.ok ? 'var(--lime)' : 'var(--rose)';
  const display = m.l === 'Perspective' ? m.v.toFixed(1) + '°' : m.v.toFixed(2);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4, padding: '4px 6px', borderLeft: '1px solid var(--line-2)' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>{m.l}</span>
        <span className="mono" style={{ fontSize: 12, color: tone, fontVariantNumeric: 'tabular-nums', fontWeight: 500 }}>{display}</span>
      </div>
      <div style={{ height: 2, background: 'var(--bg-3)', borderRadius: 1 }}>
        <div style={{ width: Math.min(100, m.v * 100) + '%', height: '100%', background: tone, borderRadius: 1 }} />
      </div>
      <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.02em', lineHeight: 1.3 }}>{m.desc}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CaptureButton({ quality, onCapture }) {
  const isOK = quality === 'good';
  return (
    <button onClick={onCapture} style={{
      all: 'unset', cursor: isOK ? 'pointer' : 'not-allowed',
      display: 'inline-flex', alignItems: 'center', gap: 10,
      padding: '10px 18px',
      background: isOK ? 'var(--lime)' : 'var(--bg-3)',
      color: isOK ? 'var(--bg-0)' : 'var(--ink-4)',
      border: '1px solid ' + (isOK ? 'var(--lime)' : 'var(--line)'),
      borderRadius: 4,
      fontFamily: 'Geist Mono, monospace', fontSize: 11, fontWeight: 600,
      letterSpacing: '0.14em', textTransform: 'uppercase',
      boxShadow: isOK ? '0 0 0 4px rgba(190,242,100,0.12)' : 'none',
      transition: 'all 150ms',
    }}>
      <span style={{ display: 'inline-flex', width: 16, height: 16, borderRadius: 99, background: isOK ? 'var(--bg-0)' : 'var(--ink-5)', alignItems: 'center', justifyContent: 'center' }}>
        <span style={{ width: 6, height: 6, borderRadius: 99, background: isOK ? 'var(--lime)' : 'transparent' }} />
      </span>
      Capture
      <span className="mono" style={{ fontSize: 9, opacity: 0.65 }}>↵ space</span>
    </button>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CoachPanel({ shot, quality }) {
  return (
    <aside style={{ display: 'flex', flexDirection: 'column', gap: 14, maxHeight: 'calc(100vh - 220px)', overflowY: 'auto' }}>
      <PoseTipsCard shot={shot} quality={quality} />
      <RecentCapturesCard />
      <SystemHealthCard />
    </aside>
  );
}

function PoseTipsCard({ shot, quality }) {
  const tipsByPose = {
    flat: [
      'Card flat on neutral surface (matte gray works best)',
      'Phone parallel to card · use a tripod or table edge',
      'Diffuse light from two sides · no overhead glare',
      'Card fills > 70% of frame; gutters of equal width',
    ],
    tilt: [
      'Tilt top edge of card ~30° toward camera',
      'Keep card roughly centered in frame',
      'Same lighting as front_full · do not move the lights',
      'Holo parallax detector needs visible foil shift',
    ],
    'corner-tl': ['Macro mode on phone', 'Fill the inner ring with the card corner', 'Card flat, lit evenly'],
    'corner-tr': ['Macro mode on phone', 'Fill the inner ring with the card corner', 'Card flat, lit evenly'],
    'corner-bl': ['Macro mode on phone', 'Fill the inner ring with the card corner', 'Card flat, lit evenly'],
    'corner-br': ['Macro mode on phone', 'Fill the inner ring with the card corner', 'Card flat, lit evenly'],
  };
  const tips = tipsByPose[shot.pose] || [];

  const liveCoachByQuality = {
    good: { tone: 'var(--lime)', label: 'Ready to capture', body: 'All gates green. Hold steady — the shutter will sound when the shot is locked.' },
    warn: { tone: 'var(--amber)', label: 'Almost there', body: 'Perspective is slightly off. Level the card or rotate the phone — gates will turn green.' },
    fail: { tone: 'var(--rose)', label: 'Hold off', body: 'Glare on the holo · diffuse the light source or move it 90°. Quality gate would reject this shot.' },
  }[quality];

  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Live coach</div>

      <div style={{
        padding: '10px 12px',
        background: liveCoachByQuality.tone + '14',
        border: '1px solid ' + liveCoachByQuality.tone + '55',
        borderRadius: 4,
      }}>
        <div className="mono" style={{ fontSize: 9, color: liveCoachByQuality.tone, letterSpacing: '0.18em', textTransform: 'uppercase' }}>{liveCoachByQuality.label}</div>
        <div style={{ fontSize: 12, color: 'var(--ink-1)', marginTop: 6, lineHeight: 1.5 }}>{liveCoachByQuality.body}</div>
      </div>

      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase', marginTop: 4 }}>Pose checklist</div>
      <ul style={{ display: 'flex', flexDirection: 'column', gap: 6, padding: 0, margin: 0, listStyle: 'none' }}>
        {tips.map((tip, i) => (
          <li key={i} style={{ display: 'flex', gap: 8, fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.45 }}>
            <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', minWidth: 12, marginTop: 3 }}>{String(i+1).padStart(2,'0')}</span>
            <span>{tip}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function RecentCapturesCard() {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline' }}>
        <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Captured this run</div>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)' }}>3 kept · 1 retake</span>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {MOCK_SHOT_RESULTS.map((r, i) => <RecentCaptureRow key={i} r={r} />)}
      </div>
    </div>
  );
}

function RecentCaptureRow({ r }) {
  const passed = r.passed;
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '36px 1fr auto',
      gap: 10, padding: '6px 8px', borderRadius: 3,
      background: r.kept ? 'transparent' : 'rgba(251,113,133,0.04)',
      border: '1px solid ' + (r.kept ? 'var(--line-2)' : 'rgba(251,113,133,0.2)'),
      alignItems: 'center',
      opacity: r.retake ? 0.55 : 1,
    }}>
      {/* thumbnail */}
      <div style={{
        width: 36, height: 50, borderRadius: 2,
        background: r.kind.startsWith('corner') ? 'radial-gradient(circle at 30% 30%, #f97316, #3a1a08 65%)' :
                    r.kind === 'tilt_30' ? 'linear-gradient(115deg, #3a1a08, #7a3a14, #f97316)' :
                    r.kind === 'front_full_flash' ? 'linear-gradient(135deg, #3a1a08, #f97316, #fff8e0)' :
                    r.kind === 'back_full' ? 'linear-gradient(135deg, #1e3a2a, #4a8a6a)' :
                    'linear-gradient(135deg, #3a1a08, #7a3a14, #f97316)',
        opacity: r.retake ? 0.4 : 0.85,
        position: 'relative', flexShrink: 0,
      }}>
        {r.retake && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <span style={{ fontSize: 16, color: 'var(--rose)' }}>↺</span>
          </div>
        )}
      </div>
      <div style={{ minWidth: 0 }}>
        <div style={{ display: 'flex', gap: 6, alignItems: 'baseline' }}>
          <span style={{ fontSize: 11, color: 'var(--ink-1)', fontWeight: 500 }}>{r.kind}</span>
          {r.retake && <span className="mono" style={{ fontSize: 8, color: 'var(--rose)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>retake</span>}
        </div>
        <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', marginTop: 2 }}>
          B {r.blur.toFixed(2)} · G {r.glare.toFixed(2)} · P {r.perspective.toFixed(1)}°
        </div>
        {r.reasons && (
          <div className="mono" style={{ fontSize: 9, color: 'var(--rose)', marginTop: 2 }}>{r.reasons.join(' · ')}</div>
        )}
      </div>
      <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', whiteSpace: 'nowrap' }}>{r.ts}</span>
    </div>
  );
}

function SystemHealthCard() {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>System</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, fontSize: 11 }}>
        <SysItem k="API" v="healthy" tone="lime" />
        <SysItem k="S3 upload" v="ready" tone="lime" />
        <SysItem k="Quality eval" v="GPU-0 · idle" tone="lime" />
        <SysItem k="Auth" v="signed in · belmer" tone="ink" />
      </div>
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', borderTop: '1px solid var(--line-2)', paddingTop: 8, marginTop: 4, lineHeight: 1.5 }}>
        Quality model · v0.4.1<br/>
        Bandwidth · 122 Mbps · 12 ms RTT<br/>
        Submission TTL · saved 24h
      </div>
    </div>
  );
}

function SysItem({ k, v, tone }) {
  const c = { lime: 'var(--lime)', ink: 'var(--ink-1)', rose: 'var(--rose)', amber: 'var(--amber)' }[tone];
  return (
    <div style={{ background: 'var(--bg-2)', borderRadius: 3, padding: '6px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
      <span className="mono" style={{ fontSize: 8, color: 'var(--ink-3)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>{k}</span>
      <span className="mono" style={{ fontSize: 11, color: c }}>{v}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CaptureTweaks({ t, setTweak, shotCount }) {
  return (
    <TweaksPanel>
      <TweakSection label="Wizard state" />
      <TweakSlider label="Current shot index" min={0} max={shotCount} step={1}
        value={t.currentShot} onChange={v => setTweak('currentShot', v)} />
      <TweakRadio label="Live quality"
        value={t.liveQuality}
        options={['good','warn','fail']}
        onChange={v => setTweak('liveQuality', v)} />
      <TweakSection label="Layout" />
      <TweakToggle label="Show shot list rail" value={t.showShotList} onChange={v => setTweak('showShotList', v)} />
      <TweakRadio label="Device"
        value={t.deviceFraming}
        options={['desktop','phone']}
        onChange={v => setTweak('deviceFraming', v)} />
    </TweaksPanel>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
