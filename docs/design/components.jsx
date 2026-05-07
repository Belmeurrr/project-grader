// components.jsx — atomic UI primitives for the Workbench
// All components are scoped to the workbench aesthetic: dark zinc, lime accent,
// Geist + Geist Mono. Composable, no per-component styles object collisions.

const wbStyles = {
  // Reusable inline tokens (used sparingly; most styling is via className+CSS-in-head)
  card: {
    background: 'var(--bg-1)',
    border: '1px solid var(--line)',
    borderRadius: 8,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
// Severity color helper
const SEV_COLOR = {
  ok: 'var(--lime)',
  minor: 'var(--amber)',
  major: 'var(--rose)',
  unknown: 'var(--ink-4)',
};
const VERDICT_COLOR = {
  authentic: 'var(--lime)',
  suspicious: 'var(--amber)',
  likely_counterfeit: 'var(--rose)',
  unverified: 'var(--ink-4)',
};
const VERDICT_LABEL = {
  authentic: 'Authentic',
  suspicious: 'Suspicious',
  likely_counterfeit: 'Counterfeit',
  unverified: 'Unverified',
};

// Grade → tone (10/9.5 = bright lime, 8s = lime, 7 = amber, <7 = rose)
function gradeTone(g) {
  if (g === null || g === undefined) return 'var(--ink-3)';
  if (g >= 9) return 'var(--lime-bright)';
  if (g >= 8) return 'var(--lime)';
  if (g >= 7) return 'var(--amber)';
  return 'var(--rose)';
}

// ─────────────────────────────────────────────────────────────────────────────
// CardArt — placeholder card face with a glyph + color, sized to 5:7
// In a real build this would be the dewarped 750x1050 canonical scan.
function CardArt({ card, size = 'md' }) {
  const w = size === 'sm' ? 64 : size === 'md' ? 120 : size === 'lg' ? 220 : 360;
  const h = Math.round(w * 7 / 5);
  return (
    <div style={{
      width: w, height: h, position: 'relative', flexShrink: 0,
      borderRadius: Math.max(4, w * 0.04),
      overflow: 'hidden',
      background: `linear-gradient(135deg, ${card.color}22 0%, ${card.color}08 60%, #0a0a0a 100%)`,
      border: '1px solid var(--line)',
      boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.02), 0 4px 16px rgba(0,0,0,0.4)',
    }}>
      <div style={{
        position: 'absolute', inset: '6%',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: Math.max(2, w * 0.02),
      }} />
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        fontSize: w * 0.45, color: card.color, opacity: 0.55,
        filter: 'drop-shadow(0 0 12px ' + card.color + '55)',
      }}>{card.glyph}</div>
      <div style={{
        position: 'absolute', left: '8%', right: '8%', top: '8%',
        fontFamily: 'Geist Mono', fontSize: Math.max(7, w * 0.055),
        color: 'rgba(255,255,255,0.5)', textTransform: 'uppercase',
        letterSpacing: '0.08em', display: 'flex', justifyContent: 'space-between',
      }}>
        <span>{card.set}</span><span>{card.number}</span>
      </div>
      <div style={{
        position: 'absolute', left: '8%', right: '8%', bottom: '8%',
        fontFamily: 'Geist', fontWeight: 600, fontSize: Math.max(8, w * 0.07),
        color: 'rgba(255,255,255,0.85)', lineHeight: 1.05,
      }}>{card.name}</div>
      {/* Scan-line overlay for that "captured" feel */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        background: 'repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 3px)',
        mixBlendMode: 'overlay',
      }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CertID — monospace pill
function CertID({ id, size = 12 }) {
  return (
    <span className="mono" style={{
      fontSize: size, color: 'var(--ink-3)', letterSpacing: '0.02em',
    }}>{id}</span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GradeBadge — large display grade with tone
function GradeBadge({ grade, size = 'md' }) {
  const fontSize = size === 'sm' ? 18 : size === 'md' ? 28 : size === 'lg' ? 56 : 80;
  const display = grade === null || grade === undefined ? '—' : grade.toFixed(1);
  return (
    <span style={{
      fontFamily: 'Geist', fontWeight: 500, fontSize, lineHeight: 1,
      color: gradeTone(grade), fontVariantNumeric: 'tabular-nums',
      letterSpacing: '-0.02em',
    }}>{display}</span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// VerdictDot
function VerdictDot({ verdict }) {
  return (
    <span style={{
      display: 'inline-block', width: 6, height: 6, borderRadius: 3,
      background: VERDICT_COLOR[verdict] ?? 'var(--ink-4)',
      boxShadow: '0 0 8px ' + (VERDICT_COLOR[verdict] ?? 'transparent'),
    }} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MicroBar — horizontal score bar, used in detector rows
function MicroBar({ value, color = 'var(--lime)', height = 4 }) {
  return (
    <div style={{
      width: '100%', height, background: 'var(--bg-3)',
      borderRadius: height / 2, overflow: 'hidden',
    }}>
      <div style={{
        width: (value * 100).toFixed(1) + '%', height: '100%',
        background: color, borderRadius: height / 2,
      }} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// ConfidenceBand — visualizes lower–upper around final grade on a 1-10 axis
function ConfidenceBand({ final, lower, upper, conf }) {
  if (final === null) return (
    <div className="mono" style={{ fontSize: 10, color: 'var(--ink-4)' }}>
      Confidence pending — heads not trained
    </div>
  );
  const pos = (v) => ((v - 1) / 9 * 100).toFixed(2) + '%';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div style={{
        position: 'relative', height: 18,
        background: 'linear-gradient(90deg, rgba(251,113,133,0.06), rgba(251,191,36,0.06) 50%, rgba(190,242,100,0.08))',
        borderRadius: 4, border: '1px solid var(--line)',
      }}>
        {/* tick marks at 1,4,7,10 */}
        {[1,4,7,10].map(t => (
          <div key={t} style={{
            position: 'absolute', left: pos(t), top: 0, bottom: 0, width: 1,
            background: 'rgba(255,255,255,0.04)',
          }} />
        ))}
        {/* band */}
        <div style={{
          position: 'absolute', left: pos(lower), width: `calc(${pos(upper)} - ${pos(lower)})`,
          top: 4, bottom: 4, background: gradeTone(final), opacity: 0.25, borderRadius: 2,
        }} />
        {/* point */}
        <div style={{
          position: 'absolute', left: `calc(${pos(final)} - 1px)`, top: 0, bottom: 0,
          width: 2, background: gradeTone(final),
          boxShadow: '0 0 8px ' + gradeTone(final),
        }} />
      </div>
      <div className="mono" style={{
        fontSize: 10, color: 'var(--ink-3)', display: 'flex',
        justifyContent: 'space-between',
      }}>
        <span>{lower.toFixed(1)} – {upper.toFixed(1)}</span>
        <span>{(conf * 100).toFixed(0)}% conf · ±{((upper - lower) / 2).toFixed(2)}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SectionHead — small uppercase label with optional meta on the right
function SectionHead({ label, meta, children }) {
  return (
    <div style={{
      display: 'flex', alignItems: 'baseline', justifyContent: 'space-between',
      borderBottom: '1px solid var(--line-2)', paddingBottom: 6, marginBottom: 12,
    }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <span className="mono" style={{
          fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.18em',
          color: 'var(--ink-3)',
        }}>{label}</span>
        {children}
      </div>
      {meta && (
        <span className="mono" style={{
          fontSize: 10, color: 'var(--ink-4)', letterSpacing: '0.04em',
        }}>{meta}</span>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MetricCell — label + big number + optional sub
function MetricCell({ label, value, sub, tone, mono = true, align = 'left' }) {
  return (
    <div style={{
      display: 'flex', flexDirection: 'column', gap: 3,
      textAlign: align,
    }}>
      <div className="mono" style={{
        fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.16em',
        color: 'var(--ink-4)',
      }}>{label}</div>
      <div className={mono ? 'mono' : ''} style={{
        fontSize: 20, color: tone || 'var(--ink-0)',
        fontVariantNumeric: 'tabular-nums', fontWeight: 500,
        letterSpacing: '-0.01em', lineHeight: 1,
      }}>{value}</div>
      {sub && (
        <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>{sub}</div>
      )}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SubgradeBlock — 4 criteria + final; vertical bar below each
function SubgradeBlock({ card }) {
  const items = [
    { k: 'Centering', v: card.centering },
    { k: 'Corners', v: card.corners },
    { k: 'Edges', v: card.edges },
    { k: 'Surface', v: card.surface },
    { k: 'Final', v: card.final, primary: true },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 1, background: 'var(--line-2)', border: '1px solid var(--line-2)' }}>
      {items.map((it) => (
        <div key={it.k} style={{
          background: it.primary ? 'rgba(190,242,100,0.04)' : 'var(--bg-1)',
          padding: '14px 14px',
          display: 'flex', flexDirection: 'column', gap: 8,
          minHeight: 100,
        }}>
          <div className="mono" style={{
            fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.18em',
            color: it.primary ? 'var(--lime)' : 'var(--ink-4)',
          }}>{it.k}</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
            <GradeBadge grade={it.v} size={it.primary ? 'lg' : 'md'} />
          </div>
          {it.v !== null && it.v !== undefined && (
            <div style={{
              height: 2, background: 'var(--bg-3)', borderRadius: 1, overflow: 'hidden', marginTop: 'auto',
            }}>
              <div style={{
                width: (it.v / 10 * 100) + '%', height: '100%',
                background: gradeTone(it.v),
              }} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DetectorRow — one of the 7 counterfeit detectors
function DetectorRow({ d }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '120px 1fr 60px 60px 70px',
      alignItems: 'center', gap: 12, padding: '8px 0',
      borderBottom: '1px solid var(--line-2)',
    }}>
      <span className="mono" style={{
        fontSize: 10, color: 'var(--ink-2)', textTransform: 'uppercase',
        letterSpacing: '0.06em',
      }}>{d.k}</span>
      <MicroBar value={d.s} color={VERDICT_COLOR[d.v]} />
      <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
        {(d.s * 100).toFixed(0)}%
      </span>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
        ±{((1-d.c) * 50).toFixed(0)}
      </span>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, justifyContent: 'flex-end' }}>
        <VerdictDot verdict={d.v} />
        <span className="mono" style={{ fontSize: 10, color: VERDICT_COLOR[d.v], textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {VERDICT_LABEL[d.v]}
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparkline — for tiny history charts
function Sparkline({ values, color = 'var(--lime)', width = 120, height = 28 }) {
  if (!values || values.length === 0) return null;
  const min = Math.min(...values), max = Math.max(...values);
  const range = Math.max(0.01, max - min);
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg width={width} height={height} style={{ display: 'block' }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={width} cy={height - ((values[values.length-1] - min) / range) * height} r="2" fill={color} />
    </svg>
  );
}

// Export to window for cross-script visibility
Object.assign(window, {
  CardArt, CertID, GradeBadge, VerdictDot, MicroBar, ConfidenceBand,
  SectionHead, MetricCell, SubgradeBlock, DetectorRow, Sparkline,
  SEV_COLOR, VERDICT_COLOR, VERDICT_LABEL, gradeTone,
});
