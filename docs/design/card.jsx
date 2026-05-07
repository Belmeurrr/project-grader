// card.jsx — Card detail (owner view)
const { useState, useMemo } = React;

const CARD = {
  card_id: 'card_4d2f-base1-charizard-holo',
  name: 'Charizard',
  set_code: 'Base Set',
  set_full: 'Base Set · 1999',
  card_number: '4/102',
  edition: 'Holo · Unlimited',
  rarity: 'Holo Rare',
  artist: 'Mitsuhiro Arita',
  estimated_value: { lo: 320, hi: 480, currency: 'USD', source: 'TCGplayer · 30d median' },
  best_grade: 9.0,
  best_cert: 'PG-2026-0F1A4C-9D71',
  current_holders: 1,
  notes: 'Acquired May 2025 from a local shop. First WOTC holo I owned as a kid was Charizard — this is a sentimental hold, not for sale.',
};

const SUBMISSIONS = [
  {
    id: 'PG-2026-0F1A4C-9D71',
    submitted_at: '2026-05-07T14:08:24Z',
    completed_at: '2026-05-07T14:08:45Z',
    status: 'complete',
    grades: { centering: 9.0, corners: 8.5, edges: 9.0, surface: 9.5, final: 9.0, confidence: 0.892 },
    auth: { verdict: 'authentic', confidence: 0.973 },
    capture_quality: 'pass', shots: 8, notes: 'After cleaning sleeve, dust removed.',
  },
  {
    id: 'PG-2026-0E91FF-2A1C',
    submitted_at: '2026-04-12T09:31:02Z',
    completed_at: '2026-04-12T09:31:25Z',
    status: 'complete',
    grades: { centering: 9.0, corners: 8.0, edges: 8.5, surface: 9.0, final: 8.5, confidence: 0.851 },
    auth: { verdict: 'authentic', confidence: 0.967 },
    capture_quality: 'warn', shots: 8, notes: 'Some glare on TR corner — retake recommended for next regrade.',
  },
  {
    id: 'PG-2025-0BCC10-77E3',
    submitted_at: '2025-11-02T19:11:48Z',
    completed_at: '2025-11-02T19:12:09Z',
    status: 'complete',
    grades: { centering: 8.5, corners: 8.0, edges: 8.0, surface: 9.0, final: 8.0, confidence: 0.812 },
    auth: { verdict: 'authentic', confidence: 0.954 },
    capture_quality: 'pass', shots: 6, notes: 'Initial grading. Skipped flash + tilt shots.',
  },
  {
    id: 'PG-2025-0B2A88-1F40',
    submitted_at: '2025-08-15T10:42:11Z',
    completed_at: null,
    status: 'failed',
    grades: null,
    auth: null,
    capture_quality: 'fail', shots: 4, notes: 'Insufficient shots — front + back rejected for blur.',
  },
];

const SHOT_KEYS = [
  { k: 'front', t: 'Front' }, { k: 'back', t: 'Back' },
  { k: 'tl', t: 'Corner TL' }, { k: 'tr', t: 'Corner TR' },
  { k: 'bl', t: 'Corner BL' }, { k: 'br', t: 'Corner BR' },
  { k: 'flash', t: 'Flash' }, { k: 'tilt', t: 'Tilt' },
];

// ─────────────────────────────────────────────────────────────────────────────
function App() {
  const DEFAULTS = window.__CARD_TWEAK_DEFAULTS_RAW || { tab: 'history', showTimeline: true, denseShots: false };
  const [t, setTweak] = useTweaks(DEFAULTS);
  const [tab, setTab] = useState(t.tab);

  return (
    <main style={{
      maxWidth: 1380, margin: '24px auto', padding: '0 28px 80px',
      display: 'flex', flexDirection: 'column', gap: 28,
    }}>
      <CardHeader card={CARD} />

      <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 380px) minmax(0, 1fr)', gap: 36 }}>
        <aside style={{ display: 'flex', flexDirection: 'column', gap: 22, position: 'sticky', top: 80, alignSelf: 'start' }}>
          <CardArtPanel />
          <CardFacts card={CARD} />
          <CardNotes card={CARD} />
        </aside>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 22, minWidth: 0 }}>
          <BestGradeStrip card={CARD} subs={SUBMISSIONS} />
          {t.showTimeline && <GradeTimeline subs={SUBMISSIONS.filter(s => s.status === 'complete')} />}

          <TabBar tab={tab} setTab={setTab} />
          {tab === 'history' && <HistoryTab subs={SUBMISSIONS} />}
          {tab === 'shots' && <ShotsTab dense={t.denseShots} />}
          {tab === 'comparison' && <ComparisonTab subs={SUBMISSIONS.filter(s => s.status === 'complete')} />}
          {tab === 'flags' && <FlagsTab subs={SUBMISSIONS} />}
        </div>
      </div>

      <CardTweaks t={t} setTweak={setTweak} />
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CardHeader({ card }) {
  return (
    <header style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', gap: 24,
      paddingBottom: 22, borderBottom: '1px solid var(--line)',
      flexWrap: 'wrap',
    }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <span className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
          Workbench · Card detail
        </span>
        <h1 style={{ fontFamily: 'Instrument Serif, serif', fontSize: 52, fontWeight: 400, color: 'var(--ink-0)', margin: 0, letterSpacing: '-0.02em', lineHeight: 1 }}>
          {card.name}
        </h1>
        <div className="mono" style={{ fontSize: 11, color: 'var(--ink-2)', display: 'flex', gap: 12, flexWrap: 'wrap', letterSpacing: '0.04em' }}>
          <span>{card.set_full}</span>
          <span style={{ color: 'var(--ink-5)' }}>·</span>
          <span>#{card.card_number}</span>
          <span style={{ color: 'var(--ink-5)' }}>·</span>
          <span>{card.edition}</span>
          <span style={{ color: 'var(--ink-5)' }}>·</span>
          <span>{card.rarity}</span>
          <span style={{ color: 'var(--ink-5)' }}>·</span>
          <span style={{ color: 'var(--ink-3)' }}>art · {card.artist}</span>
        </div>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button className="pg-btn pg-btn-ghost">View public cert ↗</button>
        <button className="pg-btn pg-btn-ghost">Add note</button>
        <button className="pg-btn pg-btn-primary">+ Regrade</button>
      </div>
    </header>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CardArtPanel() {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <SectionLabel hint="canonical · 750×1050">Art</SectionLabel>
      <CardArtCharizard />
    </section>
  );
}

function CardArtCharizard() {
  return (
    <div style={{
      aspectRatio: '5/7', borderRadius: 9, overflow: 'hidden',
      background: '#0a0a0a',
      border: '1px solid var(--line-2)',
      boxShadow: '0 30px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.02) inset',
      position: 'relative',
    }}>
      <div style={{
        position: 'absolute', inset: '4%', borderRadius: 4,
        background: 'linear-gradient(140deg, #4d2410 0%, #7a3812 35%, #c0531a 60%, #f97316 80%, #fbbf24 100%)',
      }} />
      <div style={{
        position: 'absolute', top: '10%', left: '8%', right: '8%', height: '56%',
        background: 'radial-gradient(ellipse at 50% 35%, #fde68a, transparent 60%), linear-gradient(180deg, #f97316 0%, #ea580c 50%, #7a2e0c 100%)',
        border: '4px solid #fbbf24',
        boxShadow: 'inset 0 0 30px rgba(0,0,0,0.4)',
      }}>
        <svg viewBox="0 0 100 80" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', opacity: 0.85 }}>
          <path d="M 20 60 Q 30 30 50 25 Q 70 20 78 35 Q 84 25 88 30 Q 82 38 80 45 Q 88 50 82 58 Q 70 62 60 58 Q 55 70 45 70 Q 35 72 30 65 Q 25 68 20 60 Z" fill="#7a2e0c" stroke="#1a0a04" strokeWidth="0.8" />
          <path d="M 38 50 L 45 38 L 52 48 Z" fill="#fbbf24" />
          <circle cx="72" cy="35" r="1.8" fill="#fbbf24" />
        </svg>
      </div>
      <div style={{ position: 'absolute', top: '4%', left: '8%', fontFamily: 'Instrument Serif, serif', fontSize: 13, fontWeight: 600, color: '#1a0a04' }}>Charizard</div>
      <div style={{ position: 'absolute', top: '4%', right: '8%', fontFamily: 'Geist Mono, monospace', fontSize: 11, fontWeight: 700, color: '#1a0a04', display: 'flex', alignItems: 'baseline', gap: 4 }}>
        <span style={{ fontSize: 6 }}>HP</span><span>120</span>
      </div>
      <div style={{ position: 'absolute', bottom: '4%', left: '8%', right: '8%', fontFamily: 'Geist Mono, monospace', fontSize: 7, color: '#1a0a04', display: 'flex', justifyContent: 'space-between' }}>
        <span>4/102 ★</span><span>©1995, 96, 98 Nintendo</span>
      </div>
      <div style={{ position: 'absolute', inset: '4%', borderRadius: 4, background: 'linear-gradient(105deg, transparent 30%, rgba(255,255,255,0.18) 45%, transparent 60%)', mixBlendMode: 'screen', pointerEvents: 'none' }} />
    </div>
  );
}

function CardFacts({ card }) {
  const facts = [
    ['Variant ID', card.card_id, true],
    ['Rarity', card.rarity, false],
    ['Artist', card.artist, false],
    ['Best grade', card.best_grade.toFixed(1), false],
    ['Best cert', card.best_cert, true],
    ['Held by you', card.current_holders + ' copy', false],
    ['Est. value', '$' + card.estimated_value.lo + '–' + card.estimated_value.hi, false],
  ];
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <SectionLabel hint="metadata">Facts</SectionLabel>
      <dl style={{ margin: 0, display: 'flex', flexDirection: 'column', gap: 1, background: 'var(--line-2)', border: '1px solid var(--line)', borderRadius: 5, overflow: 'hidden' }}>
        {facts.map(([k, v, mono]) => (
          <div key={k} style={{ display: 'grid', gridTemplateColumns: '110px 1fr', gap: 12, padding: '7px 12px', background: 'var(--bg-1)' }}>
            <dt className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>{k}</dt>
            <dd className={mono ? 'mono' : ''} style={{ margin: 0, fontSize: mono ? 10 : 12, color: 'var(--ink-1)', textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis' }}>{v}</dd>
          </div>
        ))}
      </dl>
      <p className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', margin: '4px 2px 0', letterSpacing: '0.02em' }}>
        Value: {card.estimated_value.source}
      </p>
    </section>
  );
}

function CardNotes({ card }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <SectionLabel hint="private · owner-only">Notes</SectionLabel>
      <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 5, padding: '12px 14px' }}>
        <p style={{ margin: 0, fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.6 }}>{card.notes}</p>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function BestGradeStrip({ card, subs }) {
  const completed = subs.filter(s => s.status === 'complete');
  const best = completed.reduce((b, s) => s.grades.final > b.grades.final ? s : b, completed[0]);
  const latest = completed[0];
  const trend = completed.length >= 2 ? completed[0].grades.final - completed[1].grades.final : 0;

  return (
    <section style={{
      display: 'grid', gridTemplateColumns: '1.4fr 1fr 1fr 1fr', gap: 1,
      background: 'var(--line-2)', border: '1px solid var(--line)', borderRadius: 8, overflow: 'hidden',
    }}>
      <BestCell label="Best grade" value={best.grades.final.toFixed(1)} sub={best.id.slice(0, 16) + '…'} accent />
      <BestCell label="Latest grade" value={latest.grades.final.toFixed(1)} sub={new Date(latest.completed_at).toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' })} />
      <BestCell label="Trend" value={(trend >= 0 ? '+' : '') + trend.toFixed(1)} sub="vs prior grade"
        valueColor={trend > 0 ? 'var(--lime)' : trend < 0 ? 'var(--rose)' : 'var(--ink-1)'} />
      <BestCell label="Submissions" value={subs.length} sub={`${completed.length} complete · ${subs.length - completed.length} failed`} />
    </section>
  );
}

function BestCell({ label, value, sub, accent = false, valueColor }) {
  return (
    <div style={{
      padding: '20px 22px',
      background: accent
        ? 'linear-gradient(135deg, rgba(190,242,100,0.04), transparent 60%), var(--bg-1)'
        : 'var(--bg-1)',
      display: 'flex', flexDirection: 'column', gap: 6,
    }}>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{label}</span>
      <span style={{
        fontFamily: 'Instrument Serif, serif', fontSize: 44, lineHeight: 0.95,
        color: valueColor || (accent ? 'var(--ink-0)' : 'var(--ink-0)'),
        letterSpacing: '-0.02em',
      }}>{value}</span>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', letterSpacing: '0.02em', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>{sub}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function GradeTimeline({ subs }) {
  // chronological order
  const sorted = [...subs].sort((a, b) => new Date(a.completed_at) - new Date(b.completed_at));
  const minG = 7.5, maxG = 10.0;
  const W = 800, H = 180, padX = 50, padY = 20;
  const xFor = i => sorted.length > 1 ? padX + (i / (sorted.length - 1)) * (W - 2 * padX) : W / 2;
  const yFor = g => H - padY - ((g - minG) / (maxG - minG)) * (H - 2 * padY);

  return (
    <section>
      <SectionLabel hint={`${sorted.length} grades · oldest → newest`}>Grade timeline</SectionLabel>
      <div style={{
        marginTop: 12, padding: '20px 24px',
        background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 8,
      }}>
        <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* gridlines + scale */}
          {[7.5, 8.0, 8.5, 9.0, 9.5, 10].map(g => (
            <g key={g}>
              <line x1={padX} x2={W - padX} y1={yFor(g)} y2={yFor(g)} stroke="var(--line-2)" strokeWidth="0.5" strokeDasharray={g === 9 || g === 10 ? '0' : '2 3'} />
              <text x={padX - 8} y={yFor(g) + 3} fontSize="9" fontFamily="Geist Mono" fill="var(--ink-4)" textAnchor="end">{g.toFixed(1)}</text>
            </g>
          ))}
          {/* PSA-10 line */}
          <text x={W - padX + 4} y={yFor(10) + 3} fontSize="8" fontFamily="Geist Mono" fill="var(--lime-deep)">PSA 10</text>

          {/* confidence band */}
          <path d={
            sorted.map((s, i) => {
              const x = xFor(i);
              const w = (1 - s.grades.confidence) * 2;
              const upper = Math.min(10, s.grades.final + w);
              return (i === 0 ? 'M' : 'L') + x + ',' + yFor(upper);
            }).join(' ') +
            ' ' +
            sorted.slice().reverse().map((s, i) => {
              const realI = sorted.length - 1 - i;
              const x = xFor(realI);
              const w = (1 - s.grades.confidence) * 2;
              const lower = Math.max(1, s.grades.final - w);
              return 'L' + x + ',' + yFor(lower);
            }).join(' ') + ' Z'
          } fill="rgba(190,242,100,0.08)" stroke="none" />

          {/* line */}
          <polyline
            points={sorted.map((s, i) => xFor(i) + ',' + yFor(s.grades.final)).join(' ')}
            fill="none" stroke="var(--lime)" strokeWidth="1.5" />

          {/* points */}
          {sorted.map((s, i) => (
            <g key={s.id}>
              <circle cx={xFor(i)} cy={yFor(s.grades.final)} r="4" fill="var(--lime)" stroke="var(--bg-1)" strokeWidth="2" />
              <text x={xFor(i)} y={yFor(s.grades.final) - 11} fontSize="10" fontFamily="Geist Mono" fontWeight="600" fill="var(--ink-0)" textAnchor="middle">
                {s.grades.final.toFixed(1)}
              </text>
              <text x={xFor(i)} y={H - 4} fontSize="8" fontFamily="Geist Mono" fill="var(--ink-3)" textAnchor="middle">
                {new Date(s.completed_at).toLocaleDateString('en-US', { month: 'short', year: '2-digit' })}
              </text>
            </g>
          ))}
        </svg>
        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 18, marginTop: 4 }}>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 8, height: 8, borderRadius: 4, background: 'var(--lime)' }} />
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>final grade</span>
          </span>
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 12, height: 6, background: 'rgba(190,242,100,0.2)', borderTop: '1px solid var(--lime)', borderBottom: '1px solid var(--lime)' }} />
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>confidence band</span>
          </span>
        </div>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function TabBar({ tab, setTab }) {
  const tabs = [
    { k: 'history',    l: 'Submission history', c: 4 },
    { k: 'shots',      l: 'Latest shots',       c: 8 },
    { k: 'comparison', l: 'Subgrade comparison',c: 3 },
    { k: 'flags',      l: 'Flags & retakes',    c: 2 },
  ];
  return (
    <div style={{ display: 'flex', gap: 0, borderBottom: '1px solid var(--line)' }}>
      {tabs.map(it => {
        const active = it.k === tab;
        return (
          <button key={it.k} onClick={() => setTab(it.k)} style={{
            padding: '10px 16px',
            background: active ? 'rgba(190,242,100,0.04)' : 'transparent',
            border: 0, borderBottom: '2px solid ' + (active ? 'var(--lime)' : 'transparent'),
            cursor: 'pointer', color: active ? 'var(--ink-0)' : 'var(--ink-3)',
            fontFamily: 'inherit', fontSize: 13, fontWeight: 500,
            display: 'inline-flex', alignItems: 'center', gap: 8,
            transition: 'all 120ms', marginBottom: -1,
          }}>
            {it.l}
            <span className="mono" style={{
              fontSize: 9, padding: '1px 6px', borderRadius: 99,
              background: active ? 'rgba(190,242,100,0.1)' : 'var(--bg-2)',
              color: active ? 'var(--lime)' : 'var(--ink-4)',
              letterSpacing: '0.04em',
            }}>{it.c}</span>
          </button>
        );
      })}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function HistoryTab({ subs }) {
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 4 }}>
      {subs.map((s, i) => <HistoryRow key={s.id} sub={s} idx={subs.length - i} latest={i === 0} />)}
    </section>
  );
}

function HistoryRow({ sub, idx, latest }) {
  const [open, setOpen] = useState(latest);
  const isFailed = sub.status === 'failed';

  const StatusIcon = () => {
    if (isFailed) return <span style={{ color: 'var(--rose)' }}>✕</span>;
    return <span style={{ color: 'var(--lime)' }}>✓</span>;
  };

  return (
    <div style={{
      background: 'var(--bg-1)', border: '1px solid var(--line)',
      borderRadius: 6, overflow: 'hidden',
      transition: 'border-color 120ms',
    }}>
      <button onClick={() => setOpen(!open)} style={{
        width: '100%', padding: '12px 16px',
        background: 'transparent', border: 0, cursor: 'pointer',
        display: 'grid', gridTemplateColumns: '50px 100px 1fr 80px 100px 80px 14px',
        gap: 14, alignItems: 'center',
        color: 'inherit', fontFamily: 'inherit', textAlign: 'left',
      }}>
        <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', letterSpacing: '0.06em' }}>
          #{String(idx).padStart(2, '0')}
        </span>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)' }}>
          {new Date(sub.submitted_at).toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' })}
        </span>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 2, minWidth: 0 }}>
          <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)' }}>{sub.id}</span>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>
            {sub.shots} shots · capture {sub.capture_quality}
          </span>
        </div>
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 11 }}>
          <StatusIcon /> <span style={{ color: isFailed ? 'var(--rose)' : 'var(--ink-1)' }}>{isFailed ? 'failed' : 'complete'}</span>
        </span>
        <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 26, color: isFailed ? 'var(--ink-5)' : 'var(--ink-0)', lineHeight: 1, textAlign: 'right' }}>
          {isFailed ? '—' : sub.grades.final.toFixed(1)}
        </span>
        <VerdictMini verdict={sub.auth?.verdict || 'unverified'} />
        <span style={{ color: 'var(--ink-4)', fontSize: 10, transform: open ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 120ms' }}>›</span>
      </button>
      {open && (
        <div style={{ padding: '14px 18px', borderTop: '1px solid var(--line-2)', display: 'flex', flexDirection: 'column', gap: 14, background: 'var(--bg-0)' }}>
          {sub.grades && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 6 }}>
              <SubMini label="Centering" v={sub.grades.centering} />
              <SubMini label="Corners" v={sub.grades.corners} />
              <SubMini label="Edges" v={sub.grades.edges} />
              <SubMini label="Surface" v={sub.grades.surface} />
              <SubMini label="Confidence" v={(sub.grades.confidence * 100).toFixed(0) + '%'} accent />
            </div>
          )}
          {sub.notes && (
            <p style={{ margin: 0, fontSize: 12, color: 'var(--ink-2)', lineHeight: 1.5, padding: '8px 10px', background: 'var(--bg-1)', borderLeft: '2px solid var(--line)', borderRadius: '0 3px 3px 0' }}>
              <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.1em', textTransform: 'uppercase', marginRight: 8 }}>note</span>
              {sub.notes}
            </p>
          )}
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            {!isFailed && <button className="pg-btn pg-btn-ghost">View public cert ↗</button>}
            <button className="pg-btn pg-btn-ghost">View raw shots</button>
            {isFailed && <button className="pg-btn pg-btn-primary">Retry submission</button>}
          </div>
        </div>
      )}
    </div>
  );
}

function SubMini({ label, v, accent }) {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line-2)', borderRadius: 4, padding: '8px 10px' }}>
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontFamily: 'Instrument Serif, serif', fontSize: 22, color: accent ? 'var(--lime)' : 'var(--ink-0)', lineHeight: 1, marginTop: 3 }}>
        {typeof v === 'number' ? v.toFixed(1) : v}
      </div>
    </div>
  );
}

function VerdictMini({ verdict }) {
  const map = {
    authentic: { c: 'var(--lime)', l: 'auth' },
    suspicious: { c: 'var(--amber)', l: 'susp' },
    likely_counterfeit: { c: 'var(--rose)', l: 'cnft' },
    unverified: { c: 'var(--ink-4)', l: 'unv' },
  };
  const m = map[verdict] || map.unverified;
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5, fontSize: 10 }}>
      <span style={{ width: 5, height: 5, borderRadius: 3, background: m.c, boxShadow: '0 0 5px ' + m.c }} />
      <span className="mono" style={{ color: m.c, letterSpacing: '0.1em', textTransform: 'uppercase' }}>{m.l}</span>
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function ShotsTab({ dense }) {
  return (
    <section>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 12 }}>
        <p style={{ margin: 0, fontSize: 12, color: 'var(--ink-3)' }}>
          From submission <span className="mono" style={{ color: 'var(--ink-1)' }}>PG-2026-0F1A4C-9D71</span> · 8 shots · all passed quality gating
        </p>
        <button className="pg-btn pg-btn-ghost" style={{ fontSize: 11 }}>Open in viewer ↗</button>
      </div>
      <div style={{
        display: 'grid',
        gridTemplateColumns: dense ? 'repeat(8, 1fr)' : 'repeat(4, 1fr)',
        gap: 10,
      }}>
        {SHOT_KEYS.map((s, i) => <ShotThumb key={s.k} shot={s} dense={dense} />)}
      </div>
    </section>
  );
}

function ShotThumb({ shot, dense }) {
  return (
    <div style={{
      background: 'var(--bg-1)', border: '1px solid var(--line)',
      borderRadius: 5, padding: dense ? 6 : 10,
      display: 'flex', flexDirection: 'column', gap: dense ? 4 : 8,
    }}>
      <div style={{
        aspectRatio: '5/7', borderRadius: 3, position: 'relative', overflow: 'hidden',
        background: shot.k === 'flash' ? 'linear-gradient(135deg, #fde68a 0%, #f97316 100%)' :
                    shot.k === 'tilt' ? 'linear-gradient(135deg, #a78bfa 0%, #4d2410 100%)' :
                    'linear-gradient(135deg, #f97316 0%, #4d2410 100%)',
        border: '1px solid var(--line-2)',
      }}>
        {shot.k.startsWith('t') && shot.k.length === 2 && (
          <div style={{ position: 'absolute', top: '12%', left: '12%', right: '12%', bottom: '12%', border: '1px dashed rgba(255,255,255,0.2)' }} />
        )}
        {(shot.k === 'tl' || shot.k === 'tr' || shot.k === 'bl' || shot.k === 'br') && (
          <div style={{
            position: 'absolute', width: 16, height: 16, borderRadius: 8,
            border: '2px solid var(--lime)',
            top: shot.k.startsWith('t') ? '15%' : 'auto',
            bottom: shot.k.startsWith('b') ? '15%' : 'auto',
            left: shot.k.endsWith('l') ? '15%' : 'auto',
            right: shot.k.endsWith('r') ? '15%' : 'auto',
            background: 'rgba(190,242,100,0.15)',
          }} />
        )}
        <div style={{
          position: 'absolute', bottom: 0, left: 0, right: 0,
          padding: '3px 6px',
          background: 'linear-gradient(to top, rgba(0,0,0,0.7), transparent)',
          fontSize: 8, fontFamily: 'Geist Mono', color: 'var(--lime)', letterSpacing: '0.1em',
        }}>✓ pass</div>
      </div>
      {!dense && <div style={{ fontSize: 11, fontWeight: 500, color: 'var(--ink-1)' }}>{shot.t}</div>}
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', display: 'flex', justifyContent: 'space-between' }}>
        <span>{shot.k}</span>
        <span>{(0.92 + Math.random() * 0.05).toFixed(2)}</span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function ComparisonTab({ subs }) {
  const sorted = [...subs].sort((a, b) => new Date(b.completed_at) - new Date(a.completed_at));
  const cats = ['centering', 'corners', 'edges', 'surface', 'final'];

  return (
    <section>
      <p style={{ margin: '0 0 14px', fontSize: 12, color: 'var(--ink-3)' }}>
        Subgrade evolution across {sorted.length} successful submissions. Newest first. Bars are scaled 1.0–10.0.
      </p>
      <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6 }}>
        <div style={{
          display: 'grid', gridTemplateColumns: '160px repeat(' + cats.length + ', 1fr)',
          padding: '10px 16px', borderBottom: '1px solid var(--line)',
          gap: 12,
        }}>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>Submission</span>
          {cats.map(c => (
            <span key={c} className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{c}</span>
          ))}
        </div>
        {sorted.map((s, i) => (
          <div key={s.id} style={{
            display: 'grid', gridTemplateColumns: '160px repeat(' + cats.length + ', 1fr)',
            padding: '14px 16px',
            borderBottom: i < sorted.length - 1 ? '1px solid var(--line-2)' : 'none',
            gap: 12, alignItems: 'center',
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)' }}>
                {new Date(s.completed_at).toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' })}
              </span>
              <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)' }}>{s.id.slice(0, 22)}…</span>
            </div>
            {cats.map(c => <CompareBar key={c} v={s.grades[c]} accent={c === 'final'} />)}
          </div>
        ))}
      </div>
    </section>
  );
}

function CompareBar({ v, accent }) {
  const pct = ((v - 1) / 9) * 100;
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span className="mono" style={{ fontSize: 12, color: accent ? 'var(--lime)' : 'var(--ink-1)', fontWeight: accent ? 600 : 400, fontVariantNumeric: 'tabular-nums' }}>
        {v.toFixed(1)}
      </span>
      <div style={{ height: 4, background: 'var(--bg-3)', borderRadius: 1, overflow: 'hidden' }}>
        <div style={{ width: pct + '%', height: '100%', background: accent ? 'var(--lime)' : 'var(--ink-4)' }} />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function FlagsTab({ subs }) {
  const flags = [
    { kind: 'corner', side: 'TR', detail: 'Whitening detected · stable across 3 grades — not getting worse', severity: 'minor', first_seen: '2025-11-02', last_seen: '2026-05-07' },
    { kind: 'edge', side: 'bottom', detail: 'Light whitening · 12% of edge length · consistent across recent grades', severity: 'minor', first_seen: '2025-11-02', last_seen: '2026-05-07' },
  ];
  const dot = { ok: 'var(--lime)', minor: 'var(--amber)', major: 'var(--rose)' };
  return (
    <section style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <p style={{ margin: 0, fontSize: 12, color: 'var(--ink-3)' }}>
        Persistent flags across submissions — defects seen in 2+ grades suggest real wear vs capture noise.
      </p>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {flags.map((f, i) => (
          <div key={i} style={{
            background: 'var(--bg-1)', border: '1px solid var(--line)',
            borderLeft: '3px solid ' + dot[f.severity],
            borderRadius: 4, padding: '12px 16px',
            display: 'grid', gridTemplateColumns: '24px 130px 1fr 200px', gap: 14, alignItems: 'center',
          }}>
            <span style={{ width: 10, height: 10, borderRadius: 5, background: dot[f.severity], boxShadow: '0 0 6px ' + dot[f.severity] }} />
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              {f.kind} · {f.side}
            </span>
            <span style={{ fontSize: 13, color: 'var(--ink-1)' }}>{f.detail}</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', textAlign: 'right' }}>
              first {f.first_seen} → last {f.last_seen}
            </span>
          </div>
        ))}
      </div>
      <div style={{ padding: '14px 18px', background: 'rgba(190,242,100,0.03)', border: '1px solid rgba(190,242,100,0.15)', borderRadius: 6 }}>
        <div className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.18em', textTransform: 'uppercase', marginBottom: 6 }}>
          Suggestion
        </div>
        <p style={{ margin: 0, fontSize: 13, color: 'var(--ink-1)', lineHeight: 1.5 }}>
          Both flags are stable — the card hasn't degraded since first grading. Fine to leave at PSA 9.0 unless you want a regrade after a deep clean.
        </p>
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function SectionLabel({ children, hint }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', borderBottom: '1px solid var(--line-2)', paddingBottom: 6 }}>
      <h2 className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', letterSpacing: '0.18em', textTransform: 'uppercase', margin: 0, fontWeight: 500 }}>{children}</h2>
      {hint && <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>{hint}</span>}
    </div>
  );
}

function CardTweaks({ t, setTweak }) {
  return (
    <TweaksPanel>
      <TweakSection label="View" />
      <TweakRadio label="Tab" value={t.tab}
        options={['history','shots','comparison','flags']}
        onChange={v => setTweak('tab', v)} />
      <TweakToggle label="Grade timeline" value={t.showTimeline}
        onChange={v => setTweak('showTimeline', v)} />
      <TweakToggle label="Dense shot grid" value={t.denseShots}
        onChange={v => setTweak('denseShots', v)} />
    </TweaksPanel>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
