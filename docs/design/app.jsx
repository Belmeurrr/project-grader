// app.jsx — The Workbench
// Sidebar (collection nav, filters, system telemetry) + main canvas (collection
// grid + focused inspector). Layout variants exposed via Tweaks.

const { useState, useMemo, useEffect } = React;

// Read tweak defaults from the EDITMODE block parsed in workbench.html
const DEFAULTS = window.__TWEAK_DEFAULTS_RAW || { layout: 'grid', density: 'comfy', showInspector: true };

function App() {
  const [t, setTweak] = useTweaks(DEFAULTS);
  const [selectedId, setSelectedId] = useState(window.COLLECTION[0].id);
  const [filter, setFilter] = useState('all');
  const [query, setQuery] = useState('');

  const selected = useMemo(
    () => window.COLLECTION.find(c => c.id === selectedId) || window.COLLECTION[0],
    [selectedId]
  );

  const visible = useMemo(() => {
    let list = window.COLLECTION;
    if (filter === 'flagged') list = list.filter(c => c.flag || c.auth !== 'authentic');
    if (filter === 'top') list = list.filter(c => c.final !== null && c.final >= 9);
    if (filter === 'pending') list = list.filter(c => c.corners === null || c.surface === null);
    if (query) {
      const q = query.toLowerCase();
      list = list.filter(c =>
        c.name.toLowerCase().includes(q) ||
        c.set.toLowerCase().includes(q) ||
        c.id.toLowerCase().includes(q)
      );
    }
    return list;
  }, [filter, query]);

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '260px 1fr',
      minHeight: '100vh',
      background: 'var(--bg-0)',
    }}>
      <Sidebar
        filter={filter} setFilter={setFilter}
        query={query} setQuery={setQuery}
        count={visible.length}
      />
      <Main
        layout={t.layout}
        density={t.density}
        showInspector={t.showInspector}
        cards={visible}
        selected={selected}
        selectedId={selectedId}
        onSelect={setSelectedId}
      />
      <WorkbenchTweaks t={t} setTweak={setTweak} />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Sidebar({ filter, setFilter, query, setQuery, count }) {
  const stats = window.STATS;
  return (
    <aside style={{
      borderRight: '1px solid var(--line)',
      background: 'var(--bg-1)',
      display: 'flex', flexDirection: 'column',
      position: 'sticky', top: 0, height: '100vh', overflow: 'hidden',
    }}>
      {/* Brand */}
      <div style={{
        padding: '18px 20px',
        borderBottom: '1px solid var(--line-2)',
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <div style={{
          width: 22, height: 22, borderRadius: 4,
          background: 'var(--lime)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          color: '#0a0a0a', fontWeight: 700, fontSize: 13,
          boxShadow: '0 0 18px rgba(190,242,100,0.4)',
        }}>◰</div>
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.1 }}>
          <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink-0)', letterSpacing: '-0.01em' }}>Project Grader</span>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>The Workbench</span>
        </div>
      </div>

      {/* Search */}
      <div style={{ padding: '14px 16px 8px' }}>
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8,
          background: 'var(--bg-2)', border: '1px solid var(--line)',
          borderRadius: 6, padding: '7px 10px',
        }}>
          <span style={{ color: 'var(--ink-4)', fontSize: 12 }}>⌕</span>
          <input
            value={query}
            onChange={e => setQuery(e.target.value)}
            placeholder="Search by name, set, cert…"
            style={{
              flex: 1, background: 'transparent', border: 0, outline: 'none',
              color: 'var(--ink-1)', fontSize: 12, fontFamily: 'inherit',
            }}
          />
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>⌘K</span>
        </div>
      </div>

      {/* Filter nav */}
      <nav style={{ padding: '8px 8px', display: 'flex', flexDirection: 'column', gap: 1 }}>
        <SidebarItem label="All graded" count={stats.total} active={filter === 'all'} onClick={() => setFilter('all')} />
        <SidebarItem label="Top tier (≥ 9)" count={6} active={filter === 'top'} onClick={() => setFilter('top')} />
        <SidebarItem label="Pending heads" count={stats.pendingHeads} active={filter === 'pending'} onClick={() => setFilter('pending')} tone="amber" />
        <SidebarItem label="Flagged" count={stats.flagged} active={filter === 'flagged'} onClick={() => setFilter('flagged')} tone="rose" />
      </nav>

      <div style={{ padding: '8px 16px 4px' }}>
        <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.18em', marginBottom: 8 }}>By scheme</div>
        <SidebarItem label="PSA · 1–10" count={12} thin />
        <SidebarItem label="BGS · subgrade" count={0} thin disabled />
        <SidebarItem label="SGC · 1–10" count={0} thin disabled />
      </div>

      <div style={{ padding: '8px 16px 4px' }}>
        <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.18em', marginBottom: 8 }}>Game</div>
        <SidebarItem label="Pokémon TCG" count={6} thin />
        <SidebarItem label="Magic: the Gathering" count={3} thin />
        <SidebarItem label="Yu-Gi-Oh!" count={2} thin />
        <SidebarItem label="Sports" count={2} thin />
      </div>

      {/* System telemetry, anchored to bottom */}
      <div style={{ marginTop: 'auto', borderTop: '1px solid var(--line-2)', padding: '14px 16px' }}>
        <div className="mono" style={{
          fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.18em',
          color: 'var(--ink-4)', marginBottom: 10,
        }}>System</div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 7 }}>
          <SystemRow label="API" value="ok · 38ms p95" tone="ok" />
          <SystemRow label="Worker" value="solo · idle" tone="ok" />
          <SystemRow label="Detectors" value="7/7 wired" tone="ok" />
          <SystemRow label="Catalog" value={stats.catalogSize} />
          <SystemRow label="Flywheel" value={stats.lastFlywheel} />
        </div>
      </div>
    </aside>
  );
}

function SidebarItem({ label, count, active, onClick, tone, thin, disabled }) {
  const dotColor = tone === 'amber' ? 'var(--amber)' : tone === 'rose' ? 'var(--rose)' : null;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: thin ? '5px 10px' : '7px 10px',
        background: active ? 'rgba(190,242,100,0.08)' : 'transparent',
        border: 0, borderLeft: active ? '2px solid var(--lime)' : '2px solid transparent',
        borderRadius: 4, cursor: disabled ? 'default' : 'pointer',
        color: disabled ? 'var(--ink-5)' : active ? 'var(--ink-0)' : 'var(--ink-2)',
        fontSize: thin ? 11 : 12,
        fontFamily: 'inherit', textAlign: 'left',
        transition: 'background 120ms',
        opacity: disabled ? 0.4 : 1,
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        {dotColor && <span style={{ width: 5, height: 5, borderRadius: 3, background: dotColor }} />}
        {label}
      </span>
      <span className="mono" style={{
        fontSize: 10, color: active ? 'var(--lime)' : 'var(--ink-4)',
        fontVariantNumeric: 'tabular-nums',
      }}>{count}</span>
    </button>
  );
}

function SystemRow({ label, value, tone }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.04em' }}>{label}</span>
      <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {tone === 'ok' && <span style={{ width: 5, height: 5, borderRadius: 3, background: 'var(--lime)', boxShadow: '0 0 6px var(--lime)' }} />}
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-1)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Main({ layout, density, showInspector, cards, selected, selectedId, onSelect }) {
  return (
    <main style={{ display: 'flex', flexDirection: 'column', minWidth: 0 }}>
      <Hero />
      <CollectionStats />
      <div style={{
        display: 'grid',
        gridTemplateColumns: showInspector ? 'minmax(0, 1fr) 480px' : 'minmax(0, 1fr)',
        gap: 0,
        borderTop: '1px solid var(--line)',
      }}>
        <CollectionView
          layout={layout}
          density={density}
          cards={cards}
          selectedId={selectedId}
          onSelect={onSelect}
        />
        {showInspector && <Inspector card={selected} />}
      </div>
      <Footer />
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Topbar() {
  return (
    <div style={{
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      padding: '12px 24px', borderBottom: '1px solid var(--line)',
      background: 'var(--bg-0)',
      position: 'sticky', top: 0, zIndex: 5,
      backdropFilter: 'blur(8px)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
        <span className="mono" style={{
          fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase',
        }}>Workbench / Collection</span>
        <span style={{ width: 1, height: 12, background: 'var(--line)' }} />
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <span style={{ width: 6, height: 6, borderRadius: 3, background: 'var(--lime)', boxShadow: '0 0 6px var(--lime)' }} />
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)' }}>pipeline online · 7 detectors</span>
        </span>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <button style={btnGhost}>Import shots</button>
        <button style={btnPrimary}>
          <span style={{ marginRight: 6, fontSize: 13 }}>+</span>New submission
        </button>
      </div>
    </div>
  );
}

const btnPrimary = {
  display: 'inline-flex', alignItems: 'center',
  background: 'var(--lime)', color: '#0a0a0a',
  border: 0, borderRadius: 5, padding: '7px 12px',
  fontSize: 12, fontWeight: 600, cursor: 'pointer',
  letterSpacing: '-0.005em',
  boxShadow: '0 0 0 1px rgba(0,0,0,0.2), 0 0 16px rgba(190,242,100,0.25)',
};
const btnGhost = {
  background: 'var(--bg-2)', color: 'var(--ink-1)',
  border: '1px solid var(--line)', borderRadius: 5,
  padding: '6px 11px', fontSize: 12, cursor: 'pointer', fontWeight: 500,
};

// ─────────────────────────────────────────────────────────────────────────────
// Hero — animated 8-shot capture sequence
function Hero() {
  const [shotIdx, setShotIdx] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const tick = setInterval(() => {
      setProgress(p => {
        if (p >= 100) {
          setShotIdx(i => (i + 1) % window.SHOT_PROTOCOL.length);
          return 0;
        }
        return p + 4;
      });
    }, 60);
    return () => clearInterval(tick);
  }, []);

  return (
    <section style={{
      padding: '36px 24px 28px',
      borderBottom: '1px solid var(--line)',
      background: 'radial-gradient(ellipse 800px 300px at 30% 0%, rgba(190,242,100,0.04), transparent 70%), var(--bg-0)',
    }}>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 480px', gap: 32, alignItems: 'center' }}>
        {/* Left: framing */}
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
            <span className="mono" style={{
              fontSize: 10, color: 'var(--lime)', letterSpacing: '0.22em',
              textTransform: 'uppercase', padding: '3px 8px',
              background: 'rgba(190,242,100,0.08)', border: '1px solid rgba(190,242,100,0.25)',
              borderRadius: 3,
            }}>Personal · v0.4.1</span>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.06em' }}>
              not a service · graded by you, for you
            </span>
          </div>
          <h1 style={{
            margin: 0,
            fontSize: 56, fontWeight: 500, lineHeight: 1.0, letterSpacing: '-0.025em',
            color: 'var(--ink-0)',
            maxWidth: 600,
          }}>
            A grading bench<br />
            for one collector,<br />
            <span className="serif" style={{ color: 'var(--lime)', fontWeight: 400 }}>built like a lab.</span>
          </h1>
          <p style={{
            margin: '20px 0 0 0', maxWidth: 520, fontSize: 14, lineHeight: 1.6,
            color: 'var(--ink-2)',
          }}>
            Eight calibrated shots per card · seven counterfeit detectors ·
            calibrated uncertainty on every grade. No SLAs. No marketplace.
            Just an honest readout of what the cards in your binder really are.
          </p>
          <div style={{
            display: 'flex', gap: 32, marginTop: 28,
            paddingTop: 20, borderTop: '1px solid var(--line-2)',
          }}>
            <MetricCell label="Cards graded" value="12" sub="this binder" />
            <MetricCell label="Avg final" value="8.4" tone="var(--lime)" sub="PSA scheme" />
            <MetricCell label="Detectors" value="7/7" sub="ensemble wired" />
            <MetricCell label="Reference catalog" value="1.28M" sub="DinoV2 + pgvector" />
          </div>
        </div>

        {/* Right: animated capture sequence */}
        <CaptureSequence shotIdx={shotIdx} progress={progress} />
      </div>
    </section>
  );
}

function CaptureSequence({ shotIdx, progress }) {
  const shots = window.SHOT_PROTOCOL;
  const current = shots[shotIdx];
  return (
    <div style={{
      background: 'var(--bg-1)', border: '1px solid var(--line)',
      borderRadius: 10, padding: 18,
      boxShadow: '0 24px 60px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.02)',
    }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{
            width: 8, height: 8, borderRadius: 4, background: 'var(--lime)',
            boxShadow: '0 0 8px var(--lime)', animation: 'pulse 1.6s ease-in-out infinite',
          }} />
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-2)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Capture · live
          </span>
        </div>
        <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>
          {String(shotIdx + 1).padStart(2, '0')} / {String(shots.length).padStart(2, '0')}
        </span>
      </div>

      {/* Card with reticle */}
      <div style={{
        position: 'relative', aspectRatio: '5/7', maxWidth: 280, margin: '0 auto',
        background: 'linear-gradient(135deg, #f97316 0%, #f9731620 60%, #0a0a0a 100%)22',
      }}>
        <div style={{
          position: 'absolute', inset: 0,
          borderRadius: 10, overflow: 'hidden',
          background: 'linear-gradient(135deg, #f9731622 0%, #f9731608 60%, #0a0a0a 100%)',
          border: '1px solid var(--line)',
        }}>
          <div style={{
            position: 'absolute', inset: 0,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 100, color: '#f97316', opacity: 0.5,
            filter: 'drop-shadow(0 0 24px #f9731688)',
          }}>🜂</div>
          {/* scan line */}
          <div style={{
            position: 'absolute', left: 0, right: 0,
            top: progress + '%', height: 2,
            background: 'linear-gradient(90deg, transparent, var(--lime), transparent)',
            boxShadow: '0 0 12px var(--lime)',
            opacity: 0.8,
          }} />
        </div>
        {/* Reticle */}
        <Reticle current={current} progress={progress} />
      </div>

      {/* Shot strip */}
      <div style={{
        display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)',
        gap: 4, marginTop: 14,
      }}>
        {shots.map((s, i) => {
          const done = i < shotIdx;
          const active = i === shotIdx;
          return (
            <div key={s.k} style={{
              aspectRatio: '5/7',
              border: '1px solid ' + (active ? 'var(--lime)' : done ? 'var(--lime-deep)' : 'var(--line)'),
              borderRadius: 3,
              background: active
                ? 'rgba(190,242,100,0.08)'
                : done ? 'rgba(190,242,100,0.04)' : 'var(--bg-2)',
              position: 'relative',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              transition: 'all 200ms',
            }}>
              <span className="mono" style={{
                fontSize: 9, color: active ? 'var(--lime)' : done ? 'var(--lime-deep)' : 'var(--ink-4)',
                textTransform: 'uppercase', letterSpacing: '0.04em',
              }}>{s.k}</span>
              {done && (
                <span style={{
                  position: 'absolute', top: 2, right: 2,
                  fontSize: 8, color: 'var(--lime-deep)',
                }}>✓</span>
              )}
            </div>
          );
        })}
      </div>

      {/* Current shot meta */}
      <div style={{
        marginTop: 14, padding: 10,
        background: 'var(--bg-2)', border: '1px solid var(--line-2)', borderRadius: 5,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div>
          <div style={{ fontSize: 12, color: 'var(--ink-1)', fontWeight: 500 }}>{current.t}</div>
          <div style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 2 }}>{current.d}</div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.04em' }}>
            QUALITY · {(0.82 + Math.sin(progress / 30) * 0.06).toFixed(2)}
          </div>
          <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)' }}>
            blur·glare·perspective
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%,100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.85); }
        }
      `}</style>
    </div>
  );
}

function Reticle({ current, progress }) {
  return (
    <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}>
      <defs>
        <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(190,242,100,0.06)" strokeWidth="0.5" />
        </pattern>
      </defs>
      <rect width="100%" height="100%" fill="url(#grid)" />
      {/* Corner reticle marks */}
      {[[8,8,'tl'],[null,8,'tr'],[8,null,'bl'],[null,null,'br']].map(([x,y,k],i) => {
        const isCurrentCorner = current.k === k;
        const stroke = isCurrentCorner ? 'var(--lime)' : 'rgba(255,255,255,0.3)';
        const sw = isCurrentCorner ? 2 : 1;
        const props = {};
        if (x !== null) props.left = x; else props.right = 8;
        if (y !== null) props.top = y; else props.bottom = 8;
        return (
          <g key={i}>
            <line x1={x !== null ? x : '95%'} y1={y !== null ? y : '95%'}
              x2={x !== null ? x + 14 : '95%'} y2={y !== null ? y : '95%'}
              stroke={stroke} strokeWidth={sw} />
            <line x1={x !== null ? x : '95%'} y1={y !== null ? y : '95%'}
              x2={x !== null ? x : '95%'} y2={y !== null ? y + 14 : '95%'}
              stroke={stroke} strokeWidth={sw} />
          </g>
        );
      })}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CollectionStats() {
  const stats = window.STATS;
  return (
    <section style={{
      padding: '20px 24px',
      borderBottom: '1px solid var(--line)',
      display: 'grid',
      gridTemplateColumns: 'repeat(6, 1fr)',
      gap: 1,
      background: 'var(--line-2)',
    }}>
      {[
        { l: 'Population', v: stats.total, s: 'cards graded' },
        { l: 'Distribution', v: '8.4', s: 'mean · σ 0.81', tone: 'var(--lime)' },
        { l: 'Highest', v: '9.5', s: 'Blue-Eyes (1st Ed.)', tone: 'var(--lime-bright)' },
        { l: 'Lowest', v: '6.5', s: 'Mantle (1952)', tone: 'var(--amber)' },
        { l: 'Auth flagged', v: '1', s: 'Pikachu Illustrator', tone: 'var(--amber)' },
        { l: 'Pending heads', v: '2', s: 'corners · surface', tone: 'var(--ink-2)' },
      ].map(s => (
        <div key={s.l} style={{
          background: 'var(--bg-0)',
          padding: '14px 16px',
          display: 'flex', flexDirection: 'column', gap: 4,
        }}>
          <span className="mono" style={{
            fontSize: 9, textTransform: 'uppercase', letterSpacing: '0.18em',
            color: 'var(--ink-4)',
          }}>{s.l}</span>
          <span style={{
            fontSize: 22, fontWeight: 500, color: s.tone || 'var(--ink-0)',
            letterSpacing: '-0.02em', lineHeight: 1, fontVariantNumeric: 'tabular-nums',
          }}>{s.v}</span>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>{s.s}</span>
        </div>
      ))}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function CollectionView({ layout, density, cards, selectedId, onSelect }) {
  return (
    <div style={{
      padding: '16px 24px 32px',
      borderRight: '1px solid var(--line)',
      background: 'var(--bg-0)',
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
        <span className="mono" style={{
          fontSize: 10, textTransform: 'uppercase', letterSpacing: '0.18em',
          color: 'var(--ink-3)',
        }}>{cards.length} cards · sorted by capture</span>
        <div style={{ display: 'flex', gap: 4 }}>
          {['grid','list','sheet','rows'].map(L => (
            <span key={L} className="mono" style={{
              fontSize: 9, padding: '3px 7px',
              background: layout === L ? 'rgba(190,242,100,0.1)' : 'transparent',
              color: layout === L ? 'var(--lime)' : 'var(--ink-4)',
              border: '1px solid ' + (layout === L ? 'rgba(190,242,100,0.25)' : 'var(--line-2)'),
              borderRadius: 3, textTransform: 'uppercase', letterSpacing: '0.1em',
            }}>{L}</span>
          ))}
        </div>
      </div>

      {layout === 'grid' && <GridView cards={cards} density={density} selectedId={selectedId} onSelect={onSelect} />}
      {layout === 'list' && <ListView cards={cards} selectedId={selectedId} onSelect={onSelect} />}
      {layout === 'sheet' && <SheetView cards={cards} selectedId={selectedId} onSelect={onSelect} />}
      {layout === 'rows' && <RowsView cards={cards} selectedId={selectedId} onSelect={onSelect} />}
    </div>
  );
}

function GridView({ cards, density, selectedId, onSelect }) {
  const cols = density === 'compact' ? 3 : density === 'comfy' ? 2 : 2;
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gap: 14 }}>
      {cards.map(c => (
        <button
          key={c.id} onClick={() => onSelect(c.id)}
          style={{
            background: 'var(--bg-1)', border: '1px solid ' + (selectedId === c.id ? 'rgba(190,242,100,0.5)' : 'var(--line)'),
            borderRadius: 8, padding: 14, cursor: 'pointer', textAlign: 'left',
            fontFamily: 'inherit', color: 'inherit',
            transition: 'border-color 120ms, background 120ms',
            boxShadow: selectedId === c.id ? '0 0 0 1px rgba(190,242,100,0.2), 0 8px 24px rgba(190,242,100,0.06)' : 'none',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <CertID id={c.id} size={9} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
              <VerdictDot verdict={c.auth} />
              <span className="mono" style={{ fontSize: 9, color: VERDICT_COLOR[c.auth], textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {VERDICT_LABEL[c.auth]}
              </span>
            </div>
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            <CardArt card={c} size="md" />
            <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 8 }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink-0)', letterSpacing: '-0.005em', lineHeight: 1.15 }}>
                  {c.name}
                </div>
                <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 3, letterSpacing: '0.02em' }}>
                  {c.set} · {c.year}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
                <GradeBadge grade={c.final} size="lg" />
                {c.final !== null && (
                  <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>
                    ±{((c.upper - c.lower) / 2).toFixed(2)}
                  </span>
                )}
              </div>
              {c.final !== null ? (
                <div style={{ height: 2, background: 'var(--bg-3)', borderRadius: 1, overflow: 'hidden' }}>
                  <div style={{ width: (c.final / 10 * 100) + '%', height: '100%', background: gradeTone(c.final) }} />
                </div>
              ) : (
                <div style={{ fontSize: 10, color: 'var(--amber)', fontFamily: 'Geist Mono' }}>
                  Preliminary
                </div>
              )}
              <div className="mono" style={{
                fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.04em',
                display: 'flex', justifyContent: 'space-between',
              }}>
                <span>1 of {c.pop.total}</span>
                <span>#{c.pop.rank} rank</span>
              </div>
            </div>
          </div>
        </button>
      ))}
    </div>
  );
}

function ListView({ cards, selectedId, onSelect }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1, background: 'var(--line-2)', border: '1px solid var(--line-2)' }}>
      <div style={{
        display: 'grid', gridTemplateColumns: '120px 1fr 100px 90px 90px 90px 90px',
        gap: 12, padding: '8px 14px', background: 'var(--bg-0)',
      }}>
        {['Cert','Card','Final','Center','Corners','Edges','Auth'].map(h => (
          <span key={h} className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.18em' }}>{h}</span>
        ))}
      </div>
      {cards.map(c => (
        <button key={c.id} onClick={() => onSelect(c.id)} style={{
          display: 'grid', gridTemplateColumns: '120px 1fr 100px 90px 90px 90px 90px',
          gap: 12, padding: '12px 14px',
          background: selectedId === c.id ? 'rgba(190,242,100,0.05)' : 'var(--bg-1)',
          border: 0, cursor: 'pointer', textAlign: 'left',
          color: 'inherit', fontFamily: 'inherit', alignItems: 'center',
        }}>
          <CertID id={c.id} size={10} />
          <div>
            <div style={{ fontSize: 13, color: 'var(--ink-0)', fontWeight: 500 }}>{c.name}</div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', marginTop: 2 }}>{c.set} · {c.number}</div>
          </div>
          <GradeBadge grade={c.final} size="md" />
          <span className="mono" style={{ color: 'var(--ink-1)', fontSize: 12 }}>{c.centering?.toFixed(1) ?? '—'}</span>
          <span className="mono" style={{ color: c.corners === null ? 'var(--ink-4)' : 'var(--ink-1)', fontSize: 12 }}>{c.corners?.toFixed(1) ?? '—'}</span>
          <span className="mono" style={{ color: 'var(--ink-1)', fontSize: 12 }}>{c.edges?.toFixed(1) ?? '—'}</span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <VerdictDot verdict={c.auth} />
            <span className="mono" style={{ fontSize: 10, color: VERDICT_COLOR[c.auth], textTransform: 'uppercase' }}>
              {VERDICT_LABEL[c.auth]}
            </span>
          </div>
        </button>
      ))}
    </div>
  );
}

function SheetView({ cards, selectedId, onSelect }) {
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8,
      background: 'var(--bg-2)', border: '1px solid var(--line)',
      padding: 10, borderRadius: 4,
    }}>
      {cards.map(c => (
        <button key={c.id} onClick={() => onSelect(c.id)} style={{
          padding: 0, background: 'transparent', border: 0, cursor: 'pointer',
          display: 'flex', flexDirection: 'column', gap: 6, alignItems: 'center',
        }}>
          <div style={{
            position: 'relative',
            outline: selectedId === c.id ? '2px solid var(--lime)' : 'none',
            outlineOffset: 2,
            borderRadius: 4,
          }}>
            <CardArt card={c} size="md" />
            <div style={{
              position: 'absolute', top: 4, right: 4,
              padding: '2px 5px', background: 'rgba(0,0,0,0.7)',
              borderRadius: 3, backdropFilter: 'blur(4px)',
            }}>
              <span className="mono" style={{ fontSize: 9, color: gradeTone(c.final), fontWeight: 600 }}>
                {c.final?.toFixed(1) ?? '—'}
              </span>
            </div>
          </div>
          <CertID id={c.id} size={9} />
        </button>
      ))}
    </div>
  );
}

function RowsView({ cards, selectedId, onSelect }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {cards.map(c => (
        <button key={c.id} onClick={() => onSelect(c.id)} style={{
          background: selectedId === c.id ? 'rgba(190,242,100,0.05)' : 'var(--bg-1)',
          border: '1px solid ' + (selectedId === c.id ? 'rgba(190,242,100,0.3)' : 'var(--line)'),
          borderRadius: 6, padding: 14, cursor: 'pointer', textAlign: 'left',
          fontFamily: 'inherit', color: 'inherit',
          display: 'grid', gridTemplateColumns: '64px 1fr auto auto', gap: 16, alignItems: 'center',
        }}>
          <CardArt card={c} size="sm" />
          <div>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: 'var(--ink-0)' }}>{c.name}</span>
              <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>{c.set} · {c.number} · {c.year}</span>
            </div>
            <CertID id={c.id} size={10} />
          </div>
          <div style={{ display: 'flex', gap: 14 }}>
            {[['Ctr',c.centering],['Cor',c.corners],['Edg',c.edges],['Sur',c.surface]].map(([l,v]) => (
              <div key={l} style={{ textAlign: 'center' }}>
                <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>{l}</div>
                <div className="mono" style={{ fontSize: 12, color: v === null ? 'var(--ink-4)' : 'var(--ink-1)', marginTop: 2 }}>{v?.toFixed(1) ?? '—'}</div>
              </div>
            ))}
          </div>
          <GradeBadge grade={c.final} size="lg" />
        </button>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Inspector({ card }) {
  return (
    <aside style={{
      background: 'var(--bg-1)', padding: '18px 22px 32px',
      display: 'flex', flexDirection: 'column', gap: 22,
      maxHeight: 'calc(100vh)', overflow: 'auto',
      position: 'sticky', top: 49, alignSelf: 'flex-start',
    }}>
      {/* Identity */}
      <div>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
          <span className="mono" style={{
            fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase',
          }}>Inspector · cert</span>
          <button style={{ ...btnGhost, padding: '4px 8px', fontSize: 10 }}>Open cert ↗</button>
        </div>
        <CertID id={card.id} size={11} />
        <div style={{ marginTop: 10, display: 'flex', gap: 14 }}>
          <CardArt card={card} size="lg" />
          <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ fontSize: 22, fontWeight: 600, color: 'var(--ink-0)', letterSpacing: '-0.015em', lineHeight: 1.1 }}>
              {card.name}
            </div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--ink-2)' }}>
              {card.set} · {card.number} · {card.year}
            </div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>
              {card.variant}
            </div>
            <div style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.18em' }}>Captured</div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--ink-1)' }}>{card.captured}</div>
            </div>
          </div>
        </div>
        {card.flag && (
          <div style={{
            marginTop: 12, padding: '8px 12px',
            background: 'rgba(251,191,36,0.06)', border: '1px solid rgba(251,191,36,0.25)',
            borderRadius: 4, fontSize: 11, color: 'var(--amber)', lineHeight: 1.5,
          }}>
            ⚠ {card.flag}
          </div>
        )}
      </div>

      {/* Subgrades */}
      <div>
        <SectionHead label="Subgrades · PSA" meta="psa-1to10" />
        <SubgradeBlock card={card} />
        <div style={{ marginTop: 14 }}>
          <ConfidenceBand
            final={card.final}
            lower={card.lower ?? 0}
            upper={card.upper ?? 0}
            conf={card.conf}
          />
        </div>
      </div>

      {/* Authenticity */}
      <div>
        <SectionHead label="Authenticity · ensemble" meta={`${(card.authConf * 100).toFixed(0)}% conf`} />
        <div style={{
          padding: '10px 12px',
          background: 'rgba(190,242,100,0.04)',
          border: '1px solid ' + VERDICT_COLOR[card.auth] + '40',
          borderRadius: 4,
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: 14,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <VerdictDot verdict={card.auth} />
            <span style={{
              fontSize: 13, fontWeight: 600, color: VERDICT_COLOR[card.auth],
              letterSpacing: '-0.005em',
            }}>
              {VERDICT_LABEL[card.auth]}
            </span>
          </div>
          <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>
            7 detectors · majority vote
          </span>
        </div>
        {card.detectors.length > 0 ? (
          <div>
            <div style={{
              display: 'grid', gridTemplateColumns: '120px 1fr 60px 60px 70px',
              gap: 12, padding: '4px 0',
              borderBottom: '1px solid var(--line-2)',
            }}>
              {['Detector','Score','Value','Δ','Verdict'].map(h => (
                <span key={h} className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.16em',
                  textAlign: ['Value','Δ','Verdict'].includes(h) ? 'right' : 'left' }}>{h}</span>
              ))}
            </div>
            {card.detectors.map(d => <DetectorRow key={d.k} d={d} />)}
          </div>
        ) : (
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', padding: '8px 0' }}>
            Detector telemetry not displayed for this view.
          </div>
        )}
      </div>

      {/* Defects */}
      {card.defects && card.defects.length > 0 && (
        <div>
          <SectionHead label="Defect log · DINGS-style" meta={`${card.defects.length} entries`} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {card.defects.map((d, i) => (
              <div key={i} style={{
                display: 'grid', gridTemplateColumns: 'auto 1fr auto',
                gap: 10, padding: '8px 10px', alignItems: 'center',
                background: 'var(--bg-2)', borderLeft: '2px solid ' + SEV_COLOR[d.severity],
                borderRadius: 2,
              }}>
                <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {d.region}
                </span>
                <span style={{ fontSize: 11, color: 'var(--ink-1)' }}>{d.note}</span>
                <span className="mono" style={{
                  fontSize: 9, color: SEV_COLOR[d.severity], textTransform: 'uppercase', letterSpacing: '0.1em',
                }}>{d.severity}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Population */}
      <div>
        <SectionHead label="Population · variant" meta="across this binder" />
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 1, background: 'var(--line-2)', border: '1px solid var(--line-2)' }}>
          <PopCell label="Total" v={card.pop.total} sub="graded" />
          <PopCell label="Rank" v={`#${card.pop.rank}`} sub="highest" tone="var(--lime)" />
          <PopCell label="Order" v={`#${card.pop.chrono}`} sub="chronological" />
        </div>
      </div>

      {/* Pipeline trace */}
      <div>
        <SectionHead label="Pipeline trace" meta="latest run" />
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {[
            ['detect+dewarp', '0.42s', 'YOLOv11-seg → 750×1050'],
            ['identification', '0.61s', 'DinoV2 + pgvector NN'],
            ['centering', '0.04s', 'geometric'],
            ['edges (MAD)', '0.18s', 'worse-face dominates'],
            ['counterfeit · 7-det', '1.84s', 'rosette · holo · knn · typo · sub · emb · cielab'],
            ['compose grade', '0.02s', 'subgrade min-merge'],
          ].map(([stage, t, sub]) => (
            <div key={stage} style={{
              display: 'grid', gridTemplateColumns: '170px 60px 1fr',
              gap: 10, fontSize: 10, color: 'var(--ink-3)',
              padding: '4px 0', borderBottom: '1px solid var(--line-2)',
            }}>
              <span className="mono" style={{ color: 'var(--ink-1)' }}>{stage}</span>
              <span className="mono" style={{ color: 'var(--lime)', textAlign: 'right' }}>{t}</span>
              <span className="mono" style={{ color: 'var(--ink-3)' }}>{sub}</span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

function PopCell({ label, v, sub, tone }) {
  return (
    <div style={{ background: 'var(--bg-1)', padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: 4 }}>
      <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', textTransform: 'uppercase', letterSpacing: '0.16em' }}>{label}</span>
      <span style={{ fontSize: 20, fontWeight: 500, color: tone || 'var(--ink-0)', fontVariantNumeric: 'tabular-nums', lineHeight: 1 }}>{v}</span>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>{sub}</span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Footer() {
  return (
    <footer style={{
      padding: '20px 24px', borderTop: '1px solid var(--line)',
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      background: 'var(--bg-1)',
    }}>
      <div style={{ display: 'flex', gap: 18 }}>
        {Object.entries(window.STATS.modelVersions).map(([k, v]) => (
          <span key={k} className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>
            <span style={{ color: 'var(--ink-4)' }}>{k}</span>={v === 'stub' ? <span style={{ color: 'var(--amber)' }}>{v}</span> : <span style={{ color: 'var(--ink-1)' }}>{v}</span>}
          </span>
        ))}
      </div>
      <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)' }}>
        not affiliated with PSA, BGS, SGC, or CGC
      </span>
    </footer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function WorkbenchTweaks({ t, setTweak }) {
  return (
    <TweaksPanel>
      <TweakSection label="Layout" />
      <TweakRadio label="View"
        value={t.layout}
        options={['grid','list','sheet','rows']}
        onChange={v => setTweak('layout', v)} />
      <TweakRadio label="Density"
        value={t.density}
        options={['compact','comfy','spacious']}
        onChange={v => setTweak('density', v)} />
      <TweakToggle label="Show inspector"
        value={t.showInspector}
        onChange={v => setTweak('showInspector', v)} />
    </TweaksPanel>
  );
}

// Mount
ReactDOM.createRoot(document.getElementById('root')).render(<App />);
