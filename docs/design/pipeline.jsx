// pipeline.jsx — live processing view
const { useState, useMemo, useEffect, useRef } = React;

// ─────────────────────────────────────────────────────────────────────────────
// Stage definitions — mapped 1:1 from apps/api/grader/workers/pipeline_runner.py
const STAGES = [
  {
    k: 'preflight', l: 'Pre-flight', code: 'pipeline.started',
    desc: 'Validate required shots present and quality-gated',
    duration: 0.4, model: '—',
    sub: ['front_full present', 'quality_passed=true', 'optional shots accepted'],
  },
  {
    k: 'detect', l: 'Detect + dewarp', code: 'pipeline.detection.completed',
    desc: 'Yolo-Seg locates card; perspective unwrap to canonical 750×1050',
    duration: 2.4, model: 'yolo11-seg-v0.4.1',
    sub: ['front_full', 'back_full', 'tilt_30 (opt)', 'front_full_flash (opt)'],
  },
  {
    k: 'identify', l: 'Identification', code: 'pipeline.identification.completed',
    desc: 'DINOv2 embedding → pgvector cosine-search across catalog',
    duration: 1.8, model: 'dinov2-base + pgvec-v1.2',
    sub: ['embed front canonical', 'top-k=5 lookup', 'pick chosen variant'],
  },
  {
    k: 'counterfeit', l: 'Counterfeit ensemble', code: 'pipeline.counterfeit.completed',
    desc: '7-detector ensemble; conservative verdict combine',
    duration: 6.2, model: 'ensemble-v0.3.2',
    sub: ['rosette FFT', 'color profile', 'embedding anomaly', 'typography', 'holographic', 'kNN reference', 'substrate'],
  },
  {
    k: 'grade', l: 'Grade compose', code: 'pipeline.completed',
    desc: 'Centering + edges; min-merged into final',
    duration: 1.2, model: 'geometric-v0.3.0 + mad-v0.4.0',
    sub: ['centering (geom)', 'edges (MAD)', 'corners (skel)', 'surface (skel)', 'persist Grade row'],
  },
];

// 7 detectors run inside stage 'counterfeit'
const DETECTORS = [
  { k: 'rosette_fft',    l: 'Rosette FFT',     duration: 0.9, model: 'fft-pipeline',   anchor: 'image-only' },
  { k: 'color_profile',  l: 'Color profile',   duration: 0.6, model: 'cielab-deltaE',  anchor: 'image-only' },
  { k: 'embedding_anom', l: 'Embedding anomaly', duration: 0.8, model: 'centroid-cos', anchor: 'needs identify' },
  { k: 'typography',     l: 'Typography OCR',  duration: 1.1, model: 'rapidocr+KS',    anchor: 'needs name' },
  { k: 'holographic',    l: 'Holo parallax',   duration: 1.4, model: 'farnebäck-flow', anchor: 'needs tilt_30' },
  { k: 'knn_reference',  l: 'kNN reference',   duration: 0.7, model: 'topk-cos',       anchor: 'needs ≥k refs' },
  { k: 'substrate',      l: 'Substrate b*',    duration: 0.7, model: 'paired-flash',   anchor: 'needs flash' },
];

const TOTAL_DURATION = STAGES.reduce((s, x) => s + x.duration, 0);

// ─────────────────────────────────────────────────────────────────────────────
function App() {
  const DEFAULTS = window.__PIPELINE_TWEAK_DEFAULTS_RAW || { phase: 'running', currentStage: 4, showTelemetry: true, speed: 1 };
  const [t, setTweak] = useTweaks(DEFAULTS);

  // playback clock — accumulates seconds when phase=running
  const [clock, setClock] = useState(0);
  const startedAt = useRef(Date.now());

  useEffect(() => {
    if (t.phase !== 'running') return;
    let raf; let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      setClock(c => Math.min(TOTAL_DURATION + 0.5, c + dt * t.speed));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [t.phase, t.speed]);

  // when tweak `currentStage` changes (manual scrub), set clock to that stage's start time
  useEffect(() => {
    if (t.phase === 'running') return;
    if (t.phase === 'failed') {
      // freeze partway through stage `currentStage`
      let acc = 0;
      for (let i = 0; i < t.currentStage; i++) acc += STAGES[i].duration;
      setClock(acc + (STAGES[t.currentStage]?.duration || 0) * 0.6);
      return;
    }
    if (t.phase === 'completed') { setClock(TOTAL_DURATION + 0.4); return; }
    if (t.phase === 'queued') { setClock(0); return; }
  }, [t.phase, t.currentStage]);

  // derive each stage's progress + status from clock
  const stageStates = useMemo(() => {
    let acc = 0;
    return STAGES.map((s, i) => {
      const start = acc;
      const end = acc + s.duration;
      acc = end;
      let status, progress;
      if (t.phase === 'queued') { status = 'pending'; progress = 0; }
      else if (t.phase === 'failed' && i === t.currentStage) { status = 'failed'; progress = 0.6; }
      else if (t.phase === 'failed' && i > t.currentStage) { status = 'pending'; progress = 0; }
      else if (clock >= end) { status = 'done'; progress = 1; }
      else if (clock >= start) { status = 'running'; progress = (clock - start) / s.duration; }
      else { status = 'pending'; progress = 0; }
      return { ...s, idx: i, start, end, status, progress };
    });
  }, [clock, t.phase, t.currentStage]);

  const overallProgress = Math.min(1, clock / TOTAL_DURATION);
  const isComplete = t.phase === 'completed' || (t.phase === 'running' && clock >= TOTAL_DURATION);
  const isFailed = t.phase === 'failed';

  return (
    <main className="pipe-main">
      <Header t={t} setTweak={setTweak} clock={clock} totalDuration={TOTAL_DURATION} stageStates={stageStates} isComplete={isComplete} isFailed={isFailed} />

      <div className="pipe-grid">
        {/* Left: stage list */}
        <section className="pipe-stages">
          <SectionLabel hint="6 stages">Pipeline stages</SectionLabel>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginTop: 14 }}>
            {stageStates.map(s => (
              <StageRow key={s.k} stage={s} />
            ))}
          </div>
        </section>

        {/* Right: live focus panel */}
        <section className="pipe-focus">
          <FocusPanel stageStates={stageStates} clock={clock} t={t} isComplete={isComplete} isFailed={isFailed} />
        </section>
      </div>

      {t.showTelemetry && <TelemetryRail clock={clock} stageStates={stageStates} t={t} />}

      {isComplete && <FinishOverlay />}

      <PipelineTweaks t={t} setTweak={setTweak} />

      <style>{`
        .pipe-main {
          max-width: 1320px; margin: 28px auto; padding: 0 28px 80px;
          display: flex; flex-direction: column; gap: 22px;
        }
        .pipe-grid {
          display: grid;
          grid-template-columns: minmax(0, 460px) minmax(0, 1fr);
          gap: 22px;
        }
        @media (max-width: 1100px) { .pipe-grid { grid-template-columns: 1fr; } }
      `}</style>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function Header({ t, setTweak, clock, totalDuration, stageStates, isComplete, isFailed }) {
  const elapsed = clock.toFixed(1);
  const eta = Math.max(0, totalDuration - clock).toFixed(1);
  const submission = {
    id: 'sub_2c4f9a8b-3d71-4ec2-9b1f-c8d0e54f1b22',
    short: 'sub_2c4f9a',
    queued_at: '14:08:04',
    worker: 'celery-worker-3 · GPU-0',
    card: 'Charizard · Base Set #4/102 · Holo Unl.',
  };

  let phaseLabel, phaseColor, phaseDot;
  if (isFailed) { phaseLabel = 'Failed'; phaseColor = 'var(--rose)'; phaseDot = 'var(--rose)'; }
  else if (isComplete) { phaseLabel = 'Complete'; phaseColor = 'var(--lime)'; phaseDot = 'var(--lime)'; }
  else if (t.phase === 'queued') { phaseLabel = 'Queued'; phaseColor = 'var(--amber)'; phaseDot = 'var(--amber)'; }
  else { phaseLabel = 'Running'; phaseColor = 'var(--lime)'; phaseDot = 'var(--lime)'; }

  return (
    <header style={{
      padding: '18px 22px', background: 'var(--bg-1)',
      border: '1px solid var(--line)', borderRadius: 8,
      display: 'flex', flexDirection: 'column', gap: 16,
    }}>
      {/* top row */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 24, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>
              Live processing
            </span>
            <span style={{
              display: 'inline-flex', alignItems: 'center', gap: 6,
              padding: '3px 9px', borderRadius: 99,
              background: phaseColor + '14', border: '1px solid ' + phaseColor + '55',
              color: phaseColor, fontSize: 10, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.12em',
            }}>
              <span style={{ width: 5, height: 5, borderRadius: 3, background: phaseDot, boxShadow: '0 0 6px ' + phaseDot,
                animation: !isComplete && !isFailed ? 'pipePulse 1.4s ease-in-out infinite' : 'none' }} />
              {phaseLabel}
            </span>
          </div>
          <h1 style={{ fontFamily: 'Instrument Serif, serif', fontSize: 32, fontWeight: 400, color: 'var(--ink-0)', margin: 0, lineHeight: 1.1, letterSpacing: '-0.01em' }}>
            {submission.card}
          </h1>
          <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', display: 'flex', gap: 14, flexWrap: 'wrap' }}>
            <span><span style={{ color: 'var(--ink-4)' }}>id</span> {submission.id}</span>
            <span><span style={{ color: 'var(--ink-4)' }}>queued</span> {submission.queued_at}</span>
            <span><span style={{ color: 'var(--ink-4)' }}>worker</span> {submission.worker}</span>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 8 }}>
          {isComplete && <a href="cert.html" className="pg-btn pg-btn-primary">View certificate →</a>}
          {!isComplete && !isFailed && <button className="pg-btn pg-btn-ghost">Open logs</button>}
          {isFailed && <button className="pg-btn pg-btn-ghost">Retry</button>}
          <button className="pg-btn pg-btn-ghost">Cancel</button>
        </div>
      </div>

      {/* big clock + progress bar */}
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr auto', gap: 22, alignItems: 'center' }}>
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: 2 }}>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>elapsed</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 56, lineHeight: 0.9, color: 'var(--ink-0)', letterSpacing: '-0.02em', fontVariantNumeric: 'tabular-nums' }}>
              {elapsed}
            </span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>s</span>
          </div>
        </div>

        <SegmentedProgress stageStates={stageStates} clock={clock} />

        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 2 }}>
          <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.22em', textTransform: 'uppercase' }}>eta</span>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{ fontFamily: 'Instrument Serif, serif', fontSize: 36, lineHeight: 0.9, color: 'var(--ink-1)', letterSpacing: '-0.02em', fontVariantNumeric: 'tabular-nums' }}>
              {isComplete ? '0.0' : eta}
            </span>
            <span className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>s</span>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes pipePulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.45; transform: scale(0.85); }
        }
      `}</style>
    </header>
  );
}

function SegmentedProgress({ stageStates, clock }) {
  const total = stageStates.reduce((s, x) => s + x.duration, 0);
  return (
    <div>
      <div style={{ display: 'flex', gap: 3, height: 14, position: 'relative', borderRadius: 2, overflow: 'hidden' }}>
        {stageStates.map(s => {
          const w = (s.duration / total) * 100;
          let bg, fg;
          if (s.status === 'done') { bg = 'var(--lime-deep)'; fg = 'var(--lime)'; }
          else if (s.status === 'running') { bg = 'var(--bg-3)'; fg = 'var(--lime)'; }
          else if (s.status === 'failed') { bg = 'var(--bg-3)'; fg = 'var(--rose)'; }
          else { bg = 'var(--bg-3)'; fg = 'transparent'; }
          return (
            <div key={s.k} style={{ flexBasis: w + '%', position: 'relative', background: bg, borderRadius: 1 }}>
              {s.status === 'running' && (
                <div style={{ position: 'absolute', inset: 0, width: (s.progress * 100) + '%', background: fg, transition: 'width 100ms linear' }}>
                  <div style={{ position: 'absolute', right: 0, top: 0, bottom: 0, width: 12, background: 'linear-gradient(90deg, transparent, rgba(190,242,100,0.6))' }} />
                </div>
              )}
              {s.status === 'done' && <div style={{ position: 'absolute', inset: 0, background: fg }} />}
              {s.status === 'failed' && <div style={{ position: 'absolute', inset: 0, width: '60%', background: fg }} />}
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 3, marginTop: 6 }}>
        {stageStates.map(s => (
          <div key={s.k} style={{ flexBasis: ((s.duration / total) * 100) + '%' }}>
            <span className="mono" style={{
              fontSize: 9, color: s.status === 'done' ? 'var(--ink-2)' : s.status === 'running' ? 'var(--lime)' : 'var(--ink-4)',
              letterSpacing: '0.04em', textTransform: 'uppercase',
            }}>
              {s.l}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function StageRow({ stage }) {
  const tone = stage.status === 'done' ? 'var(--lime)'
    : stage.status === 'running' ? 'var(--lime)'
    : stage.status === 'failed' ? 'var(--rose)'
    : 'var(--ink-5)';

  return (
    <div style={{
      padding: '12px 14px',
      background: stage.status === 'running' ? 'var(--bg-2)' : 'transparent',
      border: '1px solid ' + (stage.status === 'running' ? 'var(--line)' : 'var(--line-2)'),
      borderRadius: 5,
      display: 'flex', flexDirection: 'column', gap: 8,
      transition: 'all 200ms',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <StageGlyph status={stage.status} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.18em' }}>
              {String(stage.idx + 1).padStart(2, '0')}
            </span>
            <span style={{ fontSize: 13, fontWeight: 500, color: stage.status === 'pending' ? 'var(--ink-3)' : 'var(--ink-0)' }}>
              {stage.l}
            </span>
            <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>· {stage.code}</span>
          </div>
          <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 2, lineHeight: 1.4 }}>
            {stage.desc}
          </div>
        </div>
        <div style={{ textAlign: 'right' }}>
          <div className="mono" style={{ fontSize: 10, color: tone, fontVariantNumeric: 'tabular-nums' }}>
            {stage.status === 'done' ? stage.duration.toFixed(2) + 's' :
             stage.status === 'running' ? (stage.progress * stage.duration).toFixed(1) + ' / ' + stage.duration.toFixed(1) + 's' :
             stage.status === 'failed' ? 'failed' :
             '—'}
          </div>
          <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', marginTop: 2 }}>{stage.model}</div>
        </div>
      </div>

      {stage.status === 'running' && (
        <div style={{ height: 2, background: 'var(--bg-3)', borderRadius: 1, overflow: 'hidden' }}>
          <div style={{ width: (stage.progress * 100) + '%', height: '100%', background: tone, transition: 'width 100ms linear' }} />
        </div>
      )}

      {(stage.status === 'running' || stage.status === 'done') && (
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginLeft: 30 }}>
          {stage.sub.map((s, i) => {
            const subProgress = stage.progress * stage.sub.length;
            const subActive = i < subProgress && i >= subProgress - 1;
            const subDone = i < subProgress - 1 || stage.status === 'done';
            return (
              <span key={i} className="mono" style={{
                fontSize: 9, padding: '2px 6px', borderRadius: 2,
                color: subDone ? 'var(--lime)' : subActive ? 'var(--ink-1)' : 'var(--ink-4)',
                background: subActive ? 'var(--bg-3)' : 'transparent',
                border: subDone ? '1px solid rgba(190,242,100,0.3)' : '1px solid var(--line-2)',
                letterSpacing: '0.02em',
              }}>
                {subDone ? '✓ ' : subActive ? '› ' : '  '}{s}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StageGlyph({ status }) {
  const size = 22;
  if (status === 'done') {
    return (
      <div style={{ width: size, height: size, borderRadius: 12, background: 'rgba(190,242,100,0.12)', border: '1px solid rgba(190,242,100,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
          <path d="M2 6.5L5 9L10 3" stroke="var(--lime)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }
  if (status === 'running') {
    return (
      <div style={{ width: size, height: size, borderRadius: 12, background: 'rgba(190,242,100,0.08)', border: '1px solid rgba(190,242,100,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, animation: 'stageRun 1.6s ease-in-out infinite' }}>
        <span style={{ width: 6, height: 6, borderRadius: 4, background: 'var(--lime)', boxShadow: '0 0 6px var(--lime)' }} />
        <style>{`@keyframes stageRun { 0%,100% { box-shadow: 0 0 0 0 rgba(190,242,100,0.4); } 50% { box-shadow: 0 0 0 6px transparent; } }`}</style>
      </div>
    );
  }
  if (status === 'failed') {
    return (
      <div style={{ width: size, height: size, borderRadius: 12, background: 'rgba(251,113,133,0.12)', border: '1px solid rgba(251,113,133,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
          <path d="M3 3L9 9M9 3L3 9" stroke="var(--rose)" strokeWidth="1.6" strokeLinecap="round" />
        </svg>
      </div>
    );
  }
  return (
    <div style={{ width: size, height: size, borderRadius: 12, background: 'transparent', border: '1px dashed var(--ink-5)', flexShrink: 0 }} />
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function FocusPanel({ stageStates, clock, t, isComplete, isFailed }) {
  // Show different focus content based on what's running RIGHT NOW
  const active = stageStates.find(s => s.status === 'running') || stageStates.find(s => s.status === 'failed');
  const lastDone = [...stageStates].reverse().find(s => s.status === 'done');
  const focusStage = active || lastDone || stageStates[0];

  if (isComplete) return <FocusComplete stageStates={stageStates} />;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionLabel hint={`stage ${focusStage.idx + 1} of 6`}>{active ? 'Now running' : 'Stage detail'}</SectionLabel>

      {focusStage.k === 'preflight' && <FocusPreflight stage={focusStage} />}
      {focusStage.k === 'detect' && <FocusDetection stage={focusStage} />}
      {focusStage.k === 'identify' && <FocusIdentification stage={focusStage} />}
      {focusStage.k === 'counterfeit' && <FocusCounterfeit stage={focusStage} />}
      {focusStage.k === 'grade' && <FocusGrade stage={focusStage} />}
    </div>
  );
}

// Stage 1 — pre-flight
function FocusPreflight({ stage }) {
  const checks = [
    { l: 'front_full present', ok: true },
    { l: 'front_full quality_passed', ok: true, detail: 'B 0.91 · G 0.04 · P 0.99' },
    { l: 'back_full present', ok: true },
    { l: 'back_full quality_passed', ok: true, detail: 'B 0.88 · G 0.06 · P 0.97' },
    { l: 'tilt_30 (optional)', ok: true, detail: 'accepted' },
    { l: 'front_full_flash (optional)', ok: true, detail: 'accepted' },
    { l: 'corners_x4 (optional)', ok: false, detail: 'not captured · skip' },
  ];
  return (
    <Card>
      {checks.map(c => (
        <div key={c.l} style={{ display: 'flex', alignItems: 'baseline', gap: 12, padding: '8px 0', borderBottom: '1px solid var(--line-2)' }}>
          <span style={{ width: 14, color: c.ok ? 'var(--lime)' : 'var(--ink-4)', fontSize: 12 }}>{c.ok ? '✓' : '–'}</span>
          <span style={{ fontSize: 13, color: c.ok ? 'var(--ink-1)' : 'var(--ink-3)', flex: 1 }}>{c.l}</span>
          {c.detail && <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>{c.detail}</span>}
        </div>
      ))}
    </Card>
  );
}

// Stage 2 — detection visualization
function FocusDetection({ stage }) {
  const shotIdx = Math.min(3, Math.floor(stage.progress * 4));
  const SHOTS = [
    { k: 'front_full', l: 'Front', icon: '▢' },
    { k: 'back_full', l: 'Back', icon: '▣' },
    { k: 'tilt_30', l: 'Tilt 30°', icon: '◇', opt: true },
    { k: 'front_full_flash', l: 'Flash', icon: '✦', opt: true },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Card>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
          {SHOTS.map((s, i) => (
            <DetectionShot key={s.k} shot={s} state={
              stage.status === 'done' ? 'done' :
              i < shotIdx ? 'done' :
              i === shotIdx ? 'running' : 'pending'
            } />
          ))}
        </div>
      </Card>
      <Card>
        <KV k="Model" v="yolo11-seg-v0.4.1" />
        <KV k="Canonical size" v="750 × 1050 px" />
        <KV k="Max irregularity" v="0.30 (front/back) · 0.45 (tilt)" />
        <KV k="Output S3 prefix" v={<span className="mono">canon/sub_2c4f9a/</span>} />
      </Card>
    </div>
  );
}

function DetectionShot({ shot, state }) {
  const tone = state === 'done' ? 'var(--lime)' : state === 'running' ? 'var(--lime)' : 'var(--ink-5)';
  return (
    <div style={{
      aspectRatio: '5/7', position: 'relative', borderRadius: 4,
      background: state === 'pending' ? 'repeating-linear-gradient(45deg, var(--bg-2), var(--bg-2) 6px, var(--bg-1) 6px, var(--bg-1) 12px)' : 'var(--bg-2)',
      border: '1px ' + (state === 'pending' ? 'dashed' : 'solid') + ' ' + (state === 'pending' ? 'var(--line-2)' : 'var(--line)'),
      overflow: 'hidden',
    }}>
      {/* card silhouette */}
      {state !== 'pending' && (
        <div style={{
          position: 'absolute', inset: '6%', borderRadius: 2,
          background: 'linear-gradient(135deg, #2a1a0a, #6b3410, #f97316)',
          opacity: 0.4,
        }} />
      )}
      {/* detection corners */}
      {state !== 'pending' && (
        <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }}>
          {[[6,6],[94,6],[6,94],[94,94]].map(([x,y]) => (
            <g key={x+'-'+y}>
              <line x1={x+'%'} y1={y+'%'} x2={(x>50?x-8:x+8)+'%'} y2={y+'%'} stroke={tone} strokeWidth="1" />
              <line x1={x+'%'} y1={y+'%'} x2={x+'%'} y2={(y>50?y-8:y+8)+'%'} stroke={tone} strokeWidth="1" />
            </g>
          ))}
        </svg>
      )}
      {state === 'running' && (
        <div style={{ position: 'absolute', left: 0, right: 0, top: '50%', height: 2, background: 'var(--lime)', boxShadow: '0 0 8px var(--lime)', animation: 'scanline 1.4s ease-in-out infinite' }} />
      )}
      <div style={{ position: 'absolute', bottom: 4, left: 6, right: 6, display: 'flex', justifyContent: 'space-between' }}>
        <span className="mono" style={{ fontSize: 8, color: tone, letterSpacing: '0.06em', textTransform: 'uppercase' }}>{shot.l}</span>
        <span className="mono" style={{ fontSize: 8, color: tone }}>{state === 'done' ? '✓' : state === 'running' ? '…' : '·'}</span>
      </div>
      <style>{`@keyframes scanline { 0%,100% { top: 14%; } 50% { top: 86%; } }`}</style>
    </div>
  );
}

// Stage 3 — identification
function FocusIdentification({ stage }) {
  const candidates = [
    { id: '4d2f-base1-charizard-holo',    name: 'Charizard',           set: 'Base · 4/102 · Holo Unl', score: 0.973, chosen: true },
    { id: '7a1c-base2-charizard-holo',    name: 'Charizard',           set: 'Base 2 · 4/130 · Holo',  score: 0.681 },
    { id: '9b3e-jungle1-flareon-holo',    name: 'Flareon',             set: 'Jungle · 3/64 · Holo',   score: 0.412 },
    { id: '2c8a-shadowless-charizard-holo', name: 'Charizard (Shadowless)', set: 'Base · 4/102 · 1st Ed', score: 0.358 },
    { id: '5f0d-fossil-rapidash',         name: 'Rapidash',            set: 'Fossil · 56/62',         score: 0.241 },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Card>
        <KV k="Embedding" v="DINOv2-base · 768-dim" />
        <KV k="Catalog index" v="pgvector · ivfflat · 64 lists" />
        <KV k="top-k" v="5 candidates" />
      </Card>
      <Card label="Top-5 cosine similarity">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
          {candidates.map((c, i) => {
            const reveal = stage.status === 'done' ? 1 : Math.max(0, Math.min(1, stage.progress * 8 - i));
            return (
              <div key={c.id} style={{
                display: 'grid', gridTemplateColumns: '1fr 90px 60px',
                gap: 12, alignItems: 'center', padding: '6px 8px',
                background: c.chosen ? 'rgba(190,242,100,0.06)' : 'transparent',
                border: c.chosen ? '1px solid rgba(190,242,100,0.3)' : '1px solid var(--line-2)',
                borderRadius: 3, opacity: 0.3 + reveal * 0.7,
                transition: 'opacity 200ms',
              }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontSize: 12, color: c.chosen ? 'var(--lime)' : 'var(--ink-1)', fontWeight: c.chosen ? 600 : 400 }}>
                    {c.name} {c.chosen && <span className="mono" style={{ fontSize: 9, marginLeft: 6, letterSpacing: '0.1em', textTransform: 'uppercase' }}>chosen</span>}
                  </div>
                  <div className="mono" style={{ fontSize: 10, color: 'var(--ink-4)' }}>{c.set}</div>
                </div>
                <div style={{ height: 4, background: 'var(--bg-3)', borderRadius: 1, overflow: 'hidden' }}>
                  <div style={{ width: (c.score * 100 * reveal) + '%', height: '100%', background: c.chosen ? 'var(--lime)' : 'var(--ink-4)', transition: 'width 200ms' }} />
                </div>
                <span className="mono" style={{ fontSize: 11, color: c.chosen ? 'var(--lime)' : 'var(--ink-2)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                  {(c.score * reveal).toFixed(3)}
                </span>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

// Stage 4 — counterfeit ensemble (the hero stage)
function FocusCounterfeit({ stage }) {
  const detectorProgress = stage.progress * DETECTORS.length;
  const detectorsState = DETECTORS.map((d, i) => {
    const done = stage.status === 'done' || i < detectorProgress - 0.4;
    const running = !done && i < detectorProgress + 0.6 && i >= detectorProgress - 0.4;
    return {
      ...d,
      status: done ? 'done' : running ? 'running' : 'pending',
      score: done ? 0.85 + Math.random() * 0.15 : running ? 0.5 + Math.random() * 0.4 : null,
      progress: done ? 1 : running ? (detectorProgress - i + 0.4) : 0,
    };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Card label="7-detector ensemble · running in parallel">
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {detectorsState.map(d => <DetectorRow key={d.k} d={d} />)}
        </div>
      </Card>
      <Card label="Conservative verdict combiner">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 6, fontSize: 11 }}>
          <Stat l="Voted authentic" v={detectorsState.filter(d => d.status === 'done' && d.score > 0.85).length + ' / 7'} c="var(--lime)" />
          <Stat l="Abstained" v="0 / 7" c="var(--ink-3)" />
          <Stat l="Combined conf" v={stage.status === 'done' ? '0.973' : (0.5 + stage.progress * 0.4).toFixed(3)} c="var(--ink-1)" />
        </div>
      </Card>
    </div>
  );
}

function DetectorRow({ d }) {
  const tone = d.status === 'done' ? 'var(--lime)' : d.status === 'running' ? 'var(--lime)' : 'var(--ink-4)';
  return (
    <div style={{
      display: 'grid', gridTemplateColumns: '14px 130px 1fr 80px 50px',
      gap: 8, alignItems: 'center', padding: '5px 4px',
      background: d.status === 'running' ? 'var(--bg-3)' : 'transparent',
      borderRadius: 3,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: 4, background: tone, opacity: d.status === 'pending' ? 0.3 : 1, boxShadow: d.status === 'running' ? '0 0 6px ' + tone : 'none', animation: d.status === 'running' ? 'pipePulse 1.4s ease-in-out infinite' : 'none' }} />
      <span style={{ fontSize: 12, color: d.status === 'pending' ? 'var(--ink-4)' : 'var(--ink-1)' }}>{d.l}</span>
      <div style={{ height: 3, background: 'var(--bg-2)', borderRadius: 1, overflow: 'hidden' }}>
        <div style={{ width: (d.progress * 100) + '%', height: '100%', background: tone, transition: 'width 100ms linear' }} />
      </div>
      <span className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', letterSpacing: '0.04em', textAlign: 'right' }}>{d.anchor}</span>
      <span className="mono" style={{ fontSize: 11, color: tone, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
        {d.score !== null ? d.score.toFixed(2) : '—'}
      </span>
    </div>
  );
}

// Stage 5 — grade compose
function FocusGrade({ stage }) {
  const reveal = stage.status === 'done' ? 1 : stage.progress;
  const subs = [
    { k: 'centering', l: 'Centering', v: 9.0, by: 'geometric · L/R 54.2 · T/B 51.6' },
    { k: 'edges',     l: 'Edges',     v: 9.0, by: 'MAD · worse-face min, 4 measurements' },
    { k: 'corners',   l: 'Corners',   v: 8.5, by: 'EfficientNet-V2 · 4 corners min' },
    { k: 'surface',   l: 'Surface',   v: 9.5, by: 'paired-flash + rosette' },
  ];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <Card label="Subgrade composition · min-merged">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 8 }}>
          {subs.map((s, i) => {
            const subReveal = Math.max(0, Math.min(1, reveal * subs.length * 1.2 - i));
            return (
              <div key={s.k} style={{
                background: 'var(--bg-2)', border: '1px solid var(--line-2)',
                borderRadius: 4, padding: '12px 14px',
                opacity: 0.3 + subReveal * 0.7,
                transition: 'opacity 250ms',
              }}>
                <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{s.l}</div>
                <div style={{ fontFamily: 'Instrument Serif, serif', fontSize: 30, lineHeight: 1, color: 'var(--ink-0)', marginTop: 4, letterSpacing: '-0.02em' }}>
                  {(s.v * subReveal).toFixed(1)}
                </div>
                <div className="mono" style={{ fontSize: 9, color: 'var(--ink-4)', marginTop: 4 }}>{s.by}</div>
              </div>
            );
          })}
        </div>
      </Card>
      <Card label="Final grade — min(centering, edges, corners, surface)">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 18 }}>
          <div>
            <div className="mono" style={{ fontSize: 10, color: 'var(--lime)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>PSA · final</div>
            <div style={{ fontFamily: 'Instrument Serif, serif', fontSize: 64, lineHeight: 1, color: 'var(--ink-0)', letterSpacing: '-0.03em', textShadow: '0 0 40px rgba(190,242,100,0.2)' }}>
              {(9.0 * reveal).toFixed(1)}
            </div>
          </div>
          <div style={{ flex: 1, height: 80, position: 'relative' }}>
            <svg viewBox="0 0 200 80" style={{ width: '100%', height: '100%' }}>
              {/* baseline */}
              <line x1="10" x2="190" y1="40" y2="40" stroke="var(--line)" strokeWidth="1" />
              {[1,2,3,4,5,6,7,8,9,10].map(n => (
                <line key={n} x1={10 + (n-1)/9 * 180} x2={10 + (n-1)/9 * 180} y1="36" y2="44" stroke="var(--ink-5)" strokeWidth="0.5" />
              ))}
              {/* current */}
              <circle cx={10 + ((9 - 1) / 9) * 180 * reveal} cy="40" r="6" fill="var(--lime)" opacity={reveal} />
              <circle cx={10 + ((9 - 1) / 9) * 180 * reveal} cy="40" r="12" fill="none" stroke="var(--lime)" strokeWidth="1" opacity={reveal * 0.4}/>
              {/* labels */}
              <text x="10" y="62" fontFamily="Geist Mono" fontSize="7" fill="var(--ink-4)">1.0</text>
              <text x="100" y="62" fontFamily="Geist Mono" fontSize="7" fill="var(--ink-4)" textAnchor="middle">5.0</text>
              <text x="190" y="62" fontFamily="Geist Mono" fontSize="7" fill="var(--ink-4)" textAnchor="end">10.0</text>
            </svg>
          </div>
        </div>
      </Card>
    </div>
  );
}

// Done state
function FocusComplete({ stageStates }) {
  const total = stageStates.reduce((s, x) => s + x.duration, 0);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <SectionLabel hint="cert sealed">Pipeline complete</SectionLabel>
      <div style={{
        padding: '28px 24px',
        background: 'linear-gradient(135deg, rgba(190,242,100,0.06), transparent), var(--bg-1)',
        border: '1px solid rgba(190,242,100,0.3)', borderRadius: 8,
        display: 'flex', flexDirection: 'column', gap: 18,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <div style={{ width: 38, height: 38, borderRadius: 19, background: 'rgba(190,242,100,0.1)', border: '1px solid rgba(190,242,100,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg width="18" height="18" viewBox="0 0 12 12" fill="none">
              <path d="M2 6.5L5 9L10 3" stroke="var(--lime)" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
          <div>
            <div style={{ fontSize: 20, color: 'var(--ink-0)', fontFamily: 'Instrument Serif, serif' }}>Cert PG-2026-0F1A4C-9D71 sealed</div>
            <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)' }}>{total.toFixed(1)}s end-to-end · 6 stages · 7 detectors</div>
          </div>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
          <Stat l="Final grade" v="9.0 / 10" c="var(--lime)" />
          <Stat l="Confidence" v="89.2%" c="var(--ink-1)" />
          <Stat l="Verdict" v="Authentic" c="var(--lime)" />
          <Stat l="Rank" v="#3 of 47" c="var(--ink-1)" />
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <a href="cert.html" className="pg-btn pg-btn-primary" style={{ flex: 1, justifyContent: 'center' }}>View certificate →</a>
          <a href="card.html" className="pg-btn pg-btn-ghost">Open card history</a>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Right-side telemetry rail (resource usage, audit log, queue)
function TelemetryRail({ clock, stageStates, t }) {
  return (
    <section style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14 }}>
      <ResourcePanel clock={clock} stageStates={stageStates} />
      <AuditLogPanel stageStates={stageStates} />
      <QueuePanel />
    </section>
  );
}

function ResourcePanel({ clock, stageStates }) {
  // synthesize a sparkline that ramps with active stage
  const N = 60;
  const data = useMemo(() => {
    const arr = [];
    for (let i = 0; i < N; i++) {
      const tt = (clock - (N - i) * 0.15);
      let v = 0.15;
      if (tt > 0) {
        const sIdx = stageStates.findIndex(s => tt >= s.start && tt < s.end);
        if (sIdx >= 0) {
          const baseline = { preflight: 0.05, detect: 0.55, identify: 0.75, counterfeit: 0.85, grade: 0.4 }[stageStates[sIdx].k] || 0.2;
          v = baseline + (Math.sin(tt * 7) * 0.08) + (Math.random() * 0.04);
        }
      }
      arr.push(Math.max(0.02, Math.min(0.98, v)));
    }
    return arr;
  }, [Math.floor(clock * 4), stageStates]);

  return (
    <Card label="GPU · vram + util">
      <Sparkline data={data} color="var(--lime)" label="util" current={data[data.length - 1] * 100} unit="%" />
      <div style={{ height: 8 }} />
      <Sparkline data={data.map(d => d * 0.7 + 0.1)} color="var(--cyan)" label="vram" current={data[data.length - 1] * 70 + 10} unit="% of 24GB" />
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 4, marginTop: 10 }}>
        <KVSmall k="device" v="NVIDIA RTX 4090" />
        <KVSmall k="cuda" v="12.3" />
      </div>
    </Card>
  );
}

function Sparkline({ data, color, label, current, unit }) {
  const max = 1;
  const w = 100, h = 28;
  const path = data.map((v, i) => `${i === 0 ? 'M' : 'L'} ${(i / (data.length - 1)) * w} ${h - (v / max) * h}`).join(' ');
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
        <span className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.04em', textTransform: 'uppercase' }}>{label}</span>
        <span className="mono" style={{ fontSize: 11, color }}>{current.toFixed(0)}<span style={{ color: 'var(--ink-4)', fontSize: 9 }}> {unit}</span></span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none" style={{ width: '100%', height: 28, display: 'block' }}>
        <path d={`${path} L ${w} ${h} L 0 ${h} Z`} fill={color} opacity="0.1" />
        <path d={path} fill="none" stroke={color} strokeWidth="0.8" />
      </svg>
    </div>
  );
}

function AuditLogPanel({ stageStates }) {
  const events = [];
  events.push({ ts: '0.00', a: 'pipeline.started', tone: 'ink' });
  stageStates.forEach((s, i) => {
    if (s.status === 'done') events.push({ ts: s.end.toFixed(2), a: s.code, tone: 'lime' });
    else if (s.status === 'running') events.push({ ts: '...', a: s.code + ' (running)', tone: 'amber' });
    else if (s.status === 'failed') events.push({ ts: '!', a: 'pipeline.failed reason=' + s.code, tone: 'rose' });
  });

  return (
    <Card label="Audit log · this run">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4, maxHeight: 200, overflow: 'hidden' }}>
        {events.slice(-10).map((e, i) => {
          const c = { lime: 'var(--lime)', amber: 'var(--amber)', rose: 'var(--rose)', ink: 'var(--ink-3)' }[e.tone];
          return (
            <div key={i} className="mono" style={{
              display: 'flex', gap: 10, fontSize: 10, padding: '3px 0',
              borderBottom: '1px dashed var(--line-2)',
            }}>
              <span style={{ color: 'var(--ink-4)', width: 36, textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{e.ts}s</span>
              <span style={{ color: c, flex: 1 }}>{e.a}</span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function QueuePanel() {
  const queue = [
    { id: 'sub_2c4f9a', card: 'Charizard · Base 4/102', stage: 'counterfeit', state: 'running' },
    { id: 'sub_8d12bb', card: 'Blastoise · Base 2/102', stage: 'queued',     state: 'queued' },
    { id: 'sub_09a4cf', card: 'Pikachu · Jungle 60/64', stage: 'queued',     state: 'queued' },
    { id: 'sub_71e2b0', card: 'Venusaur · Base 15/102', stage: 'queued',     state: 'queued' },
  ];
  return (
    <Card label="Queue · celery default · 1 worker">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
        {queue.map(q => (
          <div key={q.id} style={{
            display: 'grid', gridTemplateColumns: '90px 1fr auto',
            gap: 8, padding: '5px 8px', borderRadius: 3,
            background: q.state === 'running' ? 'rgba(190,242,100,0.06)' : 'transparent',
            border: '1px solid ' + (q.state === 'running' ? 'rgba(190,242,100,0.3)' : 'var(--line-2)'),
          }}>
            <span className="mono" style={{ fontSize: 10, color: 'var(--ink-3)' }}>{q.id}</span>
            <span style={{ fontSize: 11, color: q.state === 'running' ? 'var(--ink-1)' : 'var(--ink-3)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{q.card}</span>
            <span className="mono" style={{ fontSize: 9, color: q.state === 'running' ? 'var(--lime)' : 'var(--ink-4)', letterSpacing: '0.1em', textTransform: 'uppercase' }}>{q.state}</span>
          </div>
        ))}
        <div style={{ marginTop: 6, paddingTop: 6, borderTop: '1px solid var(--line-2)', display: 'flex', justifyContent: 'space-between', fontSize: 10 }}>
          <span className="mono" style={{ color: 'var(--ink-4)' }}>p50 latency</span>
          <span className="mono" style={{ color: 'var(--ink-1)' }}>11.4s · last 24h</span>
        </div>
      </div>
    </Card>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Finish overlay
function FinishOverlay() {
  const [show, setShow] = useState(true);
  useEffect(() => { const t = setTimeout(() => setShow(false), 2400); return () => clearTimeout(t); }, []);
  if (!show) return null;
  return (
    <div style={{
      position: 'fixed', inset: 0, pointerEvents: 'none', zIndex: 50,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <div style={{
        width: 280, height: 280, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(190,242,100,0.12), transparent 70%)',
        animation: 'finishPulse 2.4s ease-out forwards',
      }} />
      <style>{`
        @keyframes finishPulse {
          0% { transform: scale(0.4); opacity: 0; }
          30% { opacity: 1; }
          100% { transform: scale(2.4); opacity: 0; }
        }
      `}</style>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
function Card({ children, label }) {
  return (
    <div style={{ background: 'var(--bg-1)', border: '1px solid var(--line)', borderRadius: 6, padding: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
      {label && <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.18em', textTransform: 'uppercase' }}>{label}</div>}
      {children}
    </div>
  );
}

function SectionLabel({ children, hint }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', borderBottom: '1px solid var(--line-2)', paddingBottom: 8 }}>
      <h2 className="mono" style={{ fontSize: 11, color: 'var(--ink-2)', letterSpacing: '0.18em', textTransform: 'uppercase', margin: 0, fontWeight: 500 }}>{children}</h2>
      {hint && <span className="mono" style={{ fontSize: 10, color: 'var(--ink-4)', letterSpacing: '0.04em' }}>{hint}</span>}
    </div>
  );
}

function KV({ k, v }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, padding: '6px 0', borderBottom: '1px dashed var(--line-2)', fontSize: 12 }}>
      <span style={{ color: 'var(--ink-3)' }}>{k}</span>
      <span style={{ color: 'var(--ink-1)', fontFamily: typeof v === 'string' && v.length < 30 ? 'inherit' : undefined }}>{v}</span>
    </div>
  );
}

function KVSmall({ k, v }) {
  return (
    <div style={{ background: 'var(--bg-2)', borderRadius: 3, padding: '4px 6px' }}>
      <div className="mono" style={{ fontSize: 8, color: 'var(--ink-4)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>{k}</div>
      <div className="mono" style={{ fontSize: 11, color: 'var(--ink-1)', marginTop: 2 }}>{v}</div>
    </div>
  );
}

function Stat({ l, v, c }) {
  return (
    <div style={{ background: 'var(--bg-2)', borderRadius: 3, padding: '8px 10px' }}>
      <div className="mono" style={{ fontSize: 9, color: 'var(--ink-3)', letterSpacing: '0.14em', textTransform: 'uppercase' }}>{l}</div>
      <div style={{ fontFamily: 'Instrument Serif, serif', fontSize: 18, color: c || 'var(--ink-0)', marginTop: 4, lineHeight: 1 }}>{v}</div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function PipelineTweaks({ t, setTweak }) {
  return (
    <TweaksPanel>
      <TweakSection label="Run state" />
      <TweakRadio label="Phase"
        value={t.phase}
        options={['queued','running','failed','completed']}
        onChange={v => setTweak('phase', v)} />
      {t.phase === 'failed' && (
        <TweakSelect label="Failed at stage"
          value={String(t.currentStage)}
          options={['0','1','2','3','4']}
          onChange={v => setTweak('currentStage', parseInt(v, 10))} />
      )}
      <TweakSlider label="Playback speed" min={0.25} max={3} step={0.25}
        value={t.speed} onChange={v => setTweak('speed', v)} />
      <TweakSection label="Layout" />
      <TweakToggle label="Show telemetry rail" value={t.showTelemetry} onChange={v => setTweak('showTelemetry', v)} />
    </TweaksPanel>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App />);
