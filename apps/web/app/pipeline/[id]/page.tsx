/**
 * `/pipeline/[id]` — live processing view for a single submission.
 *
 * Visual port of `docs/design/pipeline.jsx` with the design's mock
 * playback clock replaced by real submission status. Server-side
 * progress is coarse (we don't surface per-stage events through the
 * public API), so all stages share the submission's overall status:
 *
 *   PENDING / CAPTURING  → all stages "pending"
 *   PROCESSING           → all stages "running" (no per-stage telemetry)
 *   COMPLETED            → all stages "done"; "View certificate →" CTA
 *   FAILED               → all stages "failed"
 *   REJECTED_QUALITY     → preflight "failed", others "pending"
 *
 * Page polls `GET /submissions/{id}` every 2s while running. Useful as
 * the share-able "your card is being processed" link, and as a static
 * record of which stages ran post-completion.
 */

"use client";

import Link from "next/link";
import { use, useCallback, useEffect, useState } from "react";

import {
  type SubmissionOut,
  type SubmissionStatus,
  getSubmission,
  useAuthedFetch,
} from "@/lib/submission";

// --------------------------------------------------------------------------
// Stage definitions — mirror `apps/api/grader/workers/pipeline_runner.py`
// --------------------------------------------------------------------------

type Stage = {
  k: "preflight" | "detect" | "identify" | "counterfeit" | "grade";
  l: string;
  code: string;
  desc: string;
  model: string;
  sub: string[];
};

const STAGES: Stage[] = [
  {
    k: "preflight",
    l: "Pre-flight",
    code: "pipeline.started",
    desc: "Validate required shots present and quality-gated",
    model: "—",
    sub: ["front_full present", "quality_passed=true", "optional shots accepted"],
  },
  {
    k: "detect",
    l: "Detect + dewarp",
    code: "pipeline.detection.completed",
    desc: "Yolo-Seg locates card; perspective unwrap to canonical 750×1050",
    model: "yolo11-seg-v0.4.1",
    sub: ["front_full", "back_full", "tilt_30 (opt)", "front_full_flash (opt)"],
  },
  {
    k: "identify",
    l: "Identification",
    code: "pipeline.identification.completed",
    desc: "DINOv2 embedding → pgvector cosine-search across catalog",
    model: "dinov2-base + pgvec-v1.2",
    sub: ["embed front canonical", "top-k=5 lookup", "pick chosen variant"],
  },
  {
    k: "counterfeit",
    l: "Counterfeit ensemble",
    code: "pipeline.counterfeit.completed",
    desc: "7-detector ensemble; conservative verdict combine",
    model: "ensemble-v0.3.2",
    sub: [
      "rosette FFT",
      "color profile",
      "embedding anomaly",
      "typography",
      "holographic",
      "kNN reference",
      "substrate",
    ],
  },
  {
    k: "grade",
    l: "Grade compose",
    code: "pipeline.completed",
    desc: "Centering + edges; min-merged into final",
    model: "geometric-v0.3.0 + mad-v0.4.0",
    sub: [
      "centering (geom)",
      "edges (MAD)",
      "corners (skel)",
      "surface (skel)",
      "persist Grade row",
    ],
  },
];

type StageStatus = "pending" | "running" | "done" | "failed";
type StageState = Stage & { idx: number; status: StageStatus };

function deriveStageStates(submission: SubmissionOut | null): StageState[] {
  return STAGES.map((s, idx) => ({
    ...s,
    idx,
    status: deriveStageStatus(submission, s.k, idx),
  }));
}

function deriveStageStatus(
  submission: SubmissionOut | null,
  stageK: Stage["k"],
  idx: number,
): StageStatus {
  if (!submission) return "pending";
  switch (submission.status) {
    case "completed":
      return "done";
    case "processing":
      return "running";
    case "failed":
      // Without per-stage failure detail from the API, mark the whole
      // run as failed. A future enhancement could read the audit log
      // to pinpoint which stage emitted the rejection.
      return "failed";
    case "rejected_quality":
      // Quality gate is the preflight stage; everything after is
      // skipped on rejection.
      return idx === 0 ? "failed" : "pending";
    case "pending":
    case "capturing":
    default:
      return "pending";
  }
}

// --------------------------------------------------------------------------
// Page entry
// --------------------------------------------------------------------------

type Params = { id: string };

export default function PipelinePage({ params }: { params: Promise<Params> }) {
  const { id } = use(params);
  const authedFetch = useAuthedFetch();

  const [submission, setSubmission] = useState<SubmissionOut | null | undefined>(
    undefined,
  );
  const [error, setError] = useState<string | null>(null);
  const [startedAt] = useState(() => Date.now());
  const [tick, setTick] = useState(0);

  const refresh = useCallback(async () => {
    try {
      const sub = await getSubmission(authedFetch, id);
      setSubmission(sub);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [id, authedFetch]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Poll while processing; advance the elapsed clock every second
  // regardless so the header timer animates even when we're between
  // status fetches.
  useEffect(() => {
    if (!submission) return;
    if (submission.status === "processing") {
      const refreshHandle = setInterval(() => void refresh(), 2000);
      const tickHandle = setInterval(() => setTick((n) => n + 1), 1000);
      return () => {
        clearInterval(refreshHandle);
        clearInterval(tickHandle);
      };
    }
    return undefined;
  }, [submission, refresh]);

  if (submission === undefined) return <PageShell><LoadingState /></PageShell>;
  if (submission === null) return <PageShell><NotFoundState /></PageShell>;

  const stageStates = deriveStageStates(submission);

  return (
    <PageShell>
      <Header submission={submission} startedAt={startedAt} tick={tick} />
      {error && (
        <div
          style={{
            padding: "10px 14px",
            background: "rgba(251,113,133,0.04)",
            border: "1px solid rgba(251,113,133,0.4)",
            borderRadius: 4,
            color: "var(--rose)",
            fontSize: 12,
          }}
        >
          {error}
        </div>
      )}
      <div
        className="pipe-grid"
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 460px) minmax(0, 1fr)",
          gap: 22,
        }}
      >
        <StageList stageStates={stageStates} />
        <FocusPanel stageStates={stageStates} submission={submission} />
      </div>
      <style>{`
        @media (max-width: 1100px) {
          .pipe-grid { grid-template-columns: 1fr !important; }
        }
        @keyframes stageRun {
          0%, 100% { box-shadow: 0 0 0 0 rgba(190,242,100,0.4); }
          50% { box-shadow: 0 0 0 6px transparent; }
        }
      `}</style>
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <main
      style={{
        maxWidth: 1320,
        margin: "28px auto",
        padding: "0 28px 80px",
        display: "flex",
        flexDirection: "column",
        gap: 22,
      }}
    >
      {children}
    </main>
  );
}

function LoadingState() {
  return (
    <p
      className="mono"
      style={{
        fontSize: 11,
        color: "var(--ink-3)",
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        padding: "60px 0",
        textAlign: "center",
      }}
    >
      loading submission…
    </p>
  );
}

function NotFoundState() {
  return (
    <section
      style={{
        padding: "60px 24px",
        textAlign: "center",
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 8,
      }}
    >
      <h1
        style={{
          fontFamily: "var(--font-instrument-serif), serif",
          fontSize: 36,
          color: "var(--ink-0)",
          margin: 0,
        }}
      >
        Submission not found
      </h1>
      <p style={{ marginTop: 12, color: "var(--ink-3)", fontSize: 13 }}>
        This submission doesn&apos;t exist or isn&apos;t yours.
      </p>
    </section>
  );
}

// --------------------------------------------------------------------------
// Header — submission identity + clock + segmented progress
// --------------------------------------------------------------------------

const STATUS_TONE: Record<
  SubmissionStatus,
  { label: string; tone: string; pulse: boolean }
> = {
  pending: { label: "Queued", tone: "var(--amber)", pulse: false },
  capturing: { label: "Capturing", tone: "var(--ink-3)", pulse: false },
  processing: { label: "Running", tone: "var(--lime)", pulse: true },
  completed: { label: "Complete", tone: "var(--lime)", pulse: false },
  failed: { label: "Failed", tone: "var(--rose)", pulse: false },
  rejected_quality: { label: "Rejected", tone: "var(--rose)", pulse: false },
};

function Header({
  submission,
  startedAt,
  tick,
}: {
  submission: SubmissionOut;
  startedAt: number;
  tick: number;
}) {
  const tone = STATUS_TONE[submission.status];
  const isComplete = submission.status === "completed";
  const isFailed = submission.status === "failed" || submission.status === "rejected_quality";

  // Prefer server-side timestamps when available; fall back to
  // mount-time elapsed otherwise so the page doesn't render a stuck
  // 0.0 while we wait for the first refresh.
  const elapsedSeconds = computeElapsedSeconds(submission, startedAt, tick);
  const cardName = "Submission " + submission.id.slice(0, 8);

  return (
    <header
      style={{
        padding: "18px 22px",
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 8,
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 24,
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--ink-3)",
                letterSpacing: "0.22em",
                textTransform: "uppercase",
              }}
            >
              Live processing
            </span>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "3px 9px",
                borderRadius: 99,
                background: tone.tone + "14",
                border: `1px solid ${tone.tone}55`,
                color: tone.tone,
                fontSize: 10,
                fontWeight: 500,
                textTransform: "uppercase",
                letterSpacing: "0.12em",
              }}
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: 3,
                  background: tone.tone,
                  boxShadow: `0 0 6px ${tone.tone}`,
                  animation: tone.pulse ? "pulse 1.4s ease-in-out infinite" : "none",
                }}
              />
              {tone.label}
            </span>
          </div>
          <h1
            style={{
              fontFamily: "var(--font-instrument-serif), serif",
              fontSize: 32,
              fontWeight: 400,
              color: "var(--ink-0)",
              margin: 0,
              lineHeight: 1.1,
              letterSpacing: "-0.01em",
            }}
          >
            {cardName}
          </h1>
          <div
            className="mono"
            style={{
              fontSize: 11,
              color: "var(--ink-3)",
              display: "flex",
              gap: 14,
              flexWrap: "wrap",
            }}
          >
            <span>
              <span style={{ color: "var(--ink-4)" }}>id</span> {submission.id}
            </span>
            <span>
              <span style={{ color: "var(--ink-4)" }}>created</span>{" "}
              {new Date(submission.created_at).toLocaleTimeString("en-US", {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
            {submission.completed_at && (
              <span>
                <span style={{ color: "var(--ink-4)" }}>completed</span>{" "}
                {new Date(submission.completed_at).toLocaleTimeString("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            )}
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {isComplete && (
            <Link href={`/cert/${submission.id}`} className="pg-btn pg-btn-primary">
              View certificate →
            </Link>
          )}
          <Link href={`/grade/${submission.id}`} className="pg-btn pg-btn-ghost">
            ← Back to capture
          </Link>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: 22,
          alignItems: "center",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-start",
            gap: 2,
          }}
        >
          <span
            className="mono"
            style={{
              fontSize: 9,
              color: "var(--ink-3)",
              letterSpacing: "0.22em",
              textTransform: "uppercase",
            }}
          >
            elapsed
          </span>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span
              style={{
                fontFamily: "var(--font-instrument-serif), serif",
                fontSize: 56,
                lineHeight: 0.9,
                color: isFailed ? "var(--rose)" : "var(--ink-0)",
                letterSpacing: "-0.02em",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {elapsedSeconds.toFixed(1)}
            </span>
            <span className="mono" style={{ fontSize: 11, color: "var(--ink-3)" }}>
              s
            </span>
          </div>
        </div>

        <SegmentedProgress submission={submission} />
      </div>
    </header>
  );
}

function computeElapsedSeconds(
  submission: SubmissionOut,
  mountTime: number,
  _tick: number,
): number {
  if (submission.completed_at) {
    return (
      (new Date(submission.completed_at).getTime() -
        new Date(submission.created_at).getTime()) /
      1000
    );
  }
  if (submission.status === "processing") {
    return (Date.now() - new Date(submission.created_at).getTime()) / 1000;
  }
  return (Date.now() - mountTime) / 1000;
}

function SegmentedProgress({ submission }: { submission: SubmissionOut }) {
  const stageStates = deriveStageStates(submission);
  return (
    <div>
      <div
        style={{
          display: "flex",
          gap: 3,
          height: 14,
          position: "relative",
          borderRadius: 2,
          overflow: "hidden",
        }}
      >
        {stageStates.map((s) => {
          const bg =
            s.status === "done"
              ? "var(--lime-deep)"
              : s.status === "running"
                ? "var(--bg-3)"
                : s.status === "failed"
                  ? "rgba(251,113,133,0.4)"
                  : "var(--bg-3)";
          const fill =
            s.status === "running"
              ? "var(--lime)"
              : s.status === "failed"
                ? "var(--rose)"
                : "transparent";
          return (
            <div
              key={s.k}
              style={{
                flex: 1,
                position: "relative",
                background: bg,
                borderRadius: 1,
              }}
            >
              {s.status === "running" && (
                <div
                  style={{
                    position: "absolute",
                    inset: 0,
                    background: "linear-gradient(90deg, transparent, var(--lime), transparent)",
                    backgroundSize: "200% 100%",
                    animation: "pipeShimmer 1.6s linear infinite",
                  }}
                />
              )}
              {s.status === "done" && (
                <div style={{ position: "absolute", inset: 0, background: fill }} />
              )}
            </div>
          );
        })}
      </div>
      <div style={{ display: "flex", gap: 3, marginTop: 6 }}>
        {stageStates.map((s) => (
          <div key={s.k} style={{ flex: 1 }}>
            <span
              className="mono"
              style={{
                fontSize: 9,
                color:
                  s.status === "done"
                    ? "var(--ink-2)"
                    : s.status === "running"
                      ? "var(--lime)"
                      : "var(--ink-4)",
                letterSpacing: "0.04em",
                textTransform: "uppercase",
              }}
            >
              {s.l}
            </span>
          </div>
        ))}
      </div>
      <style>{`
        @keyframes pipeShimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}

// --------------------------------------------------------------------------
// Stage list (left column)
// --------------------------------------------------------------------------

function StageList({ stageStates }: { stageStates: StageState[] }) {
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <SectionLabel hint={`${stageStates.length} stages`}>Pipeline stages</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {stageStates.map((s) => (
          <StageRow key={s.k} stage={s} />
        ))}
      </div>
    </section>
  );
}

function StageRow({ stage }: { stage: StageState }) {
  const tone =
    stage.status === "done"
      ? "var(--lime)"
      : stage.status === "running"
        ? "var(--lime)"
        : stage.status === "failed"
          ? "var(--rose)"
          : "var(--ink-5)";
  return (
    <div
      style={{
        padding: "12px 14px",
        background: stage.status === "running" ? "var(--bg-2)" : "transparent",
        border: `1px solid ${stage.status === "running" ? "var(--line)" : "var(--line-2)"}`,
        borderRadius: 5,
        display: "flex",
        flexDirection: "column",
        gap: 8,
        transition: "all 200ms",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <StageGlyph status={stage.status} />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap" }}>
            <span
              className="mono"
              style={{ fontSize: 9, color: "var(--ink-4)", letterSpacing: "0.18em" }}
            >
              {String(stage.idx + 1).padStart(2, "0")}
            </span>
            <span
              style={{
                fontSize: 13,
                fontWeight: 500,
                color: stage.status === "pending" ? "var(--ink-3)" : "var(--ink-0)",
              }}
            >
              {stage.l}
            </span>
            <span
              className="mono"
              style={{ fontSize: 9, color: "var(--ink-4)", letterSpacing: "0.04em" }}
            >
              · {stage.code}
            </span>
          </div>
          <div
            style={{
              fontSize: 12,
              color: "var(--ink-3)",
              marginTop: 2,
              lineHeight: 1.4,
            }}
          >
            {stage.desc}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            className="mono"
            style={{
              fontSize: 10,
              color: tone,
              fontVariantNumeric: "tabular-nums",
              textTransform: "uppercase",
              letterSpacing: "0.1em",
            }}
          >
            {stage.status}
          </div>
          <div
            className="mono"
            style={{ fontSize: 9, color: "var(--ink-4)", marginTop: 2 }}
          >
            {stage.model}
          </div>
        </div>
      </div>

      {(stage.status === "running" || stage.status === "done") && (
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginLeft: 30 }}>
          {stage.sub.map((s, i) => {
            const sDone = stage.status === "done";
            return (
              <span
                key={i}
                className="mono"
                style={{
                  fontSize: 9,
                  padding: "2px 6px",
                  borderRadius: 2,
                  color: sDone ? "var(--lime)" : "var(--ink-1)",
                  background: sDone ? "transparent" : "var(--bg-3)",
                  border: sDone
                    ? "1px solid rgba(190,242,100,0.3)"
                    : "1px solid var(--line-2)",
                  letterSpacing: "0.02em",
                }}
              >
                {sDone ? "✓ " : "› "}
                {s}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StageGlyph({ status }: { status: StageStatus }) {
  const size = 22;
  if (status === "done") {
    return (
      <div
        aria-hidden
        style={{
          width: size,
          height: size,
          borderRadius: 12,
          background: "rgba(190,242,100,0.12)",
          border: "1px solid rgba(190,242,100,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
          <path
            d="M2 6.5L5 9L10 3"
            stroke="var(--lime)"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }
  if (status === "running") {
    return (
      <div
        aria-hidden
        style={{
          width: size,
          height: size,
          borderRadius: 12,
          background: "rgba(190,242,100,0.08)",
          border: "1px solid rgba(190,242,100,0.6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          animation: "stageRun 1.6s ease-in-out infinite",
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: 4,
            background: "var(--lime)",
            boxShadow: "0 0 6px var(--lime)",
          }}
        />
      </div>
    );
  }
  if (status === "failed") {
    return (
      <div
        aria-hidden
        style={{
          width: size,
          height: size,
          borderRadius: 12,
          background: "rgba(251,113,133,0.12)",
          border: "1px solid rgba(251,113,133,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
        }}
      >
        <svg width="11" height="11" viewBox="0 0 12 12" fill="none">
          <path
            d="M3 3L9 9M9 3L3 9"
            stroke="var(--rose)"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
      </div>
    );
  }
  return (
    <div
      aria-hidden
      style={{
        width: size,
        height: size,
        borderRadius: 12,
        background: "transparent",
        border: "1px dashed var(--ink-5)",
        flexShrink: 0,
      }}
    />
  );
}

// --------------------------------------------------------------------------
// Focus panel (right column) — per-stage detail card
// --------------------------------------------------------------------------

function FocusPanel({
  stageStates,
  submission,
}: {
  stageStates: StageState[];
  submission: SubmissionOut;
}) {
  if (submission.status === "completed") {
    return <FocusComplete submission={submission} />;
  }
  if (submission.status === "failed" || submission.status === "rejected_quality") {
    return <FocusFailed submission={submission} />;
  }

  const running = stageStates.find((s) => s.status === "running");
  // STAGES is a static, non-empty constant so stageStates[0] is always
  // defined; the assertion saves a redundant null-guard downstream.
  const focus = running ?? stageStates[0]!;

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <SectionLabel hint={`stage ${focus.idx + 1} of ${stageStates.length}`}>
        {running ? "Now running" : "Stage detail"}
      </SectionLabel>
      <Card>
        <KV k="Stage" v={focus.l} />
        <KV k="Event code" v={<span className="mono">{focus.code}</span>} />
        <KV k="Description" v={focus.desc} />
        <KV k="Model" v={focus.model} />
      </Card>
      <Card>
        <h4
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-3)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            margin: "0 0 8px",
            fontWeight: 500,
          }}
        >
          Sub-steps
        </h4>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          {focus.sub.map((s) => (
            <div
              key={s}
              style={{
                display: "flex",
                gap: 10,
                alignItems: "baseline",
                fontSize: 12,
                color: "var(--ink-2)",
              }}
            >
              <span style={{ color: "var(--ink-4)" }}>·</span>
              <span>{s}</span>
            </div>
          ))}
        </div>
      </Card>
    </section>
  );
}

function FocusComplete({ submission }: { submission: SubmissionOut }) {
  // The trim `SubmissionOut` shape on this endpoint doesn't carry the
  // numeric grade (only status); the cert page is where the final
  // grade renders. We just confirm completion and link out.
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <SectionLabel hint="grade row persisted">Pipeline complete</SectionLabel>
      <div
        style={{
          padding: "32px 28px",
          background:
            "linear-gradient(135deg, rgba(190,242,100,0.06) 0%, transparent 60%), var(--bg-1)",
          border: "1px solid rgba(190,242,100,0.4)",
          borderRadius: 8,
          display: "flex",
          flexDirection: "column",
          gap: 18,
          alignItems: "flex-start",
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--lime)",
            letterSpacing: "0.22em",
            textTransform: "uppercase",
          }}
        >
          ✓ Grade composed
        </span>
        <h3
          style={{
            margin: 0,
            fontFamily: "var(--font-instrument-serif), serif",
            fontSize: 36,
            color: "var(--ink-0)",
            letterSpacing: "-0.02em",
            lineHeight: 1.05,
          }}
        >
          The pipeline finished cleanly.
        </h3>
        <p
          style={{
            margin: 0,
            fontSize: 13,
            color: "var(--ink-2)",
            lineHeight: 1.5,
          }}
        >
          The certificate page renders the full evidence trail: subgrades +
          confidence band, region heatmap, defect log, per-detector
          authenticity breakdown, and pricing comps when configured.
        </p>
        <Link
          href={`/cert/${submission.id}`}
          className="pg-btn pg-btn-primary"
          style={{ padding: "10px 18px", fontSize: 13 }}
        >
          View certificate →
        </Link>
      </div>
    </section>
  );
}

function FocusFailed({ submission }: { submission: SubmissionOut }) {
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <SectionLabel>Pipeline failed</SectionLabel>
      <div
        style={{
          padding: "24px 28px",
          background: "rgba(251,113,133,0.04)",
          border: "1px solid rgba(251,113,133,0.4)",
          borderRadius: 8,
          display: "flex",
          flexDirection: "column",
          gap: 14,
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--rose)",
            letterSpacing: "0.22em",
            textTransform: "uppercase",
          }}
        >
          {submission.status === "rejected_quality"
            ? "rejected at quality gate"
            : "pipeline failed"}
        </span>
        {submission.rejection_reason && (
          <p
            style={{
              margin: 0,
              fontSize: 13,
              color: "var(--ink-1)",
              lineHeight: 1.5,
            }}
          >
            {submission.rejection_reason}
          </p>
        )}
        <p
          style={{
            margin: 0,
            fontSize: 12,
            color: "var(--ink-3)",
            lineHeight: 1.5,
          }}
        >
          The submission can be retaken from the capture wizard. The
          orphan-task reconciler will also flip wedged PROCESSING rows
          to FAILED automatically after 5 minutes of no progress.
        </p>
        <Link
          href={`/grade/${submission.id}`}
          className="pg-btn pg-btn-ghost"
          style={{ padding: "8px 14px", fontSize: 12 }}
        >
          ← Back to capture
        </Link>
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

function Card({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 6,
        padding: 16,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      {children}
    </div>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "120px 1fr",
        gap: 10,
        padding: "6px 0",
        borderBottom: "1px solid var(--line-2)",
        alignItems: "baseline",
      }}
    >
      <span
        className="mono"
        style={{
          fontSize: 10,
          color: "var(--ink-3)",
          letterSpacing: "0.16em",
          textTransform: "uppercase",
        }}
      >
        {k}
      </span>
      <span style={{ fontSize: 13, color: "var(--ink-1)" }}>{v}</span>
    </div>
  );
}

function SectionLabel({
  children,
  hint,
}: {
  children: React.ReactNode;
  hint?: string;
}) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "baseline",
        borderBottom: "1px solid var(--line-2)",
        paddingBottom: 8,
      }}
    >
      <h2
        className="mono"
        style={{
          fontSize: 11,
          color: "var(--ink-2)",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          margin: 0,
          fontWeight: 500,
        }}
      >
        {children}
      </h2>
      {hint && (
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-4)",
            letterSpacing: "0.04em",
          }}
        >
          {hint}
        </span>
      )}
    </div>
  );
}
