/**
 * `/grade/[id]` — Workbench-design capture wizard (route entry).
 *
 * Status-aware page:
 *   PENDING / CAPTURING  → 8-shot capture wizard (sidebar + viewfinder)
 *   PROCESSING           → "grading in progress" with auto-poll
 *   COMPLETED            → redirect to /cert/[id]
 *   FAILED / REJECTED    → terminal error state
 *
 * The wizard internals + Body export live in
 * `apps/web/components/grade/Wizard.tsx` so the page module stays clean
 * (Next 15 only allows specific named exports from page files).
 *
 * Visual port of `docs/design/capture.jsx` (Split layout). Keeps the
 * existing API plumbing: presign → multipart POST to S3/MinIO →
 * register (server quality gate) → submit-for-grading. Polling at 2s
 * intervals while PROCESSING, redirect to `/cert/[id]` on COMPLETED.
 */

"use client";

import { use, useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import {
  type SubmissionOut,
  type SubmissionStatus,
  getSubmission,
  useAuthedFetch,
} from "@/lib/submission";
import { Body, STATUS_LABELS } from "@/components/grade/Wizard";

// --------------------------------------------------------------------------
// Page (route entry)
// --------------------------------------------------------------------------

type Params = { id: string };

export default function GradePage({ params }: { params: Promise<Params> }) {
  const { id } = use(params);
  const router = useRouter();
  const authedFetch = useAuthedFetch();

  const [submission, setSubmission] = useState<SubmissionOut | null | undefined>(
    undefined,
  );
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => {
    if (!submission) return;
    if (submission.status === "completed") {
      router.replace(`/cert/${submission.id}`);
      return;
    }
    if (submission.status !== "processing") return;
    const handle = setInterval(() => void refresh(), 2000);
    return () => clearInterval(handle);
  }, [submission, refresh, router]);

  if (submission === undefined) return <PageShell><LoadingState /></PageShell>;
  if (submission === null) return <PageShell><NotFoundState /></PageShell>;

  return (
    <PageShell>
      <Header submission={submission} />
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      <Body
        submission={submission}
        authedFetch={authedFetch}
        onShotsChanged={refresh}
        onSubmitted={refresh}
      />
    </PageShell>
  );
}

function PageShell({ children }: { children: React.ReactNode }) {
  return (
    <main
      style={{
        maxWidth: 1280,
        margin: "32px auto",
        padding: "0 32px 80px",
        display: "flex",
        flexDirection: "column",
        gap: 24,
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

const STATUS_TONE: Record<SubmissionStatus, { bg: string; border: string; fg: string; pulse: boolean }> = {
  pending: { bg: "var(--bg-2)", border: "var(--line)", fg: "var(--ink-3)", pulse: false },
  capturing: {
    bg: "rgba(190,242,100,0.06)",
    border: "rgba(190,242,100,0.4)",
    fg: "var(--lime)",
    pulse: false,
  },
  processing: {
    bg: "rgba(251,191,36,0.06)",
    border: "rgba(251,191,36,0.4)",
    fg: "var(--amber)",
    pulse: true,
  },
  completed: {
    bg: "rgba(190,242,100,0.06)",
    border: "rgba(190,242,100,0.4)",
    fg: "var(--lime)",
    pulse: false,
  },
  failed: {
    bg: "rgba(251,113,133,0.06)",
    border: "rgba(251,113,133,0.4)",
    fg: "var(--rose)",
    pulse: false,
  },
  rejected_quality: {
    bg: "rgba(251,113,133,0.06)",
    border: "rgba(251,113,133,0.4)",
    fg: "var(--rose)",
    pulse: false,
  },
};

function Header({ submission }: { submission: SubmissionOut }) {
  return (
    <header
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 12,
        paddingBottom: 18,
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--lime)",
            letterSpacing: "0.22em",
            textTransform: "uppercase",
          }}
        >
          Project Grader · Capture
        </span>
        <StatusPill status={submission.status} />
      </div>
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
        <h1
          style={{
            fontFamily: "var(--font-instrument-serif), serif",
            fontSize: 32,
            fontWeight: 400,
            color: "var(--ink-0)",
            margin: 0,
            letterSpacing: "-0.01em",
          }}
        >
          New submission
        </h1>
        <span
          className="mono"
          style={{
            fontSize: 11,
            color: "var(--ink-3)",
            letterSpacing: "0.04em",
          }}
        >
          {submission.id}
        </span>
      </div>
    </header>
  );
}

function StatusPill({ status }: { status: SubmissionStatus }) {
  const t = STATUS_TONE[status];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 10px",
        background: t.bg,
        border: `1px solid ${t.border}`,
        borderRadius: 99,
        color: t.fg,
        fontSize: 11,
        fontWeight: 500,
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: 3,
          background: t.fg,
          boxShadow: `0 0 6px ${t.fg}`,
          animation: t.pulse ? "pulse 1.6s ease-in-out infinite" : "none",
        }}
      />
      {STATUS_LABELS[status]}
    </span>
  );
}

function ErrorBanner({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 12,
        padding: "10px 14px",
        background: "rgba(251,113,133,0.04)",
        border: "1px solid rgba(251,113,133,0.4)",
        borderRadius: 4,
      }}
    >
      <span style={{ color: "var(--rose)", fontSize: 12, lineHeight: 1.5 }}>{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="mono"
        style={{
          background: "transparent",
          border: 0,
          cursor: "pointer",
          color: "var(--rose)",
          fontSize: 10,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
        }}
      >
        dismiss
      </button>
    </div>
  );
}
