"use client";

/**
 * Capture wizard internals — extracted from `apps/web/app/grade/[id]/page.tsx`
 * because Next 15 enforces a strict whitelist of named exports for page
 * files. The page file keeps its routing shell (`<GradePage>`) and the
 * top-level header; everything from `<Body>` down lives here so we can
 * also import it directly into the unit-test path.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type ShotKind,
  type ShotOut,
  type SubmissionOut,
  type SubmissionStatus,
  submitForGrading,
  uploadShot,
  type AuthedFetch,
} from "@/lib/submission";
import { WIZARD_SHOTS, type WizardShot } from "@/components/grade/shots";
import { ShotRail, type ShotRailRow } from "@/components/grade/ShotRail";
import { PoseGuide } from "@/components/grade/PoseGuide";

// Status labels shared with `<Header>` in page.tsx — exported so the
// page module doesn't have to duplicate them.
export const STATUS_LABELS: Record<SubmissionStatus, string> = {
  pending: "Awaiting first shot",
  capturing: "Capture in progress",
  processing: "Grading…",
  completed: "Completed",
  failed: "Failed",
  rejected_quality: "Rejected — capture quality too low",
};

type ShotState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "passed"; shot: ShotOut }
  | { phase: "failed"; shot: ShotOut; reasons: string[] };

// --------------------------------------------------------------------------
// Body — status-aware top-level. Pending/capturing → wizard, processing →
// grading-pending, terminal → error pane.
// --------------------------------------------------------------------------

export function Body({
  submission,
  authedFetch,
  onShotsChanged,
  onSubmitted,
}: {
  submission: SubmissionOut;
  authedFetch: AuthedFetch;
  onShotsChanged: () => Promise<void>;
  onSubmitted: () => Promise<void>;
}) {
  const status = submission.status;

  if (status === "processing") {
    return (
      <section
        style={{
          padding: "32px 24px",
          background: "rgba(251,191,36,0.04)",
          border: "1px solid rgba(251,191,36,0.4)",
          borderRadius: 8,
          textAlign: "center",
          color: "var(--amber)",
        }}
      >
        <p
          style={{
            fontFamily: "var(--font-instrument-serif), serif",
            fontSize: 28,
            margin: 0,
            color: "var(--ink-0)",
          }}
        >
          Grading in progress…
        </p>
        <p
          className="mono"
          style={{
            marginTop: 10,
            fontSize: 11,
            color: "var(--ink-3)",
            letterSpacing: "0.04em",
          }}
        >
          this page polls automatically · typical run completes within ~20s
        </p>
      </section>
    );
  }

  if (status === "failed" || status === "rejected_quality") {
    return (
      <section
        style={{
          padding: "24px 24px",
          background: "rgba(251,113,133,0.04)",
          border: "1px solid rgba(251,113,133,0.4)",
          borderRadius: 8,
          color: "var(--rose)",
        }}
      >
        <p style={{ fontSize: 14, fontWeight: 600, margin: 0, color: "var(--ink-0)" }}>
          {STATUS_LABELS[status]}
        </p>
        {submission.rejection_reason && (
          <p
            style={{
              marginTop: 8,
              fontSize: 12,
              color: "var(--ink-2)",
              lineHeight: 1.5,
            }}
          >
            {submission.rejection_reason}
          </p>
        )}
      </section>
    );
  }

  return (
    <Wizard
      submissionId={submission.id}
      authedFetch={authedFetch}
      onShotsChanged={onShotsChanged}
      onSubmitted={onSubmitted}
    />
  );
}

// --------------------------------------------------------------------------
// Wizard — sidebar (ShotRail) + main viewfinder
// --------------------------------------------------------------------------

function Wizard({
  submissionId,
  authedFetch,
  onShotsChanged,
  onSubmitted,
}: {
  submissionId: string;
  authedFetch: AuthedFetch;
  onShotsChanged: () => Promise<void>;
  onSubmitted: () => Promise<void>;
}) {
  const [shots, setShots] = useState<Record<ShotKind, ShotState>>(() =>
    Object.fromEntries(
      WIZARD_SHOTS.map((s) => [s.kind, { phase: "idle" } as ShotState]),
    ) as Record<ShotKind, ShotState>,
  );
  const [activeKind, setActiveKind] = useState<ShotKind>(WIZARD_SHOTS[0]!.kind);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const onShotResult = useCallback(
    async (kind: ShotKind, result: ShotOut) => {
      const reasons = qualityReasons(result);
      setShots((prev) => ({
        ...prev,
        [kind]: result.quality_passed
          ? { phase: "passed", shot: result }
          : { phase: "failed", shot: result, reasons },
      }));
      await onShotsChanged();
    },
    [onShotsChanged],
  );

  const onShotError = useCallback((kind: ShotKind, message: string) => {
    setShots((prev) => ({
      ...prev,
      [kind]: { phase: "failed", shot: errorShot(kind), reasons: [message] },
    }));
  }, []);

  const onShotUploading = useCallback((kind: ShotKind) => {
    setShots((prev) => ({ ...prev, [kind]: { phase: "uploading" } }));
  }, []);

  const onSubmit = async () => {
    setSubmitting(true);
    setSubmitError(null);
    try {
      await submitForGrading(authedFetch, submissionId);
      await onSubmitted();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setSubmitError(
          "Grading can't start yet — the front shot must pass quality before grading can run.",
        );
      } else {
        setSubmitError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setSubmitting(false);
    }
  };

  const frontPassed = shots.front_full?.phase === "passed";
  const activeShot = WIZARD_SHOTS.find((s) => s.kind === activeKind) ?? WIZARD_SHOTS[0]!;
  const activeState = shots[activeKind] ?? { phase: "idle" };

  const railRows: ShotRailRow[] = WIZARD_SHOTS.map((s, idx) => ({
    ...s,
    idx,
    status:
      shots[s.kind]?.phase === "passed"
        ? "captured"
        : shots[s.kind]?.phase === "failed"
          ? "failed"
          : s.kind === activeKind
            ? "active"
            : "pending",
  }));

  return (
    <section
      className="grade-grid"
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 280px) minmax(0, 1fr)",
        gap: 24,
        alignItems: "start",
      }}
    >
      <ShotRail rows={railRows} onSelect={setActiveKind} />

      <div style={{ display: "flex", flexDirection: "column", gap: 18, minWidth: 0 }}>
        <ViewfinderHeader shot={activeShot} state={activeState} />
        <CaptureSurface
          submissionId={submissionId}
          authedFetch={authedFetch}
          shot={activeShot}
          state={activeState}
          onUploading={() => onShotUploading(activeShot.kind)}
          onResult={(r) => onShotResult(activeShot.kind, r)}
          onError={(m) => onShotError(activeShot.kind, m)}
        />
        <ShotTips shot={activeShot} state={activeState} />
        <SubmitBar
          frontPassed={frontPassed}
          submitting={submitting}
          submitError={submitError}
          onSubmit={onSubmit}
        />
      </div>

      <style>{`
        @media (max-width: 880px) {
          .grade-grid {
            grid-template-columns: 1fr !important;
          }
        }
      `}</style>
    </section>
  );
}

function ViewfinderHeader({ shot, state }: { shot: WizardShot; state: ShotState }) {
  const lightLabel: Record<typeof shot.light, string> = {
    ambient: "ambient",
    flash: "flash",
    macro: "macro",
  };
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-start",
        gap: 16,
        flexWrap: "wrap",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-3)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
          }}
        >
          Active shot · {shot.kind}
        </span>
        <h2
          style={{
            margin: 0,
            fontFamily: "var(--font-instrument-serif), serif",
            fontSize: 28,
            color: "var(--ink-0)",
            letterSpacing: "-0.01em",
          }}
        >
          {shot.label}
        </h2>
        <p style={{ margin: 0, fontSize: 13, color: "var(--ink-2)", lineHeight: 1.5 }}>
          {shot.description}
        </p>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, alignItems: "flex-end" }}>
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-3)",
            letterSpacing: "0.16em",
            textTransform: "uppercase",
            padding: "3px 8px",
            background: "var(--bg-2)",
            border: "1px solid var(--line)",
            borderRadius: 3,
          }}
        >
          {lightLabel[shot.light]}
        </span>
        <PhaseLabel state={state} />
      </div>
    </div>
  );
}

function PhaseLabel({ state }: { state: ShotState }) {
  const map: Record<ShotState["phase"], { l: string; tone: string }> = {
    idle: { l: "ready", tone: "var(--ink-3)" },
    uploading: { l: "uploading…", tone: "var(--amber)" },
    passed: { l: "captured", tone: "var(--lime)" },
    failed: { l: "retake", tone: "var(--rose)" },
  };
  const m = map[state.phase];
  return (
    <span
      className="mono"
      style={{
        fontSize: 10,
        color: m.tone,
        letterSpacing: "0.18em",
        textTransform: "uppercase",
      }}
    >
      · {m.l}
    </span>
  );
}

function CaptureSurface({
  submissionId,
  authedFetch,
  shot,
  state,
  onUploading,
  onResult,
  onError,
}: {
  submissionId: string;
  authedFetch: AuthedFetch;
  shot: WizardShot;
  state: ShotState;
  onUploading: () => void;
  onResult: (r: ShotOut) => void;
  onError: (msg: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  useEffect(() => {
    setPreviewUrl(null);
    setPreviewBlob(null);
  }, [shot.kind]);

  useEffect(() => {
    let mounted = true;
    const start = async () => {
      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        setCameraError("Camera API not available in this browser.");
        return;
      }
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" } },
          audio: false,
        });
        if (!mounted) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (e) {
        setCameraError(
          e instanceof Error
            ? `Camera unavailable: ${e.message}. Use the file picker below.`
            : "Camera unavailable. Use the file picker below.",
        );
      }
    };
    void start();
    return () => {
      mounted = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const captureFromVideo = async () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      onError("Could not get 2D context.");
      return;
    }
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob((b) => resolve(b), "image/jpeg", 0.92),
    );
    if (!blob) {
      onError("Could not encode frame.");
      return;
    }
    setPreviewBlob(blob);
    setPreviewUrl(URL.createObjectURL(blob));
  };

  const captureFromFile = (file: File) => {
    setPreviewBlob(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const accept = async () => {
    if (!previewBlob) return;
    onUploading();
    try {
      const result = await uploadShot(authedFetch, submissionId, shot.kind, previewBlob);
      onResult(result);
      setPreviewUrl(null);
      setPreviewBlob(null);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  const reset = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setPreviewBlob(null);
  };

  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 10,
        padding: 16,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          position: "relative",
          aspectRatio: "4/3",
          background: "#000",
          borderRadius: 6,
          overflow: "hidden",
          border: "1px solid var(--line-2)",
        }}
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="Captured shot preview"
            style={{ width: "100%", height: "100%", objectFit: "contain" }}
          />
        ) : !cameraError ? (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "var(--ink-4)",
              fontSize: 11,
              padding: 24,
              textAlign: "center",
            }}
          >
            <span className="mono" style={{ letterSpacing: "0.12em", textTransform: "uppercase" }}>
              {cameraError}
            </span>
          </div>
        )}
        <PoseGuide pose={shot.pose} guideColor="rgba(190,242,100,0.55)" />
      </div>

      {state.phase === "uploading" ? (
        <UploadingRow />
      ) : previewUrl ? (
        <PreviewActions onAccept={accept} onRetake={reset} />
      ) : (
        <CaptureActions
          cameraReady={!cameraError}
          onCapture={captureFromVideo}
          onFile={captureFromFile}
        />
      )}

      {state.phase === "failed" && <FailedReadout reasons={state.reasons} />}
      {state.phase === "passed" && <PassedReadout shot={state.shot} />}
    </div>
  );
}

function CaptureActions({
  cameraReady,
  onCapture,
  onFile,
}: {
  cameraReady: boolean;
  onCapture: () => void;
  onFile: (f: File) => void;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      {cameraReady && (
        <button type="button" onClick={onCapture} className="pg-btn pg-btn-primary">
          Take photo
        </button>
      )}
      <label className="pg-btn pg-btn-ghost" style={{ cursor: "pointer" }}>
        Upload from device
        <input
          type="file"
          accept="image/*"
          capture="environment"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) onFile(file);
          }}
        />
      </label>
    </div>
  );
}

function PreviewActions({ onAccept, onRetake }: { onAccept: () => void; onRetake: () => void }) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <button type="button" onClick={onAccept} className="pg-btn pg-btn-primary">
        Use this
      </button>
      <button type="button" onClick={onRetake} className="pg-btn pg-btn-ghost">
        Retake
      </button>
    </div>
  );
}

function UploadingRow() {
  return (
    <div
      className="mono"
      style={{
        fontSize: 11,
        color: "var(--amber)",
        letterSpacing: "0.16em",
        textTransform: "uppercase",
        padding: "8px 4px",
      }}
    >
      uploading · running quality gate…
    </div>
  );
}

function PassedReadout({ shot }: { shot: ShotOut }) {
  return (
    <div
      style={{
        padding: "10px 12px",
        background: "rgba(190,242,100,0.04)",
        border: "1px solid rgba(190,242,100,0.4)",
        borderRadius: 4,
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 12,
        flexWrap: "wrap",
      }}
    >
      <span
        className="mono"
        style={{
          fontSize: 11,
          color: "var(--lime)",
          letterSpacing: "0.16em",
          textTransform: "uppercase",
        }}
      >
        ✓ quality passed
      </span>
      <span className="mono" style={{ fontSize: 10, color: "var(--ink-2)" }}>
        {shot.width_px}×{shot.height_px}
      </span>
    </div>
  );
}

function FailedReadout({ reasons }: { reasons: string[] }) {
  return (
    <div
      style={{
        padding: "10px 12px",
        background: "rgba(251,113,133,0.04)",
        border: "1px solid rgba(251,113,133,0.4)",
        borderRadius: 4,
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      <span
        className="mono"
        style={{
          fontSize: 11,
          color: "var(--rose)",
          letterSpacing: "0.16em",
          textTransform: "uppercase",
        }}
      >
        ! quality gate failed
      </span>
      {reasons.length > 0 && (
        <ul
          style={{
            margin: 0,
            padding: 0,
            paddingLeft: 14,
            color: "var(--ink-2)",
            fontSize: 12,
            lineHeight: 1.5,
          }}
        >
          {reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ShotTips({ shot, state }: { shot: WizardShot; state: ShotState }) {
  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 6,
        padding: 14,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          paddingBottom: 8,
          borderBottom: "1px solid var(--line-2)",
          marginBottom: 10,
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-2)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
          }}
        >
          Capture tips
        </span>
        <span className="mono" style={{ fontSize: 10, color: "var(--ink-4)" }}>
          {shot.required ? "required" : "optional"}
        </span>
      </div>
      <ul
        style={{
          margin: 0,
          padding: 0,
          listStyle: "none",
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        {shot.tips.map((t, i) => (
          <li
            key={i}
            style={{
              display: "flex",
              gap: 8,
              fontSize: 12,
              color: "var(--ink-2)",
              lineHeight: 1.5,
            }}
          >
            <span style={{ color: "var(--lime)", flexShrink: 0 }}>·</span>
            <span>{t}</span>
          </li>
        ))}
      </ul>
      {state.phase === "passed" && (
        <p
          className="mono"
          style={{
            marginTop: 10,
            fontSize: 10,
            color: "var(--lime)",
            letterSpacing: "0.04em",
          }}
        >
          tip: pick another shot from the rail to keep going, or jump to submit.
        </p>
      )}
    </div>
  );
}

function SubmitBar({
  frontPassed,
  submitting,
  submitError,
  onSubmit,
}: {
  frontPassed: boolean;
  submitting: boolean;
  submitError: string | null;
  onSubmit: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 10,
        paddingTop: 12,
        borderTop: "1px solid var(--line-2)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap" }}>
        <button
          type="button"
          onClick={onSubmit}
          disabled={!frontPassed || submitting}
          className="pg-btn pg-btn-primary"
          style={{
            padding: "10px 18px",
            fontSize: 13,
            opacity: !frontPassed || submitting ? 0.5 : 1,
            cursor: !frontPassed || submitting ? "not-allowed" : "pointer",
          }}
        >
          {submitting ? "Submitting…" : "Submit for grading"}
        </button>
        {!frontPassed && (
          <span
            className="mono"
            style={{ fontSize: 11, color: "var(--ink-3)", letterSpacing: "0.04em" }}
          >
            the front shot must pass quality before grading can run · other shots are optional but improve the grade
          </span>
        )}
      </div>
      {submitError && (
        <div
          style={{
            padding: "10px 12px",
            background: "rgba(251,113,133,0.06)",
            border: "1px solid rgba(251,113,133,0.4)",
            borderRadius: 4,
            color: "var(--rose)",
            fontSize: 12,
            lineHeight: 1.5,
          }}
        >
          {submitError}
        </div>
      )}
    </div>
  );
}

function qualityReasons(shot: ShotOut): string[] {
  const reasons = shot.quality_metadata?.reasons;
  if (Array.isArray(reasons) && reasons.every((r) => typeof r === "string")) {
    return reasons as string[];
  }
  if (shot.quality_passed) return [];
  return ["Quality gate failed; please retake."];
}

function errorShot(kind: ShotKind): ShotOut {
  return {
    id: "",
    kind,
    s3_key: "",
    width_px: 0,
    height_px: 0,
    quality_passed: false,
    quality_metadata: {},
  };
}
