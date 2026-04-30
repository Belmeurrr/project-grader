/**
 * `/grade/[id]` — capture wizard + status page.
 *
 * Single page that adapts to the submission's lifecycle:
 *   PENDING / CAPTURING  → multi-shot capture wizard (this is where users spend most of their time)
 *   PROCESSING           → "grading in progress" with polling
 *   COMPLETED            → redirect to /cert/[id]
 *   FAILED / REJECTED    → terminal error state
 *
 * Capture flow per shot:
 *   1. User clicks "Capture <kind>"
 *   2. Browser camera streams into a <video>; user clicks shutter
 *   3. Frame is drawn onto a canvas and JPEG-encoded → Blob
 *   4. Show preview; "Use this" or "Retake"
 *   5. On accept: presign → PUT to S3 → register (server runs quality gate)
 *   6. quality_passed=true: shot saved, advance. Otherwise: show reasons, prompt retake.
 *
 * Wizard order is front → back → TL → TR → BR → BL (clockwise corners).
 * Only FRONT_FULL is strictly required by the API (REQUIRED_SHOTS in
 * pipeline_runner.py); everything else is signal that improves the
 * grade. The "Submit for grading" button enables once front passes.
 *
 * Camera fallback: if getUserMedia is unavailable or the user denies
 * permission, we fall back to a hidden <input type="file"
 * accept="image/*" capture="environment"> so the page still works
 * (especially on iOS Safari, which historically had quirks).
 *
 * No design polish — Tailwind defaults, functional only. The
 * capture-guidance acceptance test (target <2 retakes median) is
 * tracked separately as ongoing operational hygiene.
 */

"use client";

import { use, useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

import {
  ApiError,
  type ShotKind,
  type ShotOut,
  type SubmissionOut,
  type SubmissionStatus,
  getSubmission,
  submitForGrading,
  uploadShot,
} from "@/lib/submission";

// --------------------------------------------------------------------------
// Wizard ordering + labels
// --------------------------------------------------------------------------

type WizardShot = {
  kind: ShotKind;
  label: string;
  hint: string;
  required: boolean;
};

const WIZARD_SHOTS: WizardShot[] = [
  {
    kind: "front_full",
    label: "Front",
    hint: "Whole front, fill the frame, edges parallel.",
    required: true,
  },
  {
    kind: "back_full",
    label: "Back",
    hint: "Whole back, same framing as the front.",
    required: false,
  },
  {
    kind: "corner_tl",
    label: "Top-left corner",
    hint: "Zoom on the top-left corner; corner centered, sharp.",
    required: false,
  },
  {
    kind: "corner_tr",
    label: "Top-right corner",
    hint: "Zoom on the top-right corner.",
    required: false,
  },
  {
    kind: "corner_br",
    label: "Bottom-right corner",
    hint: "Zoom on the bottom-right corner.",
    required: false,
  },
  {
    kind: "corner_bl",
    label: "Bottom-left corner",
    hint: "Zoom on the bottom-left corner.",
    required: false,
  },
];

// --------------------------------------------------------------------------
// Page
// --------------------------------------------------------------------------

type Params = { id: string };

export default function GradePage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { id } = use(params);
  const router = useRouter();

  const [submission, setSubmission] = useState<SubmissionOut | null | undefined>(
    undefined,
  );
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const sub = await getSubmission(id);
      setSubmission(sub);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [id]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // While processing, poll every 2s until terminal. On completion,
  // route to the public cert page (the canonical share artifact).
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

  if (submission === undefined) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-20 text-zinc-400">Loading…</main>
    );
  }
  if (submission === null) {
    return (
      <main className="mx-auto max-w-3xl px-6 py-20">
        <h1 className="text-2xl font-semibold text-zinc-200">Submission not found</h1>
        <p className="mt-2 text-zinc-400">
          This submission doesn&apos;t exist or isn&apos;t yours.
        </p>
      </main>
    );
  }

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-8 px-6 py-12">
      <Header submission={submission} />
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      <Body
        submission={submission}
        onShotsChanged={refresh}
        onSubmitted={refresh}
      />
    </main>
  );
}

// --------------------------------------------------------------------------
// Header + status-aware body
// --------------------------------------------------------------------------

function Header({ submission }: { submission: SubmissionOut }) {
  return (
    <header className="flex flex-col gap-2 border-b border-zinc-800 pb-6">
      <p className="text-xs uppercase tracking-widest text-lime-500">
        Submission
      </p>
      <h1 className="font-mono text-sm break-all text-zinc-100">
        {submission.id}
      </h1>
      <StatusPill status={submission.status} />
    </header>
  );
}

const STATUS_LABELS: Record<SubmissionStatus, string> = {
  pending: "Awaiting first shot",
  capturing: "Capture in progress",
  processing: "Grading…",
  completed: "Completed",
  failed: "Failed",
  rejected_quality: "Rejected — capture quality too low",
};

function StatusPill({ status }: { status: SubmissionStatus }) {
  const cls =
    status === "completed"
      ? "border-lime-500/40 bg-lime-500/10 text-lime-300"
      : status === "failed" || status === "rejected_quality"
        ? "border-red-500/40 bg-red-500/10 text-red-300"
        : status === "processing"
          ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
          : "border-zinc-700 bg-zinc-800/50 text-zinc-400";
  return (
    <span
      className={`mt-1 inline-flex w-fit items-center rounded-full border px-3 py-1 text-xs font-medium ${cls}`}
    >
      {STATUS_LABELS[status]}
    </span>
  );
}

function Body({
  submission,
  onShotsChanged,
  onSubmitted,
}: {
  submission: SubmissionOut;
  onShotsChanged: () => Promise<void>;
  onSubmitted: () => Promise<void>;
}) {
  const status = submission.status;

  if (status === "processing") {
    return (
      <section className="rounded-xl border border-amber-500/40 bg-amber-500/5 px-6 py-8 text-center text-amber-200">
        <p className="text-base font-medium">Grading in progress…</p>
        <p className="mt-2 text-sm text-amber-300/80">
          This page polls automatically. Typical run completes within ~20s.
        </p>
      </section>
    );
  }

  if (status === "failed" || status === "rejected_quality") {
    return (
      <section className="rounded-xl border border-red-500/40 bg-red-500/5 px-6 py-8 text-red-200">
        <p className="text-base font-medium">{STATUS_LABELS[status]}</p>
        {submission.rejection_reason && (
          <p className="mt-2 text-sm text-red-300/80">
            {submission.rejection_reason}
          </p>
        )}
      </section>
    );
  }

  // pending | capturing → wizard.
  return (
    <Wizard
      submissionId={submission.id}
      onShotsChanged={onShotsChanged}
      onSubmitted={onSubmitted}
    />
  );
}

// --------------------------------------------------------------------------
// Wizard
// --------------------------------------------------------------------------

type ShotState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "passed"; shot: ShotOut }
  | { phase: "failed"; shot: ShotOut; reasons: string[] };

function Wizard({
  submissionId,
  onShotsChanged,
  onSubmitted,
}: {
  submissionId: string;
  onShotsChanged: () => Promise<void>;
  onSubmitted: () => Promise<void>;
}) {
  // Per-shot client state. We don't read shot history back from the
  // server because the server-side ShotOut already carries everything
  // we render (passed flag + metadata reasons). On page reload the
  // submission row tells us whether grading can be submitted.
  const [shots, setShots] = useState<Record<ShotKind, ShotState>>(() =>
    Object.fromEntries(
      WIZARD_SHOTS.map((s) => [s.kind, { phase: "idle" } as ShotState]),
    ) as Record<ShotKind, ShotState>,
  );

  const [activeKind, setActiveKind] = useState<ShotKind | null>(null);
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
      await submitForGrading(submissionId);
      await onSubmitted();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setSubmitError(
          "Grading can't start yet — at least the front shot must pass quality.",
        );
      } else {
        setSubmitError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setSubmitting(false);
    }
  };

  const frontPassed = shots.front_full?.phase === "passed";

  return (
    <section className="flex flex-col gap-5">
      <ol className="flex flex-col gap-3">
        {WIZARD_SHOTS.map((cfg) => (
          <ShotRow
            key={cfg.kind}
            cfg={cfg}
            state={shots[cfg.kind]}
            isActive={activeKind === cfg.kind}
            onActivate={() => setActiveKind(cfg.kind)}
            onCancel={() => setActiveKind(null)}
            submissionId={submissionId}
            onUploading={() => onShotUploading(cfg.kind)}
            onResult={(r) => onShotResult(cfg.kind, r)}
            onError={(m) => onShotError(cfg.kind, m)}
          />
        ))}
      </ol>

      <div className="flex flex-col gap-2 border-t border-zinc-800 pt-5">
        <button
          type="button"
          onClick={onSubmit}
          disabled={!frontPassed || submitting}
          className="self-start rounded-xl border border-lime-500/40 bg-lime-500/10 px-5 py-3 text-sm font-medium text-lime-300 transition-colors hover:bg-lime-500/20 disabled:cursor-not-allowed disabled:border-zinc-800 disabled:bg-transparent disabled:text-zinc-600"
        >
          {submitting ? "Submitting…" : "Submit for grading"}
        </button>
        {!frontPassed && (
          <p className="text-xs text-zinc-500">
            The front shot must pass quality before grading can run. Other
            shots are optional but improve the grade.
          </p>
        )}
        {submitError && (
          <p className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {submitError}
          </p>
        )}
      </div>
    </section>
  );
}

// --------------------------------------------------------------------------
// Per-shot row (idle / capturing / uploading / passed / failed)
// --------------------------------------------------------------------------

function ShotRow({
  cfg,
  state,
  isActive,
  onActivate,
  onCancel,
  submissionId,
  onUploading,
  onResult,
  onError,
}: {
  cfg: WizardShot;
  state: ShotState;
  isActive: boolean;
  onActivate: () => void;
  onCancel: () => void;
  submissionId: string;
  onUploading: () => void;
  onResult: (r: ShotOut) => void;
  onError: (msg: string) => void;
}) {
  const passed = state.phase === "passed";
  const failed = state.phase === "failed";
  const uploading = state.phase === "uploading";

  return (
    <li
      className={`rounded-xl border p-5 transition-colors ${
        passed
          ? "border-lime-500/40 bg-lime-500/5"
          : failed
            ? "border-red-500/40 bg-red-500/5"
            : "border-zinc-800"
      }`}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm font-medium text-zinc-200">
            {cfg.label}
            {cfg.required && (
              <span className="ml-2 text-xs uppercase tracking-wide text-lime-500">
                required
              </span>
            )}
          </p>
          <p className="mt-1 text-xs text-zinc-500">{cfg.hint}</p>
          {failed && state.phase === "failed" && (
            <ul className="mt-2 space-y-0.5 text-xs text-red-300">
              {state.reasons.map((r, i) => (
                <li key={i}>• {r}</li>
              ))}
            </ul>
          )}
          {passed && state.phase === "passed" && (
            <p className="mt-2 text-xs text-lime-400">
              Quality passed ({state.shot.width_px}×{state.shot.height_px})
            </p>
          )}
        </div>
        {!isActive && !uploading && (
          <button
            type="button"
            onClick={onActivate}
            className="rounded-lg border border-zinc-700 px-3 py-2 text-xs text-zinc-200 hover:border-zinc-500"
          >
            {passed ? "Retake" : failed ? "Retake" : "Capture"}
          </button>
        )}
        {uploading && (
          <span className="text-xs text-zinc-400">Uploading…</span>
        )}
      </div>

      {isActive && (
        <CaptureSurface
          submissionId={submissionId}
          kind={cfg.kind}
          onCancel={onCancel}
          onUploading={onUploading}
          onResult={(r) => {
            onResult(r);
            onCancel();
          }}
          onError={(m) => {
            onError(m);
            onCancel();
          }}
        />
      )}
    </li>
  );
}

// --------------------------------------------------------------------------
// Capture surface — camera w/ file-input fallback
// --------------------------------------------------------------------------

function CaptureSurface({
  submissionId,
  kind,
  onCancel,
  onUploading,
  onResult,
  onError,
}: {
  submissionId: string;
  kind: ShotKind;
  onCancel: () => void;
  onUploading: () => void;
  onResult: (r: ShotOut) => void;
  onError: (msg: string) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Start the stream on mount; tear it down on unmount or after capture.
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

  // Revoke any preview URL when it changes or the surface unmounts —
  // browser leaks blob URLs forever otherwise.
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
      const result = await uploadShot(submissionId, kind, previewBlob);
      onResult(result);
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    }
  };

  return (
    <div className="mt-4 flex flex-col gap-3">
      {previewUrl ? (
        <PreviewPane
          url={previewUrl}
          onAccept={accept}
          onRetake={() => {
            URL.revokeObjectURL(previewUrl);
            setPreviewUrl(null);
            setPreviewBlob(null);
          }}
          onCancel={onCancel}
        />
      ) : (
        <LiveCamera
          videoRef={videoRef}
          cameraError={cameraError}
          onCapture={captureFromVideo}
          onFile={captureFromFile}
          onCancel={onCancel}
        />
      )}
    </div>
  );
}

function LiveCamera({
  videoRef,
  cameraError,
  onCapture,
  onFile,
  onCancel,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraError: string | null;
  onCapture: () => void;
  onFile: (f: File) => void;
  onCancel: () => void;
}) {
  return (
    <>
      {!cameraError && (
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="aspect-[3/4] w-full rounded-lg bg-black object-contain"
        />
      )}
      {cameraError && (
        <p className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-3 text-sm text-amber-300">
          {cameraError}
        </p>
      )}
      <div className="flex flex-wrap gap-2">
        {!cameraError && (
          <button
            type="button"
            onClick={onCapture}
            className="rounded-xl border border-lime-500/40 bg-lime-500/10 px-4 py-2 text-sm font-medium text-lime-300 hover:bg-lime-500/20"
          >
            Take photo
          </button>
        )}
        <label className="cursor-pointer rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:border-zinc-500">
          Upload from device
          <input
            type="file"
            accept="image/*"
            capture="environment"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFile(file);
            }}
          />
        </label>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-xl border border-zinc-800 px-4 py-2 text-sm text-zinc-400 hover:border-zinc-700"
        >
          Cancel
        </button>
      </div>
    </>
  );
}

function PreviewPane({
  url,
  onAccept,
  onRetake,
  onCancel,
}: {
  url: string;
  onAccept: () => void;
  onRetake: () => void;
  onCancel: () => void;
}) {
  return (
    <>
      {/* Browser-native img is fine here — local blob URL, no remote loader.
          eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={url}
        alt="Captured shot preview"
        className="aspect-[3/4] w-full rounded-lg bg-black object-contain"
      />
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={onAccept}
          className="rounded-xl border border-lime-500/40 bg-lime-500/10 px-4 py-2 text-sm font-medium text-lime-300 hover:bg-lime-500/20"
        >
          Use this
        </button>
        <button
          type="button"
          onClick={onRetake}
          className="rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:border-zinc-500"
        >
          Retake
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="rounded-xl border border-zinc-800 px-4 py-2 text-sm text-zinc-400 hover:border-zinc-700"
        >
          Cancel
        </button>
      </div>
    </>
  );
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------

function ErrorBanner({
  message,
  onDismiss,
}: {
  message: string;
  onDismiss: () => void;
}) {
  return (
    <div className="flex items-start justify-between gap-3 rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
      <span>{message}</span>
      <button
        type="button"
        onClick={onDismiss}
        className="text-xs uppercase tracking-wide text-red-400 hover:text-red-200"
      >
        Dismiss
      </button>
    </div>
  );
}

/**
 * Pull human-readable failure reasons out of a quality_metadata blob.
 * The server's ShotOut.quality_metadata is freeform, but in practice
 * the quality service writes a `reasons: string[]` field for any failed
 * gate (blur/glare/perspective). Fall back to a generic message if the
 * key isn't there — better to surface "quality failed" than nothing.
 */
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
