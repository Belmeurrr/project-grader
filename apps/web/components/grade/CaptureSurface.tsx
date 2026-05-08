/**
 * Capture surface — wizard's per-shot viewfinder + actions.
 *
 * Clean rebuild. Lessons from the previous attempt:
 *
 * - Tap targets MUST live outside the viewfinder. The PoseGuide SVG +
 *   CoachingBanner + DetectedQuad polygon all sit inside the viewfinder
 *   and any one of them can intercept clicks if pointer-events isn't set
 *   right. The Enable camera / Take photo / etc. buttons live in the
 *   action row below the viewfinder where nothing can overlay them.
 *
 * - Camera acquisition is gated behind an explicit user-gesture button
 *   tap. iOS Safari (17+) silently no-ops getUserMedia outside a gesture.
 *
 * - Stream + video element are tracked as React state so the attach
 *   effect can fire on either changing — useRef alone misses remounts.
 *
 * - No visibility-stop-on-hidden handler. It races with iOS permission
 *   prompts (which can briefly transition the page to "hidden") and
 *   kills streams just before they're handed back. If a backgrounded
 *   stream goes black on return, the user can tap "Restart camera".
 *
 * Features kept from the OpenCV / coaching work:
 *   - Live quad detection via Canny + findContours + approxPolyDP
 *   - Polygon overlay with corner dots
 *   - Coaching banner with directional hints
 *   - Auto-capture (4 framed ticks → fires)
 *   - Auto-crop on capture (re-detects on the snapped frame so the
 *     crop is current, not 250ms-stale)
 */

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type AuthedFetch,
  type ShotKind,
  type ShotOut,
  uploadShot,
} from "@/lib/submission";
import {
  detectCardQuad,
  isOpenCVReady,
  loadOpenCV,
  quadBounds,
  type Quad,
} from "@/lib/cardDetect";
import { PoseGuide } from "@/components/grade/PoseGuide";
import type { WizardShot } from "@/components/grade/shots";

type ShotState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "passed"; shot: ShotOut }
  | { phase: "failed"; shot: ShotOut; reasons: string[] };

interface Props {
  submissionId: string;
  authedFetch: AuthedFetch;
  shot: WizardShot;
  state: ShotState;
  onUploading: () => void;
  onResult: (r: ShotOut) => void;
  onError: (msg: string) => void;
}

interface Detection {
  quad: Quad | null;
  framed: boolean;
  hint: string;
}

const TICK_MS = 250;
const FIRE_THRESHOLD = 4; // 4 framed ticks ≈ 1.0s before auto-fire
const SETTLE_MS = 220; // delay between tap and shutter to let hand jiggle settle

export function CaptureSurface({
  submissionId,
  authedFetch,
  shot,
  state,
  onUploading,
  onResult,
  onError,
}: Props) {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [videoEl, setVideoEl] = useState<HTMLVideoElement | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [autoMode, setAutoMode] = useState(false);
  const [videoAspect, setVideoAspect] = useState<number | null>(null);
  const [detection, setDetection] = useState<Detection>({
    quad: null,
    framed: false,
    hint: "",
  });

  // Reset preview when the wizard switches to a different shot.
  useEffect(() => {
    setPreviewUrl(null);
    setPreviewBlob(null);
  }, [shot.kind]);

  // Stop tracks on full unmount (page close / wizard exit).
  useEffect(() => {
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [stream]);

  // Lazy-load OpenCV.js *after* the camera is up. Eager loading on mount
  // was blocking the main thread (10MB WASM compile) at the moment the
  // user taps Enable camera, which on iOS Safari makes the gesture
  // context expire silently and getUserMedia never runs.
  useEffect(() => {
    if (!stream) return;
    void loadOpenCV().catch(() => {});
  }, [stream]);

  // Attach stream → video element. Single source of truth for srcObject.
  useEffect(() => {
    if (!videoEl || !stream) return;
    if (videoEl.srcObject !== stream) videoEl.srcObject = stream;
    const p = videoEl.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  }, [videoEl, stream]);

  // Pull video aspect ratio from the track settings (more reliable on
  // iOS than waiting for loadedmetadata).
  useEffect(() => {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings();
    if (settings?.width && settings?.height) {
      setVideoAspect(settings.width / settings.height);
    }
  }, [stream]);

  // Live OpenCV detection tick — only runs while the camera is live and
  // we're not showing a preview. Updates the polygon overlay and the
  // coaching hint, and triggers the auto-fire when framed steadily.
  const captureRef = useRef<() => Promise<void>>(async () => {});
  useEffect(() => {
    if (!stream || !videoEl || previewUrl || cameraError) return;
    const W = 320;
    const H = 240;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    let framedTicks = 0;
    let cancelled = false;

    const tick = () => {
      if (cancelled) return;
      if (videoEl.readyState < 2) return;
      if (!isOpenCVReady()) {
        setDetection({ quad: null, framed: false, hint: "Loading detector…" });
        return;
      }
      ctx.drawImage(videoEl, 0, 0, W, H);
      const data = ctx.getImageData(0, 0, W, H);
      let quad: Quad | null = null;
      try {
        quad = detectCardQuad(data);
      } catch {
        quad = null;
      }
      if (!quad) {
        framedTicks = 0;
        setDetection({
          quad: null,
          framed: false,
          hint: "Move closer — fit the card in the brackets",
        });
        return;
      }
      const b = quadBounds(quad, W, H);
      const fill = b.w * b.h;
      const cx = b.x + b.w / 2;
      const cy = b.y + b.h / 2;
      const sizeOk = fill > 0.18 && fill < 0.85;
      const centered =
        Math.abs(cx - 0.5) < 0.18 && Math.abs(cy - 0.5) < 0.18;
      const allGood = sizeOk && centered;

      let hint: string;
      if (allGood) {
        hint = autoMode
          ? "Locked — capturing"
          : "Card framed — tap Take photo";
      } else if (fill < 0.18) {
        hint = "Move closer";
      } else if (fill > 0.85) {
        hint = "Pull back a little";
      } else if (!centered) {
        const dx = cx - 0.5;
        const dy = cy - 0.5;
        hint =
          Math.abs(dx) > Math.abs(dy)
            ? dx > 0
              ? "Pan right"
              : "Pan left"
            : dy > 0
              ? "Pan down"
              : "Pan up";
      } else {
        hint = "Hold steady";
      }
      setDetection({ quad, framed: allGood, hint });

      if (allGood) framedTicks += 1;
      else framedTicks = 0;
      if (autoMode && framedTicks >= FIRE_THRESHOLD) {
        cancelled = true;
        void captureRef.current();
      }
    };
    const handle = window.setInterval(tick, TICK_MS);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
    };
  }, [stream, videoEl, previewUrl, cameraError, autoMode]);

  // -- Handlers -------------------------------------------------------

  const requestCamera = async () => {
    if (
      typeof navigator === "undefined" ||
      !navigator.mediaDevices?.getUserMedia
    ) {
      setCameraError("Camera API not available in this browser.");
      return;
    }
    setCameraError(null);
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      setStream(s);
    } catch (e) {
      setCameraError(
        e instanceof Error
          ? `Camera unavailable (${e.name}). Use the file picker instead.`
          : "Camera unavailable. Use the file picker instead.",
      );
    }
  };

  const restartCamera = async () => {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
    setVideoAspect(null);
    setDetection({ quad: null, framed: false, hint: "" });
    await requestCamera();
  };

  const captureFromVideo = useCallback(async () => {
    if (!videoEl || videoEl.readyState < 2) return;
    await new Promise((r) => setTimeout(r, SETTLE_MS));
    if (!videoEl || videoEl.readyState < 2) return;
    const fullW = videoEl.videoWidth;
    const fullH = videoEl.videoHeight;

    // Re-detect on the just-snapped frame (not the live overlay's
    // 250ms-stale rect) so the crop reflects where the card is now.
    let sx = 0;
    let sy = 0;
    let sw = fullW;
    let sh = fullH;
    if (isOpenCVReady()) {
      const probeW = 320;
      const probeH = 240;
      const probe = document.createElement("canvas");
      probe.width = probeW;
      probe.height = probeH;
      const pctx = probe.getContext("2d", { willReadFrequently: true });
      if (pctx) {
        pctx.drawImage(videoEl, 0, 0, probeW, probeH);
        try {
          const q = detectCardQuad(pctx.getImageData(0, 0, probeW, probeH));
          if (q) {
            const b = quadBounds(q, probeW, probeH);
            const margin = 0.12;
            const nx = Math.max(0, b.x - margin);
            const ny = Math.max(0, b.y - margin);
            const nw = Math.min(1 - nx, b.w + 2 * margin);
            const nh = Math.min(1 - ny, b.h + 2 * margin);
            const px = Math.round(nx * fullW);
            const py = Math.round(ny * fullH);
            const pw = Math.round(nw * fullW);
            const ph = Math.round(nh * fullH);
            if (pw >= 320 && ph >= 320) {
              sx = px;
              sy = py;
              sw = pw;
              sh = ph;
            }
          }
        } catch {
          // Detection failed; ship full frame.
        }
      }
    }

    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      onError("Could not get 2D context.");
      return;
    }
    ctx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, sw, sh);
    const blob = await new Promise<Blob | null>((res) =>
      canvas.toBlob((b) => res(b), "image/jpeg", 0.92),
    );
    if (!blob) {
      onError("Could not encode captured frame.");
      return;
    }
    setPreviewBlob(blob);
    setPreviewUrl(URL.createObjectURL(blob));
  }, [videoEl, onError]);

  // Keep captureRef pointing at the latest captureFromVideo so the
  // tick effect's auto-fire path always invokes the current closure.
  useEffect(() => {
    captureRef.current = captureFromVideo;
  }, [captureFromVideo]);

  const captureFromFile = (file: File) => {
    setPreviewBlob(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const acceptPreview = async () => {
    if (!previewBlob) return;
    onUploading();
    try {
      const result = await uploadShot(
        authedFetch,
        submissionId,
        shot.kind,
        previewBlob,
      );
      onResult(result);
      setPreviewUrl(null);
      setPreviewBlob(null);
    } catch (e) {
      const msg =
        e instanceof ApiError
          ? `Upload failed (${e.status}). Try again.`
          : e instanceof Error
            ? e.message
            : String(e);
      onError(msg);
    }
  };

  const retake = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setPreviewBlob(null);
  };

  // -- Render ---------------------------------------------------------

  const showVideo = Boolean(stream && !cameraError && !previewUrl);
  const showPreview = Boolean(previewUrl);
  const showEnablePrompt = !stream && !cameraError && !previewUrl;
  const showError = Boolean(cameraError && !previewUrl);

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
      {/* Viewfinder */}
      <div
        style={{
          position: "relative",
          aspectRatio: videoAspect ?? 4 / 3,
          background: "#000",
          borderRadius: 6,
          overflow: "hidden",
          border: "1px solid var(--line-2)",
        }}
      >
        {showPreview && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl ?? undefined}
            alt="Captured shot preview"
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "contain",
            }}
          />
        )}
        {showVideo && (
          <video
            ref={setVideoEl}
            autoPlay
            playsInline
            muted
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              if (v.videoWidth && v.videoHeight) {
                setVideoAspect(v.videoWidth / v.videoHeight);
              }
            }}
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "contain",
              display: "block",
            }}
          />
        )}
        {showEnablePrompt && (
          <CenteredOverlay>
            Tap{" "}
            <b style={{ color: "var(--lime)" }}>Enable camera</b> below to
            start the live feed.
          </CenteredOverlay>
        )}
        {showError && <CenteredOverlay tone="error">{cameraError}</CenteredOverlay>}
        {/* Cosmetic overlays — all pointer-events: none so they never
            steal taps from anything underneath. */}
        <PoseGuide pose={shot.pose} guideColor="rgba(190,242,100,0.6)" />
        {showVideo && detection.quad && (
          <DetectedQuad quad={detection.quad} framed={detection.framed} />
        )}
        {showVideo && detection.hint && (
          <CoachingBanner hint={detection.hint} framed={detection.framed} />
        )}
      </div>

      {/* Action row — buttons live OUTSIDE the viewfinder so no overlay
          can ever intercept taps. */}
      {state.phase === "uploading" ? (
        <UploadingRow />
      ) : showPreview ? (
        <PreviewActions onAccept={acceptPreview} onRetake={retake} />
      ) : showEnablePrompt || showError ? (
        <EnableCameraRow
          label={cameraError ? "Retry camera" : "Enable camera"}
          onClick={cameraError ? restartCamera : requestCamera}
          onFile={captureFromFile}
        />
      ) : (
        <LiveActions
          autoMode={autoMode}
          onToggleAuto={() => setAutoMode((v) => !v)}
          onCapture={captureFromVideo}
          onFile={captureFromFile}
          onRestart={restartCamera}
        />
      )}

      {state.phase === "failed" && <FailedReadout reasons={state.reasons} />}
      {state.phase === "passed" && <PassedReadout shot={state.shot} />}
    </div>
  );
}

// --------------------------------------------------------------------
// Subcomponents
// --------------------------------------------------------------------

function CenteredOverlay({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone?: "error";
}) {
  return (
    <div
      style={{
        position: "absolute",
        inset: 0,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        textAlign: "center",
        pointerEvents: "none",
        color: tone === "error" ? "var(--rose)" : "var(--ink-2)",
        fontSize: 13,
        lineHeight: 1.5,
      }}
    >
      <span style={{ maxWidth: 320 }}>{children}</span>
    </div>
  );
}

function EnableCameraRow({
  label,
  onClick,
  onFile,
}: {
  label: string;
  onClick: () => void;
  onFile: (f: File) => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 10,
        justifyContent: "center",
      }}
    >
      <button
        type="button"
        onClick={onClick}
        className="pg-btn pg-btn-primary"
        style={{
          padding: "16px 32px",
          fontSize: 16,
          fontWeight: 700,
          touchAction: "manipulation",
        }}
      >
        {label}
      </button>
      <FilePickerLabel onFile={onFile} />
    </div>
  );
}

function LiveActions({
  autoMode,
  onToggleAuto,
  onCapture,
  onFile,
  onRestart,
}: {
  autoMode: boolean;
  onToggleAuto: () => void;
  onCapture: () => void;
  onFile: (f: File) => void;
  onRestart: () => void;
}) {
  return (
    <div
      style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}
    >
      <button
        type="button"
        onClick={onCapture}
        className="pg-btn pg-btn-primary"
        style={{ padding: "12px 20px", fontSize: 14, touchAction: "manipulation" }}
      >
        Take photo
      </button>
      <button
        type="button"
        onClick={onToggleAuto}
        className={`pg-btn ${autoMode ? "pg-btn-primary" : "pg-btn-ghost"}`}
        aria-pressed={autoMode}
        style={{ touchAction: "manipulation" }}
      >
        {autoMode ? "Auto: on" : "Auto: off"}
      </button>
      <FilePickerLabel onFile={onFile} />
      <button
        type="button"
        onClick={onRestart}
        className="pg-btn pg-btn-ghost"
        style={{ touchAction: "manipulation" }}
        title="Re-acquire the camera if the feed froze"
      >
        Restart
      </button>
    </div>
  );
}

function FilePickerLabel({ onFile }: { onFile: (f: File) => void }) {
  return (
    <label className="pg-btn pg-btn-ghost" style={{ cursor: "pointer" }}>
      Upload file
      <input
        type="file"
        accept="image/*"
        capture="environment"
        style={{ display: "none" }}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
    </label>
  );
}

function PreviewActions({
  onAccept,
  onRetake,
}: {
  onAccept: () => void;
  onRetake: () => void;
}) {
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
      <button
        type="button"
        onClick={onAccept}
        className="pg-btn pg-btn-primary"
        style={{ padding: "12px 20px", fontSize: 14, touchAction: "manipulation" }}
      >
        Use this
      </button>
      <button
        type="button"
        onClick={onRetake}
        className="pg-btn pg-btn-ghost"
        style={{ touchAction: "manipulation" }}
      >
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

function DetectedQuad({ quad, framed }: { quad: Quad; framed: boolean }) {
  const stroke = framed ? "rgba(190,242,100,0.95)" : "rgba(251,191,36,0.9)";
  const W = 320;
  const H = 240;
  const points = quad.map((p) => `${p.x / W},${p.y / H}`).join(" ");
  return (
    <svg
      aria-hidden
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
      }}
      viewBox="0 0 1 1"
      preserveAspectRatio="none"
    >
      <polygon
        points={points}
        fill={framed ? "rgba(190,242,100,0.08)" : "none"}
        stroke={stroke}
        strokeWidth={3}
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
        style={{ transition: "stroke 120ms ease" }}
      />
      {quad.map((p, i) => (
        <circle
          key={i}
          cx={p.x / W}
          cy={p.y / H}
          r={0.012}
          fill={stroke}
          vectorEffect="non-scaling-stroke"
        />
      ))}
    </svg>
  );
}

function CoachingBanner({
  hint,
  framed,
}: {
  hint: string;
  framed: boolean;
}) {
  const lower = hint.toLowerCase();
  let icon = "📷";
  if (lower.startsWith("move closer")) icon = "⬇️";
  else if (lower.startsWith("pull back")) icon = "⬆️";
  else if (lower.startsWith("pan left")) icon = "←";
  else if (lower.startsWith("pan right")) icon = "→";
  else if (lower.startsWith("pan up")) icon = "↑";
  else if (lower.startsWith("pan down")) icon = "↓";
  else if (lower.includes("framed") || lower.includes("locked")) icon = "✓";
  else if (lower.includes("hold steady")) icon = "✋";
  return (
    <div
      style={{
        position: "absolute",
        top: 12,
        left: "50%",
        transform: "translateX(-50%)",
        background: framed ? "rgba(40,80,20,0.92)" : "rgba(0,0,0,0.78)",
        color: framed ? "#d8fa6c" : "#fff",
        padding: "8px 14px",
        borderRadius: 12,
        fontSize: 13,
        fontWeight: 600,
        border: framed
          ? "2px solid rgba(190,242,100,0.85)"
          : "1px solid rgba(255,255,255,0.2)",
        boxShadow: framed
          ? "0 4px 16px rgba(190,242,100,0.25)"
          : "0 4px 16px rgba(0,0,0,0.5)",
        maxWidth: "88%",
        textAlign: "center",
        pointerEvents: "none",
        display: "flex",
        alignItems: "center",
        gap: 8,
        whiteSpace: "nowrap",
      }}
    >
      <span style={{ fontSize: 16, flexShrink: 0 }}>{icon}</span>
      <span>{hint}</span>
    </div>
  );
}
