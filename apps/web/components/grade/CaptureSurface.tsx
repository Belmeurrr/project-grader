/**
 * Capture surface — wizard's per-shot viewfinder + actions.
 *
 * MVP. Strip-back rebuild: smallest thing that captures a card and
 * uploads it. Auto-detection / OpenCV lock-on / haptics / countdown
 * rings are all temporarily gone — they were creating compounding bugs
 * (six useEffects fighting over `stream`, rAF detector that didn't
 * actually detect frozen frames, OpenCV blocking the main thread).
 *
 * Rebuild plan: get this rock-solid, commit, then re-add features one at
 * a time behind clearly delimited state, each verified before the next:
 *   1. Live polygon overlay (cosmetic, OpenCV behind feature flag)
 *   2. Auto-capture w/ lock detection
 *   3. Haptic / audio feedback
 *
 * State machine (one truth, easy to read):
 *   idle           — no stream yet; show "Enable camera" button
 *   live           — stream attached, video playing; show Take photo
 *   preview        — captured blob staged; show Use this / Retake
 *   uploading      — mid-upload to API; show spinner
 *   passed/failed  — server verdict; show readout (parent state)
 *   cameraError    — getUserMedia rejected; show error + retry button
 *
 * Recovery: when the page transitions from hidden→visible after >500ms
 * (i.e. a real app-switch, not a momentary permission overlay), we kill
 * the stream and reset to "idle". User taps Enable camera once to
 * revive — guaranteed-fresh user gesture, never a stale frozen frame.
 */

"use client";

import { useCallback, useEffect, useState } from "react";

import {
  ApiError,
  type AuthedFetch,
  type ShotKind,
  type ShotOut,
  uploadShot,
} from "@/lib/submission";
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

const SETTLE_MS = 220;

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
  const [videoAspect, setVideoAspect] = useState<number | null>(null);

  // Reset preview when the wizard switches to a different shot.
  useEffect(() => {
    setPreviewUrl(null);
    setPreviewBlob(null);
  }, [shot.kind]);

  // Stop tracks on unmount (page exit).
  useEffect(() => {
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [stream]);

  // Single attach effect — when both the video element and the stream
  // are available, wire them together and start playback.
  useEffect(() => {
    if (!videoEl || !stream) return;
    if (videoEl.srcObject !== stream) videoEl.srcObject = stream;
    const p = videoEl.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  }, [videoEl, stream]);

  // Pull aspect ratio from the track settings — more reliable on iOS
  // than waiting for the loadedmetadata DOM event.
  useEffect(() => {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    const settings = track?.getSettings();
    if (settings?.width && settings?.height) {
      setVideoAspect(settings.width / settings.height);
    }
  }, [stream]);

  // [OpenCV loading + auto-crop removed for now — even loading-only
  // (no live tick) blocks the main thread during the WASM compile
  // long enough to queue user taps. Will re-add via a Web Worker so
  // the compile can't choke the page. Manual capture still works;
  // server-side detector runs against whatever frame we ship.]

  // Reliable backgrounding recovery. iOS keeps tracks "live" even
  // when the actual frame data has stopped flowing, so we don't try
  // to introspect — we just kill on any non-trivial hidden interval
  // and let the user tap Enable camera to get a fresh stream.
  useEffect(() => {
    if (!stream) return;
    let hiddenAt: number | null = null;
    const handleVis = () => {
      if (document.visibilityState === "hidden") {
        hiddenAt = Date.now();
      } else if (document.visibilityState === "visible") {
        const dur = hiddenAt ? Date.now() - hiddenAt : 0;
        hiddenAt = null;
        if (dur > 500) {
          stream.getTracks().forEach((t) => t.stop());
          setStream(null);
          setVideoAspect(null);
        }
      }
    };
    document.addEventListener("visibilitychange", handleVis);
    return () => document.removeEventListener("visibilitychange", handleVis);
  }, [stream]);

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
    setCameraError(null);
    await requestCamera();
  };

  const captureFromVideo = useCallback(async () => {
    if (!videoEl || videoEl.readyState < 2) return;
    // Brief settle so a tap-induced jiggle doesn't end up baked in.
    await new Promise((r) => setTimeout(r, SETTLE_MS));
    if (!videoEl || videoEl.readyState < 2) return;

    const canvas = document.createElement("canvas");
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      onError("Could not get 2D context.");
      return;
    }
    ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
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
        {showPreview && previewUrl && (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
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
        {showError && (
          <CenteredOverlay tone="error">{cameraError}</CenteredOverlay>
        )}
        {/* Cosmetic framing brackets only — pointer-events: none. */}
        <PoseGuide pose={shot.pose} guideColor="rgba(190,242,100,0.6)" />
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
  onCapture,
  onFile,
  onRestart,
}: {
  onCapture: () => void;
  onFile: (f: File) => void;
  onRestart: () => void;
}) {
  return (
    <div
      style={{
        display: "flex",
        flexWrap: "wrap",
        alignItems: "center",
        gap: 8,
      }}
    >
      <button
        type="button"
        onClick={onCapture}
        className="pg-btn pg-btn-primary"
        style={{
          padding: "12px 20px",
          fontSize: 14,
          touchAction: "manipulation",
        }}
      >
        Take photo
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
        style={{
          padding: "12px 20px",
          fontSize: 14,
          touchAction: "manipulation",
        }}
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
