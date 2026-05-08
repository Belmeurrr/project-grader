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

import { useCallback, useEffect, useRef, useState } from "react";

import {
  ApiError,
  type AuthedFetch,
  type ShotKind,
  type ShotOut,
  uploadShot,
} from "@/lib/submission";
import {
  createCardDetectWorker,
  type CardDetectWorker,
} from "@/lib/cardDetectWorker";

interface Point {
  x: number;
  y: number;
}
type Quad = [Point, Point, Point, Point];

type HintIcon =
  | "search"
  | "down"
  | "up"
  | "left"
  | "right"
  | "tilt-up"
  | "tilt-down"
  | "ok";
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

// Tunable timing for capture + auto-fire.
// SETTLE_MS: brief pause after a Take-photo tap so the hand-jiggle from
//   the tap doesn't end up baked into the snapped frame. iPhone autofocus
//   typically re-locks in under 350ms after the user stops moving.
const SETTLE_MS = 350;
// AUTO_TICKS_TO_FIRE: how many consecutive 250ms-tick framings the
//   detector needs before auto-mode fires the shutter. 4 ticks ≈ 1s,
//   long enough to feel deliberate (not a misfire) but short enough to
//   feel responsive.
const AUTO_TICKS_TO_FIRE = 4;

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
  // Live polygon overlay: latest quad reported by the detector worker.
  // Updated as fast as the worker can respond (~30-80ms/frame on phone),
  // capped at ~4×/sec by the tick interval. `null` means no card found
  // this round.
  const [liveQuad, setLiveQuad] = useState<Quad | null>(null);
  // Smoothed visibility state: keeps the last quad on-screen briefly
  // after the detector drops it, so brief misses don't make the polygon
  // flicker. Drives the actual SVG render.
  const [displayedQuad, setDisplayedQuad] = useState<Quad | null>(null);
  // Coaching hint — one short instruction that tells the user how to
  // adjust framing. Computed from each detection result; drives a
  // banner at the top of the viewfinder.
  const [hint, setHint] = useState<{
    icon: HintIcon;
    text: string;
    framed: boolean;
  }>({ icon: "search", text: "Loading detector…", framed: false });
  // Worker readiness — flipped to true when OpenCV's WASM has compiled
  // inside the worker thread. Until then, hints stay on
  // "Loading detector…" instead of confusingly saying "Hold a card in
  // front of the camera" when the detector simply isn't online yet.
  const [detectorReady, setDetectorReady] = useState(false);
  // Auto-capture state. When `autoMode` is on and the live detector
  // reports `framed=true` for `AUTO_TICKS_TO_FIRE` consecutive ticks,
  // the shutter fires automatically. `autoProgress` (0..1) drives a
  // visual countdown so the user sees the fire approaching and can
  // back off / re-frame to abort.
  const [autoMode, setAutoMode] = useState(false);
  const [autoProgress, setAutoProgress] = useState(0);

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

  // OpenCV-backed card detector lives in a Web Worker so its 10 MB
  // WASM compile happens off-main-thread. We only need it for one
  // job: figure out where the card is in the snapped frame so we can
  // crop to it before upload. The worker spins up the moment the
  // camera comes online; by the time the user taps Take photo it's
  // ready (or close to it). If it's not, we ship the full frame —
  // detection is best-effort, never load-bearing.
  const detectorRef = useRef<CardDetectWorker | null>(null);
  useEffect(() => {
    if (!stream) return;
    if (detectorRef.current) {
      setDetectorReady(true); // already initialized on prior mount
      return;
    }
    const w = createCardDetectWorker();
    detectorRef.current = w;
    void w.ready
      .then(() => setDetectorReady(true))
      .catch(() => setDetectorReady(false));
    return () => {
      // Don't tear down on stream change — re-warming the worker on
      // every Enable-camera tap throws away the WASM compile cost we
      // just paid. Keep it alive until full unmount.
    };
  }, [stream]);
  useEffect(() => {
    return () => {
      detectorRef.current?.destroy();
      detectorRef.current = null;
    };
  }, []);

  // Live detection tick: every ~250ms (or as fast as the worker can
  // respond, whichever is slower) send a downsampled frame to the
  // worker and store the resulting quad for the polygon overlay. The
  // `inFlight` flag provides natural backpressure — we never queue
  // requests faster than the worker can answer them, so a slow phone
  // just runs the overlay at a lower frame rate without piling up.
  useEffect(() => {
    if (!stream || !videoEl || previewUrl || cameraError) return;
    const detector = detectorRef.current;
    if (!detector) return;
    const W = 320;
    const H = 240;
    const probe = document.createElement("canvas");
    probe.width = W;
    probe.height = H;
    const pctx = probe.getContext("2d", { willReadFrequently: true });
    if (!pctx) return;
    let cancelled = false;
    let inFlight = false;
    let framedTicks = 0;
    const tick = async () => {
      if (cancelled || inFlight) return;
      if (videoEl.readyState < 2) return;
      inFlight = true;
      try {
        pctx.drawImage(videoEl, 0, 0, W, H);
        const quad = await detector.detect(pctx.getImageData(0, 0, W, H));
        if (cancelled) return;
        setLiveQuad(quad);
        const h = computeHint(quad, W, H, detectorReady);
        setHint(h);
        // Auto-capture countdown. Increments while framed; resets the
        // moment the user moves out of frame so the visual countdown
        // reflects current state (lets them abort by jiggling).
        if (autoMode && h.framed) {
          framedTicks += 1;
        } else {
          framedTicks = 0;
        }
        setAutoProgress(Math.min(1, framedTicks / AUTO_TICKS_TO_FIRE));
        if (autoMode && framedTicks >= AUTO_TICKS_TO_FIRE) {
          framedTicks = 0;
          setAutoProgress(0);
          // Latch on a ref so the closure here uses the freshest
          // captureFromVideo (defined further down).
          if (captureRef.current) void captureRef.current();
        }
      } catch {
        if (!cancelled) {
          setLiveQuad(null);
          setHint({
            icon: "search",
            text: "Looking for card…",
            framed: false,
          });
          framedTicks = 0;
          setAutoProgress(0);
        }
      } finally {
        inFlight = false;
      }
    };
    const handle = window.setInterval(tick, 250);
    return () => {
      cancelled = true;
      window.clearInterval(handle);
      setLiveQuad(null);
      setAutoProgress(0);
    };
  }, [stream, videoEl, previewUrl, cameraError, autoMode, detectorReady]);

  // Hold a stable reference to the latest captureFromVideo so the live
  // tick's auto-fire path always invokes the up-to-date closure.
  const captureRef = useRef<() => Promise<void>>(async () => {});

  // Smoothing: when the detector finds a quad, show it immediately;
  // when it drops, hold the last one on screen for 350ms before
  // clearing. Eliminates flicker on brief detection misses without
  // making the polygon "stick" if the user moves the camera away.
  useEffect(() => {
    if (liveQuad) {
      setDisplayedQuad(liveQuad);
      return;
    }
    const t = window.setTimeout(() => setDisplayedQuad(null), 350);
    return () => window.clearTimeout(t);
  }, [liveQuad]);

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
    const fullW = videoEl.videoWidth;
    const fullH = videoEl.videoHeight;

    // Auto-crop step 1: ask the detector worker if it can find a card
    // in a downsampled probe of the snapped frame. Worker may not be
    // ready yet, in which case `detect()` resolves to null fast and we
    // ship the full frame.
    let sx = 0;
    let sy = 0;
    let sw = fullW;
    let sh = fullH;
    const detector = detectorRef.current;
    if (detector) {
      const probeW = 320;
      const probeH = 240;
      const probe = document.createElement("canvas");
      probe.width = probeW;
      probe.height = probeH;
      const pctx = probe.getContext("2d", { willReadFrequently: true });
      if (pctx) {
        pctx.drawImage(videoEl, 0, 0, probeW, probeH);
        try {
          const quad = await detector.detect(
            pctx.getImageData(0, 0, probeW, probeH),
          );
          if (quad) {
            // Axis-aligned bounding box of the quad + 14% margin on
            // each side, clamped to the frame.
            const xs = quad.map((p) => p.x);
            const ys = quad.map((p) => p.y);
            const x0 = Math.min(...xs) / probeW;
            const x1 = Math.max(...xs) / probeW;
            const y0 = Math.min(...ys) / probeH;
            const y1 = Math.max(...ys) / probeH;
            // Tight crop — keeps just enough background for the server's
            // detector to find the card's edges, but not so much that
            // the server's tighter bbox ends up below the fill threshold
            // (we used to ship 14% margin; the server kept measuring
            // ~34% fill on that, missing the 40% gate). 6% gives the
            // server detector enough lead-in without diluting the fill.
            const margin = 0.06;
            const nx = Math.max(0, x0 - margin);
            const ny = Math.max(0, y0 - margin);
            const nw = Math.min(1 - nx, x1 - x0 + 2 * margin);
            const nh = Math.min(1 - ny, y1 - y0 + 2 * margin);
            const px = Math.round(nx * fullW);
            const py = Math.round(ny * fullH);
            const pw = Math.round(nw * fullW);
            const ph = Math.round(nh * fullH);
            // Sanity-check: don't ship something tiny. 150 px on either
            // dimension is the floor — the worker's geometry validation
            // already rejects bad quads, so anything reaching here is a
            // real card. The floor exists only to bail when the source
            // video itself was unusually small (legacy devices, weird
            // constraints) and the crop would be unusably blurry on
            // upscale.
            if (pw >= 150 && ph >= 150) {
              sx = px;
              sy = py;
              sw = pw;
              sh = ph;
            }
          }
        } catch {
          // Detector errored; ship full frame.
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
        {/* Live polygon — drawn on top of the brackets so the user sees
            the detector locking onto their actual card. Pointer-events:
            none, never blocks taps. */}
        {showVideo && displayedQuad && <LivePolygon quad={displayedQuad} />}
        {/* Coaching banner — pinned to the top of the viewfinder, tells
            the user exactly what to adjust. Bottom progress bar fills
            during the auto-mode countdown. */}
        {showVideo && (
          <HintBanner
            hint={hint}
            autoMode={autoMode}
            autoProgress={autoProgress}
          />
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
          onCapture={captureFromVideo}
          onFile={captureFromFile}
          onRestart={restartCamera}
          autoMode={autoMode}
          onToggleAuto={() => setAutoMode((v) => !v)}
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

// Inspect the detector's quad and pick the single most actionable hint.
// Order matters — surface the biggest framing problem first, so the user
// has one clear next step rather than a pile of complaints.
//
// Thresholds intentionally MATCH the server-side quality gate so the
// "Looks great" green doesn't lie: if this returns framed=true on the
// pre-crop image, the card+margin auto-crop will pass the server's
// fill/centering/tilt checks too.
function computeHint(
  quad: Quad | null,
  W: number,
  H: number,
  detectorReady: boolean,
): { icon: HintIcon; text: string; framed: boolean } {
  if (!detectorReady) {
    return { icon: "search", text: "Loading detector…", framed: false };
  }
  if (!quad) {
    return {
      icon: "search",
      text: "Hold a card in front of the camera",
      framed: false,
    };
  }
  const xs = quad.map((p) => p.x);
  const ys = quad.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const cardW = (x1 - x0) / W;
  const cardH = (y1 - y0) / H;
  const fill = cardW * cardH;
  const cx = (x0 + x1) / (2 * W);
  const cy = (y0 + y1) / (2 * H);

  // Size first — too far is the most common framing problem on a phone.
  // 0.25 matches the server's relaxed min_fill_ratio. Aligns "Looks
  // great" with what the server will actually accept.
  if (fill < 0.25) return { icon: "down", text: "Move closer", framed: false };
  if (fill > 0.90)
    return { icon: "up", text: "Pull back a little", framed: false };

  // Tilt detection. A perfectly-square-on card has equal opposite-side
  // lengths; perspective skew shows up as a side-length mismatch. If
  // the worst opposite-side ratio dips below 0.85, the server's
  // perspective_deg measurement is likely to land near 20° (the gate).
  const top = Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
  const bottom = Math.hypot(quad[2].x - quad[3].x, quad[2].y - quad[3].y);
  const left = Math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y);
  const right = Math.hypot(quad[2].x - quad[1].x, quad[2].y - quad[1].y);
  const lrRatio = Math.min(top, bottom) / Math.max(top, bottom);
  const tbRatio = Math.min(left, right) / Math.max(left, right);
  const skewy = Math.min(lrRatio, tbRatio) < 0.82;

  // Then centering. 18% off-center either way is the cutoff used by the
  // server-side framing checks, so the hint matches what'll fail.
  const dx = cx - 0.5;
  const dy = cy - 0.5;
  if (Math.abs(dx) > 0.18 || Math.abs(dy) > 0.18) {
    if (Math.abs(dx) > Math.abs(dy)) {
      return {
        icon: dx > 0 ? "right" : "left",
        text: dx > 0 ? "Pan right" : "Pan left",
        framed: false,
      };
    }
    return {
      icon: dy > 0 ? "tilt-down" : "tilt-up",
      text: dy > 0 ? "Pan down" : "Pan up",
      framed: false,
    };
  }

  if (skewy) {
    return {
      icon: "tilt-up",
      text: "Hold the phone level with the card",
      framed: false,
    };
  }

  return { icon: "ok", text: "Looks great — hold steady", framed: true };
}

function HintBanner({
  hint,
  autoMode,
  autoProgress,
}: {
  hint: { icon: HintIcon; text: string; framed: boolean };
  autoMode: boolean;
  autoProgress: number;
}) {
  // When auto-mode is on AND framing is good, the bottom of the banner
  // animates a progress bar that fills as the auto-fire countdown
  // approaches. Gives the user a visible "snap is coming" cue with a
  // built-in window to abort by jiggling the phone.
  const showProgress = autoMode && hint.framed && autoProgress > 0;
  return (
    <div
      style={{
        position: "absolute",
        top: 10,
        left: 10,
        right: 10,
        background: hint.framed
          ? "rgba(20,60,5,0.94)"
          : "rgba(0,0,0,0.82)",
        color: hint.framed ? "#e9ff9b" : "#fff",
        padding: "11px 14px",
        borderRadius: 12,
        fontSize: 16,
        fontWeight: 700,
        letterSpacing: "-0.01em",
        border: hint.framed
          ? "2px solid rgba(190,242,100,0.9)"
          : "1px solid rgba(255,255,255,0.18)",
        boxShadow: hint.framed
          ? "0 6px 20px rgba(190,242,100,0.3)"
          : "0 4px 14px rgba(0,0,0,0.5)",
        textAlign: "center",
        pointerEvents: "none",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        gap: 12,
        lineHeight: 1.2,
        overflow: "hidden",
        transition:
          "background 200ms ease, color 200ms ease, border-color 200ms ease",
      }}
    >
      <HintGlyph icon={hint.icon} />
      <span style={{ flex: "0 1 auto", minWidth: 0 }}>{hint.text}</span>
      {showProgress && (
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            bottom: 0,
            height: 4,
            background: "rgba(190,242,100,0.25)",
          }}
        >
          <div
            style={{
              width: `${Math.max(0, Math.min(1, autoProgress)) * 100}%`,
              height: "100%",
              background: "rgba(190,242,100,0.95)",
              transition: "width 220ms linear",
            }}
          />
        </div>
      )}
    </div>
  );
}

function HintGlyph({ icon }: { icon: HintIcon }) {
  // SVG arrows scale crisp at any size and don't depend on emoji
  // rendering (iOS sometimes draws emoji slightly offset mid-text).
  if (icon === "search") {
    return (
      <svg viewBox="0 0 24 24" style={{ width: 26, height: 26 }} aria-hidden>
        <circle
          cx="11"
          cy="11"
          r="6"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
        />
        <path
          d="M16 16 L21 21"
          stroke="currentColor"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    );
  }
  if (icon === "ok") {
    return (
      <svg viewBox="0 0 24 24" style={{ width: 26, height: 26 }} aria-hidden>
        <path
          d="M5 12 L10 17 L20 6"
          fill="none"
          stroke="currentColor"
          strokeWidth="3"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      </svg>
    );
  }
  // Direction arrows share a path; rotate the SVG to point.
  const rot: Record<HintIcon, number> = {
    search: 0,
    ok: 0,
    up: 0,
    "tilt-up": 0,
    right: 90,
    down: 180,
    "tilt-down": 180,
    left: 270,
  };
  return (
    <svg
      viewBox="0 0 24 24"
      style={{
        width: 26,
        height: 26,
        transform: `rotate(${rot[icon]}deg)`,
      }}
      aria-hidden
    >
      <path
        d="M12 3 L4 11 L9 11 L9 21 L15 21 L15 11 L20 11 Z"
        fill="currentColor"
      />
    </svg>
  );
}

function LivePolygon({ quad }: { quad: Quad }) {
  // Design language drawn from iOS Notes / Adobe Scan / Google Lens:
  //   - Thin bright stroke that doesn't compete with the card art
  //   - Faint inner fill (8%) so the card region is unmistakable
  //   - Bold corner dots with a dark halo so they read on either bright
  //     or dark cards (matches the "Adobe Scan precision corners" cue)
  //   - 200ms fade-in via CSS so the polygon doesn't pop in jarringly
  //   - No marching-ants / scan-line animation — that signaled
  //     "still searching" in earlier iterations and added noise once
  //     the detector locked
  const W = 320;
  const H = 240;
  const points = quad.map((p) => `${p.x / W},${p.y / H}`).join(" ");
  return (
    <>
      <style>{`
        @keyframes pg-poly-in {
          0%   { opacity: 0; transform: scale(1.02); }
          100% { opacity: 1; transform: scale(1); }
        }
      `}</style>
      <svg
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          pointerEvents: "none",
          animation: "pg-poly-in 180ms ease-out",
          transformOrigin: "center",
        }}
        viewBox="0 0 1 1"
        preserveAspectRatio="none"
      >
        <polygon
          points={points}
          fill="rgba(190,242,100,0.08)"
          stroke="rgba(190,242,100,0.95)"
          strokeWidth="2"
          strokeLinejoin="round"
          vectorEffect="non-scaling-stroke"
        />
        {quad.map((p, i) => (
          <g key={i}>
            <circle
              cx={p.x / W}
              cy={p.y / H}
              r="0.022"
              fill="rgba(0,0,0,0.45)"
              vectorEffect="non-scaling-stroke"
            />
            <circle
              cx={p.x / W}
              cy={p.y / H}
              r="0.014"
              fill="rgb(190,242,100)"
              vectorEffect="non-scaling-stroke"
            />
          </g>
        ))}
      </svg>
    </>
  );
}

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
  autoMode,
  onToggleAuto,
}: {
  onCapture: () => void;
  onFile: (f: File) => void;
  onRestart: () => void;
  autoMode: boolean;
  onToggleAuto: () => void;
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
      <button
        type="button"
        onClick={onToggleAuto}
        className={`pg-btn ${autoMode ? "pg-btn-primary" : "pg-btn-ghost"}`}
        aria-pressed={autoMode}
        style={{ touchAction: "manipulation" }}
        title={
          autoMode
            ? "Auto-fire is on — shutter snaps when framing is good for ~1s"
            : "Auto-fire is off — tap Take photo to capture"
        }
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
