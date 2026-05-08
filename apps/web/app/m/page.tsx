/**
 * `/m` — minimal mobile capture test page.
 *
 * Bare-bones isolation of the camera + upload flow so we can debug what
 * works and what doesn't on a phone, separately from the full wizard.
 * Creates a submission on load, opens the back camera, lets you snap and
 * upload a single front_full shot, and shows the server response.
 *
 * No quad detection, no overlays, no fullscreen, no auto-capture, no
 * device motion. Just: video stream → canvas → blob → upload → result.
 */

"use client";

import { useEffect, useRef, useState } from "react";

import {
  createSubmission,
  uploadShot,
  useAuthedFetch,
  type ShotOut,
} from "@/lib/submission";

type Step =
  | "init"
  | "creating_submission"
  | "submission_ready"
  | "requesting_camera"
  | "camera_ready"
  | "camera_denied"
  | "snapping"
  | "uploading"
  | "result"
  | "error";

export default function MobileTestPage() {
  const authedFetch = useAuthedFetch();
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [streamState, setStreamState] = useState<MediaStream | null>(null);
  const [submissionId, setSubmissionId] = useState<string | null>(null);
  const [step, setStep] = useState<Step>("init");
  const [logs, setLogs] = useState<string[]>([]);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewBlob, setPreviewBlob] = useState<Blob | null>(null);
  const [shotResult, setShotResult] = useState<ShotOut | null>(null);
  const [videoState, setVideoState] = useState<{
    width: number;
    height: number;
    readyState: number;
  } | null>(null);

  const log = (msg: string) =>
    setLogs((prev) => [...prev, `${new Date().toISOString().slice(11, 23)}  ${msg}`]);

  // Create a submission as soon as the page mounts.
  useEffect(() => {
    let cancelled = false;
    setStep("creating_submission");
    log("creating submission...");
    createSubmission(authedFetch)
      .then((sub) => {
        if (cancelled) return;
        setSubmissionId(sub.id);
        setStep("submission_ready");
        log(`submission ${sub.id.slice(0, 8)} ready`);
      })
      .catch((e) => {
        if (cancelled) return;
        setStep("error");
        setErrorMsg(e instanceof Error ? e.message : String(e));
        log(`submission error: ${e}`);
      });
    return () => {
      cancelled = true;
    };
  }, [authedFetch]);

  // Camera request is triggered by an explicit button tap so iOS Safari
  // gets a clean user-gesture context (it silently no-ops on auto-call in
  // recent iOS versions). The handler is awaitable from a button onClick.
  const requestCamera = async () => {
    if (step !== "submission_ready" && step !== "camera_denied") return;
    setStep("requesting_camera");
    log("requesting camera...");
    log(
      `mediaDevices: ${typeof navigator !== "undefined" && !!navigator.mediaDevices}, getUserMedia: ${typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia}`,
    );
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" } },
        audio: false,
      });
      setStreamState(s);
      setStep("camera_ready");
      const t = s.getVideoTracks()[0];
      const settings = t?.getSettings();
      log(
        `camera ready, track=${t?.label ?? "?"}, ${settings?.width ?? "?"}x${settings?.height ?? "?"}`,
      );
    } catch (e) {
      setStep("camera_denied");
      setErrorMsg(e instanceof Error ? `${e.name}: ${e.message}` : String(e));
      log(`camera error: ${e instanceof Error ? `${e.name}: ${e.message}` : e}`);
    }
  };

  // Stop tracks when the page unmounts.
  useEffect(() => {
    return () => {
      streamState?.getTracks().forEach((t) => t.stop());
    };
  }, [streamState]);

  // Attach stream → video.
  useEffect(() => {
    if (!streamState || !videoRef.current) return;
    const v = videoRef.current;
    v.srcObject = streamState;
    log("srcObject attached, calling play()");
    const p = v.play();
    if (p && typeof p.catch === "function") {
      p.then(() => log("play() resolved")).catch((e) =>
        log(`play() rejected: ${e}`),
      );
    }
    const onMeta = () => {
      log(`loadedmetadata: ${v.videoWidth}x${v.videoHeight}`);
      setVideoState({
        width: v.videoWidth,
        height: v.videoHeight,
        readyState: v.readyState,
      });
    };
    const onCanPlay = () => {
      log(`canplay (readyState=${v.readyState})`);
      setVideoState({
        width: v.videoWidth,
        height: v.videoHeight,
        readyState: v.readyState,
      });
    };
    v.addEventListener("loadedmetadata", onMeta);
    v.addEventListener("canplay", onCanPlay);
    return () => {
      v.removeEventListener("loadedmetadata", onMeta);
      v.removeEventListener("canplay", onCanPlay);
    };
  }, [streamState]);

  const snap = async () => {
    const v = videoRef.current;
    if (!v) {
      log("snap: no video ref");
      return;
    }
    if (v.readyState < 2) {
      log(`snap: readyState too low (${v.readyState})`);
      return;
    }
    setStep("snapping");
    const canvas = document.createElement("canvas");
    canvas.width = v.videoWidth;
    canvas.height = v.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      log("snap: no 2d ctx");
      setStep("camera_ready");
      return;
    }
    ctx.drawImage(v, 0, 0);
    const blob = await new Promise<Blob | null>((res) =>
      canvas.toBlob((b) => res(b), "image/jpeg", 0.92),
    );
    if (!blob) {
      log("snap: toBlob returned null");
      setStep("camera_ready");
      return;
    }
    setPreviewBlob(blob);
    setPreviewUrl(URL.createObjectURL(blob));
    log(`snapped ${canvas.width}x${canvas.height}, ${(blob.size / 1024).toFixed(0)} KiB`);
    setStep("camera_ready");
  };

  const upload = async () => {
    if (!previewBlob || !submissionId) return;
    setStep("uploading");
    log("uploading...");
    try {
      const r = await uploadShot(
        authedFetch,
        submissionId,
        "front_full",
        previewBlob,
      );
      setShotResult(r);
      setStep("result");
      log(`upload done: passed=${r.quality_passed}`);
      const reasons = (r.quality_metadata as { reasons?: string[] } | undefined)
        ?.reasons;
      if (reasons?.length) log(`reasons: ${reasons.join(" | ")}`);
    } catch (e) {
      setStep("error");
      setErrorMsg(e instanceof Error ? e.message : String(e));
      log(`upload error: ${e}`);
    }
  };

  const retake = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(null);
    setPreviewBlob(null);
    setShotResult(null);
  };

  return (
    <main
      style={{
        fontFamily: "system-ui, -apple-system, sans-serif",
        background: "#0a0a0a",
        color: "#fff",
        minHeight: "100dvh",
        padding: "16px",
        boxSizing: "border-box",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <h1 style={{ margin: 0, fontSize: 20 }}>mobile camera test</h1>

      <div
        style={{
          fontFamily: "ui-monospace, monospace",
          fontSize: 11,
          background: "#181818",
          border: "1px solid #2a2a2a",
          padding: 8,
          borderRadius: 6,
          display: "grid",
          gridTemplateColumns: "auto 1fr",
          gap: "2px 8px",
        }}
      >
        <span style={{ color: "#888" }}>step</span>
        <span>{step}</span>
        <span style={{ color: "#888" }}>submission</span>
        <span>{submissionId ? submissionId.slice(0, 8) + "…" : "—"}</span>
        <span style={{ color: "#888" }}>stream</span>
        <span>{streamState ? "yes" : "no"}</span>
        <span style={{ color: "#888" }}>video</span>
        <span>
          {videoState
            ? `${videoState.width}x${videoState.height} rs=${videoState.readyState}`
            : "—"}
        </span>
        {errorMsg && (
          <>
            <span style={{ color: "#f88" }}>error</span>
            <span style={{ color: "#f88" }}>{errorMsg}</span>
          </>
        )}
      </div>

      <div
        style={{
          position: "relative",
          width: "100%",
          aspectRatio: "3 / 4",
          background: "#000",
          border: "1px solid #2a2a2a",
          borderRadius: 8,
          overflow: "hidden",
        }}
      >
        {previewUrl ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={previewUrl}
            alt="snapped preview"
            style={{
              position: "absolute",
              inset: 0,
              width: "100%",
              height: "100%",
              objectFit: "contain",
            }}
          />
        ) : (
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
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
      </div>

      <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
        {(step === "submission_ready" || step === "camera_denied") && (
          <button type="button" onClick={requestCamera} style={btnPrimary}>
            {step === "camera_denied" ? "Retry camera" : "Enable camera"}
          </button>
        )}
        {!previewUrl && step === "camera_ready" && (
          <button type="button" onClick={snap} style={btnPrimary}>
            Snap
          </button>
        )}
        {previewUrl && step !== "uploading" && (
          <>
            <button type="button" onClick={upload} style={btnPrimary}>
              Upload
            </button>
            <button type="button" onClick={retake} style={btnGhost}>
              Retake
            </button>
          </>
        )}
        <label style={{ ...btnGhost, cursor: "pointer" }}>
          File…
          <input
            type="file"
            accept="image/*"
            capture="environment"
            style={{ display: "none" }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              setPreviewBlob(f);
              setPreviewUrl(URL.createObjectURL(f));
              log(`file picked: ${f.name} (${(f.size / 1024).toFixed(0)} KiB)`);
            }}
          />
        </label>
      </div>

      {shotResult && (
        <div
          style={{
            background: shotResult.quality_passed
              ? "#10301055"
              : "#5e1d1d55",
            border: shotResult.quality_passed
              ? "1px solid #4ade80"
              : "1px solid #f87171",
            borderRadius: 6,
            padding: 10,
            fontFamily: "ui-monospace, monospace",
            fontSize: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {shotResult.quality_passed ? "✓ PASSED" : "✗ FAILED"} ·{" "}
          {shotResult.width_px}x{shotResult.height_px}
          {"\n"}
          {JSON.stringify(shotResult.quality_metadata, null, 2)}
        </div>
      )}

      <div
        style={{
          marginTop: "auto",
          background: "#0f0f0f",
          border: "1px solid #2a2a2a",
          borderRadius: 6,
          padding: 8,
          fontFamily: "ui-monospace, monospace",
          fontSize: 10,
          color: "#bbb",
          maxHeight: 220,
          overflowY: "auto",
        }}
      >
        {logs.length === 0 ? (
          <span style={{ color: "#666" }}>(no events yet)</span>
        ) : (
          logs.map((l, i) => (
            <div key={i} style={{ whiteSpace: "pre-wrap" }}>
              {l}
            </div>
          ))
        )}
      </div>
    </main>
  );
}

const btnPrimary: React.CSSProperties = {
  appearance: "none",
  background: "#bef264",
  color: "#0a0a0a",
  border: "1px solid #84cc16",
  borderRadius: 6,
  padding: "10px 18px",
  fontSize: 14,
  fontWeight: 600,
  cursor: "pointer",
};

const btnGhost: React.CSSProperties = {
  appearance: "none",
  background: "#181818",
  color: "#fff",
  border: "1px solid #2a2a2a",
  borderRadius: 6,
  padding: "10px 18px",
  fontSize: 14,
  cursor: "pointer",
};
