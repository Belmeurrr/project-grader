"use client";

/**
 * <AuthenticityDetectors> — collapsible per-detector breakdown.
 *
 * Each row expands to show forensic metadata pulled from the API's
 * `metadata: Record<string, unknown>` field. The metadata key ↔ display
 * mapping lives in `DETECTOR_META` so adding a new detector is a single
 * dictionary entry; the page itself doesn't change.
 */

import { useState } from "react";

import type { AuthenticityVerdict, DetectorScore } from "@/lib/cert";

const VERDICT_TONE: Record<AuthenticityVerdict, { tone: string; short: string }> = {
  authentic: { tone: "var(--lime)", short: "auth" },
  suspicious: { tone: "var(--amber)", short: "susp" },
  likely_counterfeit: { tone: "var(--rose)", short: "cnft" },
  unverified: { tone: "var(--ink-3)", short: "unv" },
};

type DetectorMeta = {
  desc: string;
  /** Pull these metadata keys (in order) and label them for the expanded view. */
  metrics: { key: string; label: string }[];
};

const DETECTOR_META: Record<string, DetectorMeta> = {
  rosette_fft: {
    desc:
      "FFT analysis of the print rosette pattern. Counterfeits typically show degraded peak strength below 0.6 due to inkjet vs. offset printing.",
    metrics: [
      { key: "peak_strength", label: "peak strength" },
      { key: "peak", label: "peak" },
    ],
  },
  rosette: {
    desc:
      "FFT analysis of the print rosette pattern. Counterfeits typically show degraded peak strength below 0.6 due to inkjet vs. offset.",
    metrics: [{ key: "peak_strength", label: "peak strength" }],
  },
  holo_parallax: {
    desc:
      "Farnebäck optical flow on the front + tilt-30 shot pair. Authentic foil substrates show characteristic parallax flow.",
    metrics: [
      { key: "flow_ratio", label: "flow ratio" },
      { key: "foil_mask_pct", label: "foil mask %" },
    ],
  },
  holographic: {
    desc:
      "Farnebäck optical flow on the front + tilt-30 shot pair. Authentic foil substrates show characteristic parallax flow.",
    metrics: [{ key: "flow_ratio", label: "flow ratio" }],
  },
  substrate_bstar: {
    desc:
      "CIELAB b* channel under controlled flash. WOTC-era stock has b* between -2.0 and -2.5; reprints drift positive.",
    metrics: [
      { key: "b_delta", label: "b* delta" },
      { key: "border_mad", label: "border MAD" },
    ],
  },
  substrate: {
    desc:
      "CIELAB b* channel under controlled flash. WOTC-era stock has b* between -2.0 and -2.5; reprints drift positive.",
    metrics: [{ key: "b_delta", label: "b* delta" }],
  },
  cielab_profile: {
    desc:
      "Full CIELAB color profile compared to manufacturer reference via mean ΔE76.",
    metrics: [{ key: "delta_e", label: "ΔE76" }],
  },
  color: {
    desc:
      "Full CIELAB color profile compared to manufacturer reference via mean ΔE76.",
    metrics: [{ key: "delta_e", label: "ΔE76" }],
  },
  typography_ocr: {
    desc:
      "RapidOCR title-region read, normalized Levenshtein distance vs. the identified card name.",
    metrics: [
      { key: "edit_distance", label: "edit distance" },
      { key: "ocr_text", label: "ocr text" },
    ],
  },
  typography: {
    desc:
      "RapidOCR title-region read, normalized Levenshtein distance vs. the identified card name.",
    metrics: [{ key: "edit_distance", label: "edit distance" }],
  },
  knn_reference: {
    desc:
      "Mean cosine distance to top-3 nearest authentic exemplars. Catches manifold cases that centroid distance misses.",
    metrics: [
      { key: "mean_distance", label: "mean distance" },
      { key: "n_references", label: "n refs" },
    ],
  },
  embedding: {
    desc:
      "DinoV2 embedding distance from the per-variant centroid. Coarser than k-NN reference; catches off-manifold submissions.",
    metrics: [{ key: "centroid_distance", label: "centroid dist" }],
  },
  embedding_anomaly: {
    desc:
      "DinoV2 embedding distance from the per-variant centroid. Coarser than k-NN reference; catches off-manifold submissions.",
    metrics: [{ key: "centroid_distance", label: "centroid dist" }],
  },
};

export default function AuthenticityDetectors({ detectors }: { detectors: DetectorScore[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  return (
    <div>
      <h4
        className="mono"
        style={{
          fontSize: 10,
          color: "var(--ink-3)",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
          margin: "0 0 10px",
          fontWeight: 500,
        }}
      >
        Detector breakdown · {detectors.length} signals
      </h4>
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {detectors.map((d) => (
          <DetectorRow
            key={d.detector}
            d={d}
            expanded={expanded === d.detector}
            onToggle={() =>
              setExpanded((cur) => (cur === d.detector ? null : d.detector))
            }
          />
        ))}
      </div>
    </div>
  );
}

function DetectorRow({
  d,
  expanded,
  onToggle,
}: {
  d: DetectorScore;
  expanded: boolean;
  onToggle: () => void;
}) {
  const tone = VERDICT_TONE[d.verdict];
  const meta = DETECTOR_META[d.detector] ?? { desc: "", metrics: [] };

  // Pull display metrics from the API metadata bag. Each detector
  // publishes its own typed payload; we surface the most-relevant one
  // and a generic confidence column as a fallback.
  const metrics: Array<[string, string]> = [];
  for (const m of meta.metrics) {
    const v = d.metadata?.[m.key];
    if (v !== undefined && v !== null) {
      metrics.push([m.label, formatMetric(v)]);
    }
  }
  metrics.push(["conf", `${(d.confidence * 100).toFixed(0)}%`]);

  return (
    <div
      style={{
        background: "var(--bg-2)",
        border: "1px solid var(--line-2)",
        borderRadius: 4,
        overflow: "hidden",
        transition: "background 120ms",
      }}
    >
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        style={{
          width: "100%",
          padding: "10px 14px",
          background: "transparent",
          border: 0,
          cursor: "pointer",
          display: "grid",
          gridTemplateColumns: "minmax(120px, 180px) 1fr 70px 60px 14px",
          gap: 12,
          alignItems: "center",
          color: "inherit",
          fontFamily: "inherit",
          textAlign: "left",
        }}
      >
        <span
          className="mono"
          style={{ fontSize: 10, color: "var(--ink-2)", letterSpacing: "0.04em" }}
        >
          {d.detector}
        </span>
        <div
          style={{
            height: 4,
            background: "var(--bg-3)",
            borderRadius: 1,
            overflow: "hidden",
          }}
        >
          <div
            style={{ width: `${d.score * 100}%`, height: "100%", background: tone.tone }}
          />
        </div>
        <span
          className="mono"
          style={{
            fontSize: 11,
            color: "var(--ink-1)",
            textAlign: "right",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {(d.score * 100).toFixed(0)}%
        </span>
        <span
          className="mono"
          style={{
            fontSize: 9,
            color: tone.tone,
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            textAlign: "right",
          }}
        >
          {tone.short}
        </span>
        <span
          style={{
            color: "var(--ink-4)",
            fontSize: 14,
            transform: expanded ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 120ms",
          }}
        >
          ›
        </span>
      </button>

      {expanded && (
        <div
          style={{
            padding: "12px 14px",
            borderTop: "1px solid var(--line-2)",
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          {meta.desc && (
            <p
              style={{
                margin: 0,
                fontSize: 12,
                color: "var(--ink-2)",
                lineHeight: 1.5,
              }}
            >
              {meta.desc}
            </p>
          )}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: `repeat(${Math.max(metrics.length, 1)}, 1fr)`,
              gap: 6,
            }}
          >
            {metrics.map(([l, v]) => (
              <div
                key={l}
                style={{ background: "var(--bg-1)", borderRadius: 3, padding: "6px 8px" }}
              >
                <div
                  className="mono"
                  style={{
                    fontSize: 9,
                    color: "var(--ink-4)",
                    textTransform: "uppercase",
                    letterSpacing: "0.06em",
                  }}
                >
                  {l}
                </div>
                <div
                  className="mono"
                  style={{
                    fontSize: 11,
                    color: "var(--ink-1)",
                    fontVariantNumeric: "tabular-nums",
                    marginTop: 2,
                  }}
                >
                  {v}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function formatMetric(v: unknown): string {
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toString();
    return v.toFixed(Math.abs(v) >= 10 ? 1 : 3);
  }
  if (typeof v === "string") return v.length > 32 ? `${v.slice(0, 32)}…` : v;
  if (typeof v === "boolean") return v ? "true" : "false";
  return "—";
}
