"use client";

/**
 * <CardVisionPanel> — Workbench-design Card Vision with mode tabs.
 *
 * Replaces the old <CardVisionSlider> on /cert/[id]. Adds:
 *   - Canonical / Flash / Overlay / Tilt mode toggle
 *   - Region overlay rendered over the card (centering frame, corners,
 *     edges) so the defect heatmap sits on the actual canonical scan
 *     instead of next to it
 *   - Opacity slider for Overlay mode (matches the design's mockup)
 *
 * Falls back gracefully when an image is null:
 *   - Canonical missing → render a placeholder card frame with regions
 *   - Flash missing     → Flash tab disabled
 *   - Tilt missing      → Tilt tab disabled
 */

import { useEffect, useState } from "react";

import type { CertImage, Region, RegionSeverity } from "@/lib/cert";

type Mode = "canonical" | "flash" | "overlay" | "tilt";

type Props = {
  images: CertImage | null;
  regions: Region[];
  cardName: string;
};

const SEVERITY_COLOR: Record<RegionSeverity, { fill: string; stroke: string }> = {
  ok: { fill: "rgba(190,242,100,0.15)", stroke: "rgba(190,242,100,0.6)" },
  minor: { fill: "rgba(251,191,36,0.18)", stroke: "rgba(251,191,36,0.7)" },
  major: { fill: "rgba(251,113,133,0.22)", stroke: "rgba(251,113,133,0.8)" },
  unknown: { fill: "rgba(82,82,91,0.18)", stroke: "rgba(82,82,91,0.5)" },
};

export default function CardVisionPanel({ images, regions, cardName }: Props) {
  const front = images?.front_canonical_url ?? null;
  const flash = images?.front_flash_url ?? null;
  const tilt = images?.tilt_url ?? null;

  // Pick a sensible default mode based on what we have.
  const initialMode: Mode = front ? "canonical" : flash ? "flash" : tilt ? "tilt" : "canonical";
  const [mode, setMode] = useState<Mode>(initialMode);
  const [opacity, setOpacity] = useState(0.45);

  useEffect(() => {
    setMode(initialMode);
  }, [initialMode]);

  const tabs: { k: Mode; l: string; available: boolean }[] = [
    { k: "canonical", l: "Canonical", available: !!front },
    { k: "flash", l: "Flash", available: !!flash },
    { k: "overlay", l: "Overlay", available: !!front },
    { k: "tilt", l: "Tilt", available: !!tilt },
  ];

  const activeUrl =
    mode === "flash" ? flash : mode === "tilt" ? tilt : front;

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <SectionLabel hint="card-vision · ml/canonicals">Card vision</SectionLabel>

      {/* Mode tabs */}
      <div
        style={{
          display: "flex",
          gap: 4,
          padding: 3,
          background: "var(--bg-2)",
          border: "1px solid var(--line)",
          borderRadius: 5,
        }}
      >
        {tabs.map((m) => (
          <button
            key={m.k}
            onClick={() => m.available && setMode(m.k)}
            disabled={!m.available}
            title={m.available ? "" : "Capture not available for this submission"}
            style={{
              flex: 1,
              padding: "6px 10px",
              background: mode === m.k ? "var(--bg-3)" : "transparent",
              border: 0,
              borderRadius: 3,
              cursor: m.available ? "pointer" : "not-allowed",
              color:
                mode === m.k
                  ? "var(--ink-0)"
                  : m.available
                  ? "var(--ink-3)"
                  : "var(--ink-5)",
              fontFamily: "inherit",
              fontSize: 11,
              fontWeight: 500,
              transition: "all 120ms",
              opacity: m.available ? 1 : 0.5,
            }}
          >
            {m.l}
          </button>
        ))}
      </div>

      {/* Card frame */}
      <div style={{ position: "relative" }}>
        <CardFrame
          imageUrl={activeUrl}
          mode={mode}
          opacity={opacity}
          alt={`${cardName} — ${mode}`}
        />
        <RegionOverlay regions={regions} />
      </div>

      {mode === "overlay" && (
        <div style={{ display: "flex", alignItems: "center", gap: 10, paddingTop: 4 }}>
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-3)",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            opacity
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            aria-label="Region overlay opacity"
            style={{ flex: 1, accentColor: "var(--lime)" }}
          />
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-2)",
              minWidth: 32,
              textAlign: "right",
            }}
          >
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      <SeverityLegend />
    </section>
  );
}

function CardFrame({
  imageUrl,
  mode,
  opacity,
  alt,
}: {
  imageUrl: string | null;
  mode: Mode;
  opacity: number;
  alt: string;
}) {
  const isTilt = mode === "tilt";
  const isOverlay = mode === "overlay";

  return (
    <div
      style={{
        aspectRatio: "5/7",
        width: "100%",
        position: "relative",
        borderRadius: 9,
        background: "#0a0a0a",
        border: "1px solid var(--line-2)",
        boxShadow:
          "0 30px 80px rgba(0,0,0,0.6), 0 0 0 1px rgba(255,255,255,0.02) inset",
        overflow: "hidden",
        transform: isTilt
          ? "perspective(1200px) rotateY(-8deg) rotateX(4deg)"
          : "none",
        transition: "transform 400ms cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      {imageUrl ? (
        // Use a plain <img> rather than next/image: presigned URLs are
        // dynamic and shouldn't go through Next's image optimization
        // pipeline (which would try to refetch them after the presign
        // expires). The cert page already lazy-loads above the fold.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageUrl}
          alt={alt}
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
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
          }}
        >
          <span className="mono" style={{ letterSpacing: "0.16em", textTransform: "uppercase" }}>
            no canonical
          </span>
        </div>
      )}

      {isOverlay && (
        <div
          style={{
            position: "absolute",
            inset: 0,
            background: `rgba(10,10,10,${opacity})`,
            pointerEvents: "none",
          }}
        />
      )}
    </div>
  );
}

function RegionOverlay({ regions }: { regions: Region[] }) {
  const centering = regions.find((r) => r.kind === "centering");
  const corners = regions.filter((r) => r.kind === "corner");
  const edges = regions.filter((r) => r.kind === "edge");

  const cornerPos: Record<string, React.CSSProperties> = {
    top_left: { top: "3%", left: "3%" },
    top_right: { top: "3%", right: "3%" },
    bottom_left: { bottom: "3%", left: "3%" },
    bottom_right: { bottom: "3%", right: "3%" },
  };
  const edgePos: Record<string, React.CSSProperties> = {
    top: { top: "3%", left: "14%", right: "14%", height: 4 },
    bottom: { bottom: "3%", left: "14%", right: "14%", height: 4 },
    left: { left: "3%", top: "14%", bottom: "14%", width: 4 },
    right: { right: "3%", top: "14%", bottom: "14%", width: 4 },
  };

  return (
    <div style={{ position: "absolute", inset: 0, pointerEvents: "none" }}>
      {centering && (
        <div
          style={{
            position: "absolute",
            top: "14%",
            bottom: "14%",
            left: "10%",
            right: "10%",
            border: `1.5px dashed ${SEVERITY_COLOR[centering.severity].stroke}`,
            borderRadius: 4,
          }}
        />
      )}
      {corners.map((c) => {
        const sev = SEVERITY_COLOR[c.severity];
        const pos = cornerPos[c.position];
        if (!pos) return null;
        return (
          <div
            key={c.position}
            style={{
              position: "absolute",
              ...pos,
              width: 24,
              height: 24,
              borderRadius: 12,
              border: `1.5px solid ${sev.stroke}`,
              background: sev.fill,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {c.severity !== "ok" && (
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: 3,
                  background: sev.stroke,
                }}
              />
            )}
          </div>
        );
      })}
      {edges.map((e) => {
        const sev = SEVERITY_COLOR[e.severity];
        const pos = edgePos[e.position];
        if (!pos) return null;
        return (
          <div
            key={e.position}
            style={{
              position: "absolute",
              ...pos,
              background: sev.fill,
              border: `1px solid ${sev.stroke}`,
              borderRadius: 2,
            }}
          />
        );
      })}
    </div>
  );
}

function SeverityLegend() {
  const items: { k: RegionSeverity; l: string; c: string }[] = [
    { k: "ok", l: "OK", c: "var(--lime)" },
    { k: "minor", l: "Minor", c: "var(--amber)" },
    { k: "major", l: "Major", c: "var(--rose)" },
    { k: "unknown", l: "Unknown", c: "var(--ink-5)" },
  ];
  return (
    <div style={{ display: "flex", gap: 14, flexWrap: "wrap", paddingTop: 4 }}>
      {items.map((it) => (
        <span
          key={it.k}
          style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              background: it.c,
              opacity: 0.5,
              border: `1px solid ${it.c}`,
            }}
          />
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-3)",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            {it.l}
          </span>
        </span>
      ))}
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
