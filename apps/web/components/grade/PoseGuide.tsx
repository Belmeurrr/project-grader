/**
 * Pose-specific guide overlay drawn inside the viewfinder canvas.
 *
 * Mirrors the SVG poses in `docs/design/capture.jsx` so the active
 * shot tells the user exactly how to orient the card. The overlay is
 * purely cosmetic — the actual quality gate runs server-side after
 * upload — but it's load-bearing for keeping the median retake count
 * down (see capture-guidance acceptance test in TODO).
 */
import type { WizardPose } from "./shots";

type Props = { pose: WizardPose; guideColor: string };

export function PoseGuide({ pose, guideColor }: Props) {
  if (pose === "flat") {
    return (
      <svg
        aria-hidden
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid meet"
      >
        <rect
          x="130"
          y="65"
          width="140"
          height="190"
          rx="6"
          fill="rgba(190,242,100,0.04)"
          stroke={guideColor}
          strokeWidth="1.4"
          strokeDasharray="6 4"
        />
        <rect
          x="138"
          y="73"
          width="124"
          height="174"
          rx="4"
          fill="url(#flatSilhouette)"
          stroke="var(--ink-3)"
          strokeWidth="0.5"
          opacity="0.6"
        />
        <defs>
          <linearGradient id="flatSilhouette" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#3a1a08" />
            <stop offset="50%" stopColor="#7a3a14" />
            <stop offset="100%" stopColor="#f97316" stopOpacity="0.6" />
          </linearGradient>
        </defs>
        <line x1="200" y1="155" x2="200" y2="165" stroke={guideColor} strokeWidth="1" />
        <line x1="195" y1="160" x2="205" y2="160" stroke={guideColor} strokeWidth="1" />
        <line x1="130" y1="160" x2="125" y2="160" stroke={guideColor} strokeWidth="1.5" />
        <line x1="270" y1="160" x2="275" y2="160" stroke={guideColor} strokeWidth="1.5" />
        <line x1="200" y1="65" x2="200" y2="60" stroke={guideColor} strokeWidth="1.5" />
        <line x1="200" y1="255" x2="200" y2="260" stroke={guideColor} strokeWidth="1.5" />
        <text
          x="200"
          y="50"
          textAnchor="middle"
          fontFamily="var(--font-geist-mono), monospace"
          fontSize="7"
          fill={guideColor}
          letterSpacing="0.18em"
        >
          FILL THE GUIDE
        </text>
      </svg>
    );
  }
  if (pose === "tilt") {
    return (
      <svg
        aria-hidden
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid meet"
      >
        <polygon
          points="148,95 260,85 270,225 138,235"
          fill="rgba(190,242,100,0.04)"
          stroke={guideColor}
          strokeWidth="1.4"
          strokeDasharray="6 4"
        />
        <polygon
          points="156,103 252,93 262,217 146,227"
          fill="url(#tiltSilhouette)"
          opacity="0.6"
        />
        <defs>
          <linearGradient id="tiltSilhouette" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#3a1a08" />
            <stop offset="60%" stopColor="#7a3a14" />
            <stop offset="100%" stopColor="#f97316" stopOpacity="0.5" />
          </linearGradient>
        </defs>
        <g transform="translate(200,260)">
          <line x1="-44" y1="0" x2="44" y2="0" stroke="var(--ink-4)" strokeWidth="0.8" />
          <line x1="0" y1="0" x2="38" y2="-22" stroke={guideColor} strokeWidth="1.2" />
          <text
            x="22"
            y="-6"
            fontFamily="var(--font-geist-mono), monospace"
            fontSize="8"
            fill={guideColor}
          >
            30°
          </text>
        </g>
        <text
          x="200"
          y="74"
          textAnchor="middle"
          fontFamily="var(--font-geist-mono), monospace"
          fontSize="7"
          fill={guideColor}
          letterSpacing="0.18em"
        >
          TILT TOP TOWARD CAMERA
        </text>
      </svg>
    );
  }
  if (pose.startsWith("corner-")) {
    const which = pose.split("-")[1] ?? "tl";
    const isLeft = which.endsWith("l");
    const isTop = which.startsWith("t");
    return (
      <svg
        aria-hidden
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }}
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid meet"
      >
        <rect
          x={isLeft ? 130 : 60}
          y={isTop ? 80 : 30}
          width="210"
          height="220"
          rx="14"
          fill="url(#cornerSilhouette)"
          opacity="0.55"
        />
        <defs>
          <radialGradient id="cornerSilhouette">
            <stop offset="0%" stopColor="#f97316" stopOpacity="0.7" />
            <stop offset="80%" stopColor="#3a1a08" />
          </radialGradient>
        </defs>
        <circle
          cx="200"
          cy="150"
          r="62"
          fill="none"
          stroke={guideColor}
          strokeWidth="1.4"
          strokeDasharray="6 4"
        />
        <circle
          cx="200"
          cy="150"
          r="76"
          fill="none"
          stroke={guideColor}
          strokeWidth="0.6"
          opacity="0.5"
        />
        <line x1="200" y1="135" x2="200" y2="165" stroke={guideColor} strokeWidth="1" />
        <line x1="185" y1="150" x2="215" y2="150" stroke={guideColor} strokeWidth="1" />
        <text
          x="200"
          y="245"
          textAnchor="middle"
          fontFamily="var(--font-geist-mono), monospace"
          fontSize="7"
          fill={guideColor}
          letterSpacing="0.18em"
        >
          FILL INNER RING · {which.toUpperCase()}
        </text>
      </svg>
    );
  }
  return null;
}
