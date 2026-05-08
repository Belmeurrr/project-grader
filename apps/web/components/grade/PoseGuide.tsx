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
    // Bright corner brackets ("L"-shapes) at the four corners of the target
    // card region — much more visible than a dashed rectangle and immediately
    // communicates "fill this area." Pulses gently so it reads even on busy
    // backgrounds. Subtle inner card silhouette stays for shape reference.
    const x0 = 130,
      y0 = 65,
      x1 = 270,
      y1 = 255;
    const armLen = 22;
    const sw = 4;
    const c = guideColor;
    return (
      <svg
        aria-hidden
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
        viewBox="0 0 400 300"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Four L-bracket corners */}
        <g stroke={c} strokeWidth={sw} fill="none" strokeLinecap="round">
          <polyline points={`${x0},${y0 + armLen} ${x0},${y0} ${x0 + armLen},${y0}`} />
          <polyline points={`${x1 - armLen},${y0} ${x1},${y0} ${x1},${y0 + armLen}`} />
          <polyline points={`${x0},${y1 - armLen} ${x0},${y1} ${x0 + armLen},${y1}`} />
          <polyline points={`${x1 - armLen},${y1} ${x1},${y1} ${x1},${y1 - armLen}`} />
          <animate
            attributeName="opacity"
            values="0.6;1;0.6"
            dur="1.6s"
            repeatCount="indefinite"
          />
        </g>
        <text
          x="200"
          y="160"
          textAnchor="middle"
          fontFamily="var(--font-geist-mono), monospace"
          fontSize="9"
          fill={c}
          opacity="0.7"
          letterSpacing="0.22em"
        >
          FIT CARD HERE
        </text>
      </svg>
    );
  }
  if (pose === "tilt") {
    return (
      <svg
        aria-hidden
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
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
        style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
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
