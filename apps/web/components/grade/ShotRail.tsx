/**
 * Left-rail shot list. Click any row to make it the active shot in the
 * viewfinder. Mirrors the design's `<ShotRail/>` plus the optional-shots
 * boost callout.
 */
"use client";

import type { ShotKind } from "@/lib/submission";
import { WIZARD_SHOTS, type WizardShot } from "./shots";

export type ShotStatus = "captured" | "active" | "pending" | "failed";

export type ShotRailRow = WizardShot & {
  status: ShotStatus;
  idx: number;
};

export function ShotRail({
  rows,
  onSelect,
}: {
  rows: ShotRailRow[];
  onSelect: (kind: ShotKind) => void;
}) {
  const reqTotal = WIZARD_SHOTS.filter((s) => s.required).length;
  const optTotal = WIZARD_SHOTS.length - reqTotal;
  return (
    <aside
      aria-label="Shot list"
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 8,
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
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
            fontSize: 10,
            color: "var(--ink-2)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            margin: 0,
            fontWeight: 500,
          }}
        >
          Shot list
        </h2>
        <span className="mono" style={{ fontSize: 10, color: "var(--ink-4)" }}>
          {reqTotal} req · {optTotal} opt
        </span>
      </div>

      <ol
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 4,
          margin: 0,
          padding: 0,
          listStyle: "none",
        }}
      >
        {rows.map((row) => (
          <li key={row.kind}>
            <ShotRailItem row={row} onClick={() => onSelect(row.kind)} />
          </li>
        ))}
      </ol>

      <div
        style={{
          marginTop: 4,
          padding: 10,
          background: "var(--bg-2)",
          borderRadius: 4,
          borderLeft: "2px solid var(--lime)",
        }}
      >
        <div
          className="mono"
          style={{
            fontSize: 9,
            color: "var(--lime)",
            letterSpacing: "0.18em",
            textTransform: "uppercase",
          }}
        >
          Optional shots boost
        </div>
        <div
          style={{
            fontSize: 11,
            color: "var(--ink-2)",
            marginTop: 6,
            lineHeight: 1.45,
          }}
        >
          <span style={{ color: "var(--ink-1)" }}>+ Tilt 30°</span> unlocks holo
          parallax detector. <span style={{ color: "var(--ink-1)" }}>+ Flash</span>{" "}
          unlocks substrate b*.{" "}
          <span style={{ color: "var(--ink-1)" }}>+ Corners</span> sharpens
          corner subgrade.
        </div>
      </div>
    </aside>
  );
}

function ShotRailItem({
  row,
  onClick,
}: {
  row: ShotRailRow;
  onClick: () => void;
}) {
  const isActive = row.status === "active";
  const isCaptured = row.status === "captured";
  const isFailed = row.status === "failed";
  const tone = isFailed
    ? "var(--rose)"
    : isCaptured || isActive
      ? "var(--lime)"
      : "var(--ink-5)";

  const ariaLabel = `${row.label} · ${row.required ? "required" : "optional"} · ${row.status}`;

  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={ariaLabel}
      aria-current={isActive ? "step" : undefined}
      style={{
        all: "unset",
        cursor: "pointer",
        display: "grid",
        gridTemplateColumns: "24px 1fr auto",
        gap: 10,
        padding: "8px 10px",
        borderRadius: 4,
        width: "100%",
        boxSizing: "border-box",
        background: isActive ? "var(--bg-2)" : "transparent",
        border:
          "1px solid " + (isActive ? "var(--line)" : "transparent"),
        transition: "background 150ms, border-color 150ms",
      }}
    >
      <ShotGlyph status={row.status} />
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
          <span
            style={{
              fontSize: 12,
              fontWeight: 500,
              color:
                row.status === "pending" ? "var(--ink-3)" : "var(--ink-0)",
            }}
          >
            {row.label}
          </span>
          {row.required ? (
            <span
              className="mono"
              style={{
                fontSize: 8,
                color: "var(--amber)",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              req
            </span>
          ) : (
            <span
              className="mono"
              style={{
                fontSize: 8,
                color: "var(--ink-4)",
                letterSpacing: "0.1em",
                textTransform: "uppercase",
              }}
            >
              opt
            </span>
          )}
        </div>
        <div
          className="mono"
          style={{ fontSize: 9, color: "var(--ink-4)", marginTop: 2 }}
        >
          {row.kind}
        </div>
      </div>
      <span
        className="mono"
        style={{
          fontSize: 9,
          color: tone,
          letterSpacing: "0.1em",
          textTransform: "uppercase",
        }}
        aria-hidden
      >
        {isCaptured ? "✓" : isFailed ? "!" : isActive ? "●" : "·"}
      </span>
    </button>
  );
}

function ShotGlyph({ status }: { status: ShotStatus }) {
  if (status === "captured") {
    return (
      <div
        aria-hidden
        style={{
          width: 22,
          height: 22,
          borderRadius: 11,
          background: "rgba(190,242,100,0.12)",
          border: "1px solid rgba(190,242,100,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <svg width="10" height="10" viewBox="0 0 12 12" fill="none">
          <path
            d="M2 6.5L5 9L10 3"
            stroke="var(--lime)"
            strokeWidth="1.6"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>
    );
  }
  if (status === "active") {
    return (
      <div
        aria-hidden
        style={{
          width: 22,
          height: 22,
          borderRadius: 11,
          background: "rgba(190,242,100,0.08)",
          border: "1px solid rgba(190,242,100,0.6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: 4,
            background: "var(--lime)",
            boxShadow: "0 0 6px var(--lime)",
          }}
        />
      </div>
    );
  }
  if (status === "failed") {
    return (
      <div
        aria-hidden
        style={{
          width: 22,
          height: 22,
          borderRadius: 11,
          background: "rgba(251,113,133,0.1)",
          border: "1px solid rgba(251,113,133,0.5)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: "var(--rose)",
          fontSize: 11,
          fontWeight: 600,
        }}
      >
        !
      </div>
    );
  }
  return (
    <div
      aria-hidden
      style={{
        width: 22,
        height: 22,
        borderRadius: 11,
        border: "1px dashed var(--ink-5)",
      }}
    />
  );
}
