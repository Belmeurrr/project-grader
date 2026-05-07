"use client";

/**
 * The Workbench — `/` (homepage).
 *
 * Personal-use grading dashboard. Sidebar (search + filter + system
 * telemetry) on the left, main canvas on the right (hero + stats +
 * collection grid + inspector). Ported from docs/design/app.jsx, with
 * the static `window.COLLECTION` mock replaced by a live
 * `GET /submissions` call against the owner-side API.
 *
 * Client component because:
 *   - Auth header for the API call comes from `useAuthedFetch` (a hook)
 *   - Sidebar search / filter / inspector selection are React state
 *   - Hero capture animation is a setInterval timer
 *
 * Empty state: when the user has 0 submissions (the common case on a
 * fresh dev box), the sidebar drops its filter counts to zero and the
 * collection grid renders a "Start grading" CTA pointing at `/grade`.
 */

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import {
  type SubmissionFull,
  listSubmissions,
  useAuthedFetch,
} from "@/lib/submission";

// ─────────────────────────────────────────────────────────────────────────────
// View-model — the shape the Workbench cares about, derived from the
// API's SubmissionFull. Centralizes the field names so a schema change
// only ripples through the projector.
type CardVM = {
  id: string;
  status: SubmissionFull["status"];
  shortId: string;
  name: string;
  set: string;
  number: string;
  capturedAt: string;
  capturedTs: number;
  centering: number | null;
  corners: number | null;
  edges: number | null;
  surface: number | null;
  final: number | null;
  conf: number;
  authVerdict: SubmissionFull["authenticity"] extends infer A
    ? A extends { verdict: infer V }
      ? V
      : "unverified"
    : "unverified";
  authConf: number;
  /** 1-letter glyph fallback when no real card image exists. */
  glyph: string;
  color: string;
  /** Set when something is preliminary (corners/surface stub) */
  flag: string | null;
};

const FALLBACK_GLYPH = "◰";
const COLOR_PALETTE = [
  "#f97316",
  "#a3a3a3",
  "#fbbf24",
  "#60a5fa",
  "#dc2626",
  "#7c3aed",
  "#262626",
  "#facc15",
  "#e0e7ff",
  "#ef4444",
  "#f472b6",
  "#7e22ce",
];

function deriveColor(id: string): string {
  // Stable per-id pick — hashing the first 6 chars of the UUID.
  let h = 0;
  for (let i = 0; i < id.length && i < 8; i += 1) h = (h * 31 + id.charCodeAt(i)) | 0;
  return COLOR_PALETTE[Math.abs(h) % COLOR_PALETTE.length] ?? "#a3a3a3";
}

function projectSubmission(s: SubmissionFull): CardVM {
  const primary = s.grades.find((g) => g.scheme === "psa") ?? s.grades[0];
  const auth = s.authenticity;
  const card = s.identified_card;
  const captured = s.completed_at ?? s.created_at;
  const flagParts: string[] = [];
  if (primary?.final === null && (primary?.corners === null || primary?.surface === null)) {
    flagParts.push("Preliminary — corners/surface heads not trained");
  }
  if (s.status === "rejected_quality") flagParts.push("Rejected at quality gate");
  if (s.rejection_reason) flagParts.push(s.rejection_reason);

  return {
    id: s.id,
    status: s.status,
    shortId: `PG-${s.id.slice(0, 4).toUpperCase()}-${s.id.slice(9, 13).toUpperCase()}`,
    name: card?.name ?? "Unidentified card",
    set: card?.set_code ?? "—",
    number: card?.card_number ?? "—",
    capturedAt: new Date(captured).toLocaleString("en-US", {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    }),
    capturedTs: new Date(captured).getTime(),
    centering: primary?.centering ?? null,
    corners: primary?.corners ?? null,
    edges: primary?.edges ?? null,
    surface: primary?.surface ?? null,
    final: primary?.final ?? null,
    conf: primary?.confidence ?? 0,
    authVerdict: (auth?.verdict ?? "unverified") as CardVM["authVerdict"],
    authConf: auth?.confidence ?? 0,
    glyph: FALLBACK_GLYPH,
    color: deriveColor(s.id),
    flag: flagParts[0] ?? null,
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Color tokens
const VERDICT_COLOR: Record<string, string> = {
  authentic: "var(--lime)",
  suspicious: "var(--amber)",
  likely_counterfeit: "var(--rose)",
  unverified: "var(--ink-4)",
};
const VERDICT_LABEL: Record<string, string> = {
  authentic: "Authentic",
  suspicious: "Suspicious",
  likely_counterfeit: "Counterfeit",
  unverified: "Unverified",
};
function gradeTone(g: number | null): string {
  if (g === null) return "var(--ink-3)";
  if (g >= 9) return "var(--lime-bright)";
  if (g >= 8) return "var(--lime)";
  if (g >= 7) return "var(--amber)";
  return "var(--rose)";
}

// ─────────────────────────────────────────────────────────────────────────────
type FilterKind = "all" | "top" | "pending" | "flagged";

export default function WorkbenchPage() {
  const authedFetch = useAuthedFetch();
  const [data, setData] = useState<SubmissionFull[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState<FilterKind>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const rows = await listSubmissions(authedFetch, { limit: 50 });
        if (!cancelled) {
          setData(rows);
          if (rows[0] && !selectedId) setSelectedId(rows[0].id);
        }
      } catch (e) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e));
          setData([]);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const cards: CardVM[] = useMemo(
    () => (data ? data.map(projectSubmission) : []),
    [data],
  );

  const visible = useMemo(() => {
    let list = cards;
    if (filter === "top") list = list.filter((c) => c.final !== null && c.final >= 9);
    if (filter === "pending")
      list = list.filter((c) => c.corners === null || c.surface === null);
    if (filter === "flagged")
      list = list.filter((c) => c.flag !== null || c.authVerdict !== "authentic");
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(
        (c) =>
          c.name.toLowerCase().includes(q) ||
          c.set.toLowerCase().includes(q) ||
          c.id.toLowerCase().includes(q),
      );
    }
    return list;
  }, [cards, filter, query]);

  const selected = useMemo(
    () => cards.find((c) => c.id === selectedId) ?? cards[0] ?? null,
    [cards, selectedId],
  );

  const stats = useMemo(() => deriveStats(cards), [cards]);

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "260px 1fr",
        minHeight: "calc(100vh - 49px)",
        background: "var(--bg-0)",
      }}
    >
      <Sidebar
        filter={filter}
        setFilter={setFilter}
        query={query}
        setQuery={setQuery}
        stats={stats}
        loading={data === null}
      />
      <Main
        cards={visible}
        selected={selected}
        selectedId={selectedId}
        onSelect={setSelectedId}
        loading={data === null}
        empty={data?.length === 0}
        error={error}
        stats={stats}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SIDEBAR
type Stats = {
  total: number;
  flagged: number;
  pendingHeads: number;
  topTier: number;
  meanFinal: number | null;
  highest: { name: string; grade: number } | null;
  lowest: { name: string; grade: number } | null;
};

function deriveStats(cards: CardVM[]): Stats {
  const finals = cards.map((c) => c.final).filter((g): g is number => g !== null);
  const mean =
    finals.length > 0 ? finals.reduce((a, b) => a + b, 0) / finals.length : null;
  const sortedHigh = [...cards]
    .filter((c) => c.final !== null)
    .sort((a, b) => (b.final ?? 0) - (a.final ?? 0));
  const highest = sortedHigh[0]
    ? { name: sortedHigh[0].name, grade: sortedHigh[0].final ?? 0 }
    : null;
  const sortedLow = [...sortedHigh].reverse();
  const lowest = sortedLow[0]
    ? { name: sortedLow[0].name, grade: sortedLow[0].final ?? 0 }
    : null;
  return {
    total: cards.length,
    flagged: cards.filter((c) => c.authVerdict !== "authentic" || c.flag !== null)
      .length,
    pendingHeads: cards.filter((c) => c.corners === null || c.surface === null)
      .length,
    topTier: cards.filter((c) => c.final !== null && c.final >= 9).length,
    meanFinal: mean,
    highest,
    lowest,
  };
}

function Sidebar({
  filter,
  setFilter,
  query,
  setQuery,
  stats,
  loading,
}: {
  filter: FilterKind;
  setFilter: (v: FilterKind) => void;
  query: string;
  setQuery: (v: string) => void;
  stats: Stats;
  loading: boolean;
}) {
  return (
    <aside
      style={{
        borderRight: "1px solid var(--line)",
        background: "var(--bg-1)",
        display: "flex",
        flexDirection: "column",
        position: "sticky",
        top: 49,
        height: "calc(100vh - 49px)",
        overflow: "auto",
      }}
    >
      <div style={{ padding: "14px 16px 8px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: "var(--bg-2)",
            border: "1px solid var(--line)",
            borderRadius: 6,
            padding: "7px 10px",
          }}
        >
          <span style={{ color: "var(--ink-4)", fontSize: 12 }}>⌕</span>
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by name, set, cert…"
            style={{
              flex: 1,
              background: "transparent",
              border: 0,
              outline: "none",
              color: "var(--ink-1)",
              fontSize: 12,
              fontFamily: "inherit",
            }}
          />
        </div>
      </div>

      <nav
        style={{
          padding: "8px 8px",
          display: "flex",
          flexDirection: "column",
          gap: 1,
        }}
      >
        <SidebarItem
          label="All graded"
          count={stats.total}
          active={filter === "all"}
          onClick={() => setFilter("all")}
        />
        <SidebarItem
          label="Top tier (≥ 9)"
          count={stats.topTier}
          active={filter === "top"}
          onClick={() => setFilter("top")}
        />
        <SidebarItem
          label="Pending heads"
          count={stats.pendingHeads}
          active={filter === "pending"}
          onClick={() => setFilter("pending")}
          tone="amber"
        />
        <SidebarItem
          label="Flagged"
          count={stats.flagged}
          active={filter === "flagged"}
          onClick={() => setFilter("flagged")}
          tone="rose"
        />
      </nav>

      <div
        style={{
          marginTop: "auto",
          borderTop: "1px solid var(--line-2)",
          padding: "14px 16px",
        }}
      >
        <div
          className="mono"
          style={{
            fontSize: 9,
            textTransform: "uppercase",
            letterSpacing: "0.18em",
            color: "var(--ink-4)",
            marginBottom: 10,
          }}
        >
          System
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 7 }}>
          <SystemRow label="API" value={loading ? "loading…" : "ok"} tone="ok" />
          <SystemRow label="Detectors" value="7/7 wired" tone="ok" />
          <SystemRow
            label="Pipeline"
            value={loading ? "—" : `${stats.total} graded`}
          />
          <SystemRow label="Catalog" value="local-disk" />
          <SystemRow label="Flywheel" value="Mac · queued" tone="amber" />
        </div>
      </div>
    </aside>
  );
}

function SidebarItem({
  label,
  count,
  active,
  onClick,
  tone,
}: {
  label: string;
  count: number;
  active?: boolean;
  onClick?: () => void;
  tone?: "amber" | "rose";
}) {
  const dotColor =
    tone === "amber" ? "var(--amber)" : tone === "rose" ? "var(--rose)" : null;
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "7px 10px",
        background: active ? "rgba(190,242,100,0.08)" : "transparent",
        border: 0,
        borderLeft: active ? "2px solid var(--lime)" : "2px solid transparent",
        borderRadius: 4,
        cursor: "pointer",
        color: active ? "var(--ink-0)" : "var(--ink-2)",
        fontSize: 12,
        fontFamily: "inherit",
        textAlign: "left",
        transition: "background 120ms",
      }}
    >
      <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
        {dotColor && (
          <span
            style={{
              width: 5,
              height: 5,
              borderRadius: 3,
              background: dotColor,
            }}
          />
        )}
        {label}
      </span>
      <span
        className="mono"
        style={{
          fontSize: 10,
          color: active ? "var(--lime)" : "var(--ink-4)",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {count}
      </span>
    </button>
  );
}

function SystemRow({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "ok" | "amber";
}) {
  const dot =
    tone === "ok" ? "var(--lime)" : tone === "amber" ? "var(--amber)" : null;
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
      <span
        className="mono"
        style={{ fontSize: 10, color: "var(--ink-3)", letterSpacing: "0.04em" }}
      >
        {label}
      </span>
      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {dot && (
          <span
            style={{
              width: 5,
              height: 5,
              borderRadius: 3,
              background: dot,
              boxShadow: `0 0 6px ${dot}`,
            }}
          />
        )}
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-1)",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {value}
        </span>
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN CANVAS
function Main({
  cards,
  selected,
  selectedId,
  onSelect,
  loading,
  empty,
  error,
  stats,
}: {
  cards: CardVM[];
  selected: CardVM | null;
  selectedId: string | null;
  onSelect: (id: string) => void;
  loading: boolean;
  empty: boolean | undefined;
  error: string | null;
  stats: Stats;
}) {
  const showInspector = selected !== null;
  return (
    <main style={{ display: "flex", flexDirection: "column", minWidth: 0 }}>
      <Hero stats={stats} />
      <CollectionStats stats={stats} />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: showInspector ? "minmax(0, 1fr) 480px" : "minmax(0, 1fr)",
          gap: 0,
          borderTop: "1px solid var(--line)",
          minHeight: 400,
        }}
      >
        <CollectionView
          cards={cards}
          selectedId={selectedId}
          onSelect={onSelect}
          loading={loading}
          empty={empty}
          error={error}
        />
        {selected && <Inspector card={selected} />}
      </div>
      <Footer />
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// HERO — animated 8-shot capture sequence (kept from the design as the
// only client-side animation; it's the visual identity of the workbench).
const SHOT_PROTOCOL = [
  { k: "front", t: "Front", d: "Centered, dewarp anchor" },
  { k: "back", t: "Back", d: "Reverse face, watermark" },
  { k: "tl", t: "Corner · TL", d: "Macro, top-left" },
  { k: "tr", t: "Corner · TR", d: "Macro, top-right" },
  { k: "bl", t: "Corner · BL", d: "Macro, bottom-left" },
  { k: "br", t: "Corner · BR", d: "Macro, bottom-right" },
  { k: "flash", t: "Flash", d: "Paired-flash for substrate" },
  { k: "tilt", t: "Tilt", d: "30° for holographic-parallax" },
];

function Hero({ stats }: { stats: Stats }) {
  const [shotIdx, setShotIdx] = useState(0);
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    const tick = setInterval(() => {
      setProgress((p) => {
        if (p >= 100) {
          setShotIdx((i) => (i + 1) % SHOT_PROTOCOL.length);
          return 0;
        }
        return p + 4;
      });
    }, 60);
    return () => clearInterval(tick);
  }, []);

  return (
    <section
      style={{
        padding: "36px 24px 28px",
        borderBottom: "1px solid var(--line)",
        background:
          "radial-gradient(ellipse 800px 300px at 30% 0%, rgba(190,242,100,0.04), transparent 70%), var(--bg-0)",
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 480px",
          gap: 32,
          alignItems: "center",
        }}
      >
        <div>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 14,
              flexWrap: "wrap",
            }}
          >
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--lime)",
                letterSpacing: "0.22em",
                textTransform: "uppercase",
                padding: "3px 8px",
                background: "rgba(190,242,100,0.08)",
                border: "1px solid rgba(190,242,100,0.25)",
                borderRadius: 3,
              }}
            >
              Personal · v0.4.1
            </span>
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--ink-3)",
                letterSpacing: "0.06em",
              }}
            >
              not a service · graded by you, for you
            </span>
          </div>
          <h1
            style={{
              margin: 0,
              fontSize: 56,
              fontWeight: 500,
              lineHeight: 1.0,
              letterSpacing: "-0.025em",
              color: "var(--ink-0)",
              maxWidth: 600,
            }}
          >
            A grading bench
            <br />
            for one collector,
            <br />
            <span
              className="serif"
              style={{ color: "var(--lime)", fontWeight: 400 }}
            >
              built like a lab.
            </span>
          </h1>
          <p
            style={{
              margin: "20px 0 0",
              maxWidth: 520,
              fontSize: 14,
              lineHeight: 1.6,
              color: "var(--ink-2)",
            }}
          >
            Eight calibrated shots per card · seven counterfeit detectors ·
            calibrated uncertainty on every grade. No SLAs. No marketplace.
            Just an honest readout of what the cards in your binder really are.
          </p>
          <div
            style={{
              display: "flex",
              gap: 16,
              marginTop: 22,
            }}
          >
            <Link
              href="/grade"
              className="pg-btn pg-btn-primary"
              style={{ padding: "10px 18px", fontSize: 13 }}
            >
              + Start a new submission
            </Link>
            <span
              className="mono"
              style={{
                fontSize: 11,
                color: "var(--ink-3)",
                alignSelf: "center",
                letterSpacing: "0.04em",
              }}
            >
              walks the 8-shot protocol →
            </span>
          </div>
          <div
            style={{
              display: "flex",
              gap: 32,
              marginTop: 28,
              paddingTop: 20,
              borderTop: "1px solid var(--line-2)",
              flexWrap: "wrap",
            }}
          >
            <MetricCell
              label="Cards graded"
              value={stats.total.toString()}
              sub="this binder"
            />
            <MetricCell
              label="Avg final"
              value={stats.meanFinal !== null ? stats.meanFinal.toFixed(1) : "—"}
              sub="PSA scheme"
              tone="var(--lime)"
            />
            <MetricCell label="Detectors" value="7/7" sub="ensemble wired" />
            <MetricCell
              label="Reference catalog"
              value="local"
              sub="DinoV2 + pgvector"
            />
          </div>
        </div>

        <CaptureSequence shotIdx={shotIdx} progress={progress} />
      </div>
    </section>
  );
}

function MetricCell({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub: string;
  tone?: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
      <div
        className="mono"
        style={{
          fontSize: 9,
          textTransform: "uppercase",
          letterSpacing: "0.16em",
          color: "var(--ink-4)",
        }}
      >
        {label}
      </div>
      <div
        className="mono"
        style={{
          fontSize: 22,
          fontWeight: 500,
          color: tone || "var(--ink-0)",
          fontVariantNumeric: "tabular-nums",
          letterSpacing: "-0.01em",
          lineHeight: 1,
        }}
      >
        {value}
      </div>
      <div className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>
        {sub}
      </div>
    </div>
  );
}

function CaptureSequence({
  shotIdx,
  progress,
}: {
  shotIdx: number;
  progress: number;
}) {
  const current = SHOT_PROTOCOL[shotIdx] ?? SHOT_PROTOCOL[0]!;
  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 10,
        padding: 18,
        boxShadow:
          "0 24px 60px rgba(0,0,0,0.4), inset 0 0 0 1px rgba(255,255,255,0.02)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 14,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: 4,
              background: "var(--lime)",
              boxShadow: "0 0 8px var(--lime)",
              animation: "pulse 1.6s ease-in-out infinite",
            }}
          />
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-2)",
              letterSpacing: "0.12em",
              textTransform: "uppercase",
            }}
          >
            Capture · live
          </span>
        </div>
        <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>
          {String(shotIdx + 1).padStart(2, "0")} /{" "}
          {String(SHOT_PROTOCOL.length).padStart(2, "0")}
        </span>
      </div>

      <div
        style={{
          position: "relative",
          aspectRatio: "5/7",
          maxWidth: 280,
          margin: "0 auto",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: 10,
            overflow: "hidden",
            background:
              "linear-gradient(135deg, #f9731622 0%, #f9731608 60%, #0a0a0a 100%)",
            border: "1px solid var(--line)",
          }}
        >
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 100,
              color: "#f97316",
              opacity: 0.5,
              filter: "drop-shadow(0 0 24px #f9731688)",
            }}
          >
            🜂
          </div>
          <div
            style={{
              position: "absolute",
              left: 0,
              right: 0,
              top: `${progress}%`,
              height: 2,
              background:
                "linear-gradient(90deg, transparent, var(--lime), transparent)",
              boxShadow: "0 0 12px var(--lime)",
              opacity: 0.8,
            }}
          />
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(8, 1fr)",
          gap: 4,
          marginTop: 14,
        }}
      >
        {SHOT_PROTOCOL.map((s, i) => {
          const done = i < shotIdx;
          const active = i === shotIdx;
          return (
            <div
              key={s.k}
              style={{
                aspectRatio: "5/7",
                border: `1px solid ${
                  active
                    ? "var(--lime)"
                    : done
                    ? "var(--lime-deep)"
                    : "var(--line)"
                }`,
                borderRadius: 3,
                background: active
                  ? "rgba(190,242,100,0.08)"
                  : done
                  ? "rgba(190,242,100,0.04)"
                  : "var(--bg-2)",
                position: "relative",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                transition: "all 200ms",
              }}
            >
              <span
                className="mono"
                style={{
                  fontSize: 9,
                  color: active
                    ? "var(--lime)"
                    : done
                    ? "var(--lime-deep)"
                    : "var(--ink-4)",
                  textTransform: "uppercase",
                  letterSpacing: "0.04em",
                }}
              >
                {s.k}
              </span>
              {done && (
                <span
                  style={{
                    position: "absolute",
                    top: 2,
                    right: 2,
                    fontSize: 8,
                    color: "var(--lime-deep)",
                  }}
                >
                  ✓
                </span>
              )}
            </div>
          );
        })}
      </div>

      <div
        style={{
          marginTop: 14,
          padding: 10,
          background: "var(--bg-2)",
          border: "1px solid var(--line-2)",
          borderRadius: 5,
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div>
          <div style={{ fontSize: 12, color: "var(--ink-1)", fontWeight: 500 }}>
            {current.t}
          </div>
          <div
            style={{ fontSize: 10, color: "var(--ink-3)", marginTop: 2 }}
          >
            {current.d}
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--lime)",
              letterSpacing: "0.04em",
            }}
          >
            QUALITY · {(0.82 + Math.sin(progress / 30) * 0.06).toFixed(2)}
          </div>
          <div className="mono" style={{ fontSize: 9, color: "var(--ink-4)" }}>
            blur · glare · perspective
          </div>
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COLLECTION STATS
function CollectionStats({ stats }: { stats: Stats }) {
  const cells = [
    { l: "Population", v: stats.total.toString(), s: "cards graded" },
    {
      l: "Mean final",
      v: stats.meanFinal !== null ? stats.meanFinal.toFixed(1) : "—",
      s: stats.total > 0 ? `n=${stats.total}` : "no data yet",
      tone: stats.meanFinal !== null ? "var(--lime)" : "var(--ink-3)",
    },
    {
      l: "Highest",
      v: stats.highest ? stats.highest.grade.toFixed(1) : "—",
      s: stats.highest ? stats.highest.name : "—",
      tone: "var(--lime-bright)",
    },
    {
      l: "Lowest",
      v: stats.lowest ? stats.lowest.grade.toFixed(1) : "—",
      s: stats.lowest ? stats.lowest.name : "—",
      tone: "var(--amber)",
    },
    {
      l: "Auth flagged",
      v: stats.flagged.toString(),
      s: stats.flagged > 0 ? "review queue" : "all clean",
      tone: stats.flagged > 0 ? "var(--amber)" : "var(--ink-2)",
    },
    {
      l: "Pending heads",
      v: stats.pendingHeads.toString(),
      s: "corners · surface",
      tone: "var(--ink-2)",
    },
  ];
  return (
    <section
      style={{
        padding: 0,
        borderBottom: "1px solid var(--line)",
        display: "grid",
        gridTemplateColumns: "repeat(6, 1fr)",
        gap: 1,
        background: "var(--line-2)",
      }}
    >
      {cells.map((s) => (
        <div
          key={s.l}
          style={{
            background: "var(--bg-0)",
            padding: "14px 16px",
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <span
            className="mono"
            style={{
              fontSize: 9,
              textTransform: "uppercase",
              letterSpacing: "0.18em",
              color: "var(--ink-4)",
            }}
          >
            {s.l}
          </span>
          <span
            style={{
              fontSize: 22,
              fontWeight: 500,
              color: s.tone || "var(--ink-0)",
              letterSpacing: "-0.02em",
              lineHeight: 1,
              fontVariantNumeric: "tabular-nums",
            }}
          >
            {s.v}
          </span>
          <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            {s.s}
          </span>
        </div>
      ))}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// COLLECTION VIEW (grid)
function CollectionView({
  cards,
  selectedId,
  onSelect,
  loading,
  empty,
  error,
}: {
  cards: CardVM[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  loading: boolean;
  empty: boolean | undefined;
  error: string | null;
}) {
  return (
    <div
      style={{
        padding: "16px 24px 32px",
        borderRight: "1px solid var(--line)",
        background: "var(--bg-0)",
        minHeight: 400,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 14,
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            textTransform: "uppercase",
            letterSpacing: "0.18em",
            color: "var(--ink-3)",
          }}
        >
          {loading ? "loading…" : `${cards.length} cards · sorted by capture`}
        </span>
      </div>

      {error && (
        <div
          style={{
            padding: 14,
            background: "rgba(251,113,133,0.06)",
            border: "1px solid rgba(251,113,133,0.25)",
            borderRadius: 5,
            color: "var(--rose)",
            fontSize: 12,
          }}
        >
          Failed to load submissions: {error}
        </div>
      )}

      {!loading && !error && empty && <EmptyState />}

      {!loading && !error && !empty && cards.length === 0 && (
        <p style={{ color: "var(--ink-3)", fontSize: 13, fontStyle: "italic" }}>
          No matches for the current filter.
        </p>
      )}

      {cards.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
            gap: 14,
          }}
        >
          {cards.map((c) => (
            <CardCell
              key={c.id}
              card={c}
              selected={selectedId === c.id}
              onSelect={() => onSelect(c.id)}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function EmptyState() {
  return (
    <div
      style={{
        marginTop: 24,
        padding: "40px 24px",
        background: "var(--bg-1)",
        border: "1px dashed var(--line)",
        borderRadius: 8,
        textAlign: "center",
        display: "flex",
        flexDirection: "column",
        gap: 14,
        alignItems: "center",
      }}
    >
      <span
        className="mono"
        style={{
          fontSize: 10,
          color: "var(--ink-4)",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
        }}
      >
        empty binder
      </span>
      <h3
        className="serif"
        style={{
          margin: 0,
          fontSize: 28,
          color: "var(--ink-0)",
          fontStyle: "normal",
          fontFamily: "var(--font-instrument-serif), serif",
        }}
      >
        No grades yet.
      </h3>
      <p
        style={{
          margin: 0,
          maxWidth: 360,
          fontSize: 13,
          color: "var(--ink-2)",
          lineHeight: 1.6,
        }}
      >
        Walk a card through the 8-shot wizard and the cert page comes back
        with your grade, the authenticity ensemble, and the defect heatmap.
      </p>
      <Link
        href="/grade"
        className="pg-btn pg-btn-primary"
        style={{ padding: "10px 18px", fontSize: 13, marginTop: 4 }}
      >
        + Start grading
      </Link>
    </div>
  );
}

function CardCell({
  card,
  selected,
  onSelect,
}: {
  card: CardVM;
  selected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      style={{
        background: "var(--bg-1)",
        border: `1px solid ${selected ? "rgba(190,242,100,0.5)" : "var(--line)"}`,
        borderRadius: 8,
        padding: 14,
        cursor: "pointer",
        textAlign: "left",
        fontFamily: "inherit",
        color: "inherit",
        transition: "border-color 120ms, background 120ms",
        boxShadow: selected
          ? "0 0 0 1px rgba(190,242,100,0.2), 0 8px 24px rgba(190,242,100,0.06)"
          : "none",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 12,
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 9,
            color: "var(--ink-3)",
            letterSpacing: "0.02em",
          }}
        >
          {card.shortId}
        </span>
        <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
          <span
            style={{
              width: 6,
              height: 6,
              borderRadius: 3,
              background: VERDICT_COLOR[card.authVerdict],
              boxShadow: `0 0 6px ${VERDICT_COLOR[card.authVerdict]}`,
            }}
          />
          <span
            className="mono"
            style={{
              fontSize: 9,
              color: VERDICT_COLOR[card.authVerdict],
              textTransform: "uppercase",
              letterSpacing: "0.06em",
            }}
          >
            {VERDICT_LABEL[card.authVerdict]}
          </span>
        </div>
      </div>
      <div style={{ display: "flex", gap: 12 }}>
        <CardArt card={card} />
        <div
          style={{
            flex: 1,
            minWidth: 0,
            display: "flex",
            flexDirection: "column",
            gap: 8,
          }}
        >
          <div>
            <div
              style={{
                fontSize: 14,
                fontWeight: 600,
                color: "var(--ink-0)",
                letterSpacing: "-0.005em",
                lineHeight: 1.15,
                whiteSpace: "nowrap",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {card.name}
            </div>
            <div
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--ink-3)",
                marginTop: 3,
                letterSpacing: "0.02em",
              }}
            >
              {card.set} · {card.number}
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
            <span
              style={{
                fontFamily: "var(--font-instrument-serif), serif",
                fontSize: 36,
                lineHeight: 1,
                color: gradeTone(card.final),
                letterSpacing: "-0.02em",
              }}
            >
              {card.final !== null ? card.final.toFixed(1) : "—"}
            </span>
          </div>
          {card.final !== null ? (
            <div
              style={{
                height: 2,
                background: "var(--bg-3)",
                borderRadius: 1,
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  width: `${(card.final / 10) * 100}%`,
                  height: "100%",
                  background: gradeTone(card.final),
                }}
              />
            </div>
          ) : (
            <div
              className="mono"
              style={{ fontSize: 10, color: "var(--amber)" }}
            >
              Preliminary
            </div>
          )}
          <Link
            href={`/cert/${card.id}`}
            onClick={(e) => e.stopPropagation()}
            className="mono"
            style={{
              fontSize: 9,
              color: "var(--ink-3)",
              letterSpacing: "0.04em",
              textAlign: "right",
              textDecoration: "underline",
              textDecorationColor: "var(--line-2)",
            }}
          >
            open cert ↗
          </Link>
        </div>
      </div>
    </button>
  );
}

function CardArt({ card }: { card: CardVM }) {
  // Stylized placeholder — in a future iteration we'd swap to the
  // canonical scan via a presigned URL, like the cert page does.
  return (
    <div
      style={{
        width: 100,
        height: 140,
        position: "relative",
        flexShrink: 0,
        borderRadius: 4,
        overflow: "hidden",
        background: `linear-gradient(135deg, ${card.color}22 0%, ${card.color}08 60%, #0a0a0a 100%)`,
        border: "1px solid var(--line)",
        boxShadow:
          "inset 0 0 0 1px rgba(255,255,255,0.02), 0 4px 16px rgba(0,0,0,0.4)",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 44,
          color: card.color,
          opacity: 0.55,
          filter: `drop-shadow(0 0 12px ${card.color}55)`,
        }}
      >
        {card.glyph}
      </div>
      <div
        className="mono"
        style={{
          position: "absolute",
          left: "8%",
          right: "8%",
          top: "8%",
          fontSize: 8,
          color: "rgba(255,255,255,0.5)",
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{card.set.slice(0, 8)}</span>
        <span>{card.number}</span>
      </div>
      <div
        style={{
          position: "absolute",
          inset: 0,
          pointerEvents: "none",
          background:
            "repeating-linear-gradient(0deg, rgba(255,255,255,0.02) 0 1px, transparent 1px 3px)",
          mixBlendMode: "overlay",
        }}
      />
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// INSPECTOR
function Inspector({ card }: { card: CardVM }) {
  return (
    <aside
      style={{
        background: "var(--bg-1)",
        padding: "18px 22px 32px",
        display: "flex",
        flexDirection: "column",
        gap: 22,
        maxHeight: "calc(100vh - 49px)",
        overflow: "auto",
        position: "sticky",
        top: 49,
        alignSelf: "flex-start",
      }}
    >
      <div>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            marginBottom: 12,
          }}
        >
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-3)",
              letterSpacing: "0.18em",
              textTransform: "uppercase",
            }}
          >
            Inspector · cert
          </span>
          <Link
            href={`/cert/${card.id}`}
            className="pg-btn pg-btn-ghost"
            style={{ padding: "4px 8px", fontSize: 10 }}
          >
            Open cert ↗
          </Link>
        </div>
        <span
          className="mono"
          style={{
            fontSize: 11,
            color: "var(--ink-3)",
            letterSpacing: "0.02em",
          }}
        >
          {card.shortId}
        </span>
        <div style={{ marginTop: 10, display: "flex", gap: 14 }}>
          <CardArt card={card} />
          <div
            style={{
              flex: 1,
              minWidth: 0,
              display: "flex",
              flexDirection: "column",
              gap: 6,
            }}
          >
            <div
              style={{
                fontSize: 22,
                fontWeight: 600,
                color: "var(--ink-0)",
                letterSpacing: "-0.015em",
                lineHeight: 1.1,
              }}
            >
              {card.name}
            </div>
            <div className="mono" style={{ fontSize: 10, color: "var(--ink-2)" }}>
              {card.set} · {card.number}
            </div>
            <div
              style={{
                marginTop: "auto",
                display: "flex",
                flexDirection: "column",
                gap: 6,
              }}
            >
              <div
                className="mono"
                style={{
                  fontSize: 9,
                  color: "var(--ink-4)",
                  textTransform: "uppercase",
                  letterSpacing: "0.18em",
                }}
              >
                Captured
              </div>
              <div
                className="mono"
                style={{ fontSize: 11, color: "var(--ink-1)" }}
              >
                {card.capturedAt}
              </div>
            </div>
          </div>
        </div>
        {card.flag && (
          <div
            style={{
              marginTop: 12,
              padding: "8px 12px",
              background: "rgba(251,191,36,0.06)",
              border: "1px solid rgba(251,191,36,0.25)",
              borderRadius: 4,
              fontSize: 11,
              color: "var(--amber)",
              lineHeight: 1.5,
            }}
          >
            ⚠ {card.flag}
          </div>
        )}
      </div>

      <SubgradeBlock card={card} />

      <div>
        <SectionHead
          label="Authenticity · ensemble"
          meta={`${(card.authConf * 100).toFixed(0)}% conf`}
        />
        <div
          style={{
            padding: "10px 12px",
            background: "rgba(190,242,100,0.04)",
            border: `1px solid ${VERDICT_COLOR[card.authVerdict]}40`,
            borderRadius: 4,
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: 3,
                background: VERDICT_COLOR[card.authVerdict],
                boxShadow: `0 0 6px ${VERDICT_COLOR[card.authVerdict]}`,
              }}
            />
            <span
              style={{
                fontSize: 13,
                fontWeight: 600,
                color: VERDICT_COLOR[card.authVerdict],
                letterSpacing: "-0.005em",
              }}
            >
              {VERDICT_LABEL[card.authVerdict]}
            </span>
          </div>
          <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>
            7 detectors · majority vote
          </span>
        </div>
      </div>
    </aside>
  );
}

function SectionHead({ label, meta }: { label: string; meta?: string }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        justifyContent: "space-between",
        borderBottom: "1px solid var(--line-2)",
        paddingBottom: 6,
        marginBottom: 12,
      }}
    >
      <span
        className="mono"
        style={{
          fontSize: 10,
          textTransform: "uppercase",
          letterSpacing: "0.18em",
          color: "var(--ink-3)",
        }}
      >
        {label}
      </span>
      {meta && (
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-4)",
            letterSpacing: "0.04em",
          }}
        >
          {meta}
        </span>
      )}
    </div>
  );
}

function SubgradeBlock({ card }: { card: CardVM }) {
  const items: Array<{ k: string; v: number | null; primary?: boolean }> = [
    { k: "Centering", v: card.centering },
    { k: "Corners", v: card.corners },
    { k: "Edges", v: card.edges },
    { k: "Surface", v: card.surface },
    { k: "Final", v: card.final, primary: true },
  ];
  return (
    <div>
      <SectionHead label="Subgrades · PSA" meta="psa-1to10" />
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: 1,
          background: "var(--line-2)",
          border: "1px solid var(--line-2)",
        }}
      >
        {items.map((it) => (
          <div
            key={it.k}
            style={{
              background: it.primary ? "rgba(190,242,100,0.04)" : "var(--bg-1)",
              padding: "14px 12px",
              display: "flex",
              flexDirection: "column",
              gap: 8,
              minHeight: 100,
            }}
          >
            <div
              className="mono"
              style={{
                fontSize: 9,
                textTransform: "uppercase",
                letterSpacing: "0.18em",
                color: it.primary ? "var(--lime)" : "var(--ink-4)",
              }}
            >
              {it.k}
            </div>
            <div
              style={{
                fontFamily: "var(--font-instrument-serif), serif",
                fontSize: it.primary ? 36 : 24,
                lineHeight: 1,
                color: gradeTone(it.v),
                letterSpacing: "-0.02em",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {it.v !== null ? it.v.toFixed(1) : "—"}
            </div>
            {it.v !== null && (
              <div
                style={{
                  height: 2,
                  background: "var(--bg-3)",
                  borderRadius: 1,
                  overflow: "hidden",
                  marginTop: "auto",
                }}
              >
                <div
                  style={{
                    width: `${(it.v / 10) * 100}%`,
                    height: "100%",
                    background: gradeTone(it.v),
                  }}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// FOOTER
function Footer() {
  const versions: Array<[string, string, boolean]> = [
    ["centering", "v0.4.1-geom", false],
    ["edges", "v0.4.0-mad", false],
    ["corners", "stub", true],
    ["surface", "stub", true],
    ["ident", "dinov2-vitb14", false],
    ["detect", "yolo11-seg-v0.4.1", false],
    ["counterfeit", "ensemble-v0.3.2", false],
  ];
  return (
    <footer
      style={{
        padding: "20px 24px",
        borderTop: "1px solid var(--line)",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        background: "var(--bg-1)",
        flexWrap: "wrap",
        gap: 18,
      }}
    >
      <div style={{ display: "flex", gap: 18, flexWrap: "wrap" }}>
        {versions.map(([k, v, stub]) => (
          <span
            key={k}
            className="mono"
            style={{ fontSize: 10, color: "var(--ink-3)" }}
          >
            <span style={{ color: "var(--ink-4)" }}>{k}</span>=
            <span style={{ color: stub ? "var(--amber)" : "var(--ink-1)" }}>
              {v}
            </span>
          </span>
        ))}
      </div>
      <span className="mono" style={{ fontSize: 10, color: "var(--ink-4)" }}>
        not affiliated with PSA, BGS, SGC, or CGC
      </span>
    </footer>
  );
}
