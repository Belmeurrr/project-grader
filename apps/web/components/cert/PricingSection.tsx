/**
 * <PricingSection> — TCGplayer market-price comps on /cert/[id].
 *
 * Slots below <PopulationPanel> in the right column of the Split
 * layout. Renders nothing when no pricing data is on the cert
 * payload (graceful no-op for personal-use without an API key,
 * unidentified cards, or lookup failures).
 *
 * Server component — no interactive state. The "Updated …" timestamp
 * uses the cache-fill time, not request time, so users can see when
 * the comp data is stale.
 */

import type { Pricing } from "@/lib/cert";

type Props = { pricing: Pricing };

export default function PricingSection({ pricing }: Props) {
  const fetched = new Date(pricing.fetched_at);
  const fetchedRel = relativeDays(fetched);
  const cells: Array<{ l: string; v: number | null; sub: string; primary: boolean }> = [
    {
      l: "Market",
      v: pricing.market_price_usd,
      sub: "current marketplace",
      primary: true,
    },
    { l: "Median", v: pricing.median_price_usd, sub: "all listings", primary: false },
    { l: "Low", v: pricing.low_price_usd, sub: "marketplace floor", primary: false },
  ];

  return (
    <section>
      <SectionLabel hint="tcgplayer · 24h cache">Pricing</SectionLabel>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 6,
          marginTop: 12,
        }}
      >
        {cells.map((c) => (
          <PriceCell
            key={c.l}
            label={c.l}
            value={c.v}
            sub={c.sub}
            primary={c.primary}
          />
        ))}
      </div>

      {pricing.direct_low_price_usd !== null && (
        <div
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-3)",
            letterSpacing: "0.04em",
            marginTop: 10,
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span style={{ color: "var(--ink-4)" }}>TCGplayer Direct low</span>
          <span style={{ color: "var(--ink-1)" }}>
            ${pricing.direct_low_price_usd.toFixed(2)}
          </span>
        </div>
      )}

      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginTop: 12,
          paddingTop: 10,
          borderTop: "1px solid var(--line-2)",
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-4)",
            letterSpacing: "0.04em",
          }}
        >
          updated {fetchedRel}
        </span>
        {pricing.product_url && (
          <a
            href={pricing.product_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--lime)",
              letterSpacing: "0.04em",
              textTransform: "uppercase",
            }}
          >
            View on TCGplayer ↗
          </a>
        )}
      </div>
    </section>
  );
}

function PriceCell({
  label,
  value,
  sub,
  primary,
}: {
  label: string;
  value: number | null;
  sub: string;
  primary: boolean;
}) {
  const display = value !== null ? `$${value.toFixed(2)}` : "—";
  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 5,
        padding: "12px 14px",
      }}
    >
      <div
        className="mono"
        style={{
          fontSize: 9,
          color: "var(--ink-3)",
          letterSpacing: "0.18em",
          textTransform: "uppercase",
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "var(--font-instrument-serif), serif",
          fontSize: 28,
          color: value !== null ? (primary ? "var(--lime)" : "var(--ink-0)") : "var(--ink-4)",
          lineHeight: 1,
          marginTop: 4,
          letterSpacing: "-0.02em",
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {display}
      </div>
      <div
        className="mono"
        style={{
          fontSize: 9,
          color: "var(--ink-4)",
          marginTop: 4,
          letterSpacing: "0.02em",
        }}
      >
        {sub}
      </div>
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

function relativeDays(then: Date): string {
  const ms = Date.now() - then.getTime();
  const hours = Math.floor(ms / (1000 * 60 * 60));
  if (hours < 1) return "just now";
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return days === 1 ? "1d ago" : `${days}d ago`;
}
