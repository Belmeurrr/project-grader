/**
 * Public cert page — `/cert/[id]`.
 *
 * Server component that fetches the cert via the public API endpoint
 * and renders Grade + AuthenticityResult. ISR-cached (revalidate
 * window matches the API's Cache-Control); no client-side JS required.
 *
 * Phase 1 MVP per the project plan: every grade gets a shareable URL
 * with a full evidence trail. This is the artifact a user posts to
 * Twitter / pins to a marketplace listing / shares with a buyer.
 */

import { notFound } from "next/navigation";

import {
  type AuthenticityVerdict,
  type Certificate,
  type DetectorScore,
  type Grade,
  type Region,
  type RegionSeverity,
  fetchCertificate,
} from "@/lib/cert";

// Force static generation w/ revalidation — Next.js ISR. The cert
// payload is immutable in practice once a submission completes, so
// pin a 5-minute window matching the API's Cache-Control.
export const revalidate = 300;

type Params = { id: string };

export default async function CertPage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { id } = await params;
  const cert = await fetchCertificate(id);
  if (!cert) notFound();

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-8 px-6 py-12">
      <Header cert={cert} />
      <GradesSection cert={cert} />
      <AuthenticitySection cert={cert} />
      <Footer cert={cert} />
    </main>
  );
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}) {
  const { id } = await params;
  return {
    title: `Cert ${id.slice(0, 8)}… — Project Grader`,
    description: "AI-graded card with full evidence trail.",
  };
}

function Header({ cert }: { cert: Certificate }) {
  return (
    <header className="flex flex-col gap-2 border-b border-zinc-800 pb-6">
      <p className="text-xs uppercase tracking-widest text-lime-500">
        Project Grader • Certificate
      </p>
      <h1 className="font-mono text-lg break-all text-zinc-100">
        {cert.cert_id}
      </h1>
      <p className="text-xs text-zinc-500">
        Completed {new Date(cert.completed_at).toLocaleString()}
      </p>
    </header>
  );
}

function GradesSection({ cert }: { cert: Certificate }) {
  // PSA-scheme grade is the canonical one; if multiple schemes ever land,
  // PSA is what the cert page leads with. With `noUncheckedIndexedAccess`
  // on, `cert.grades[0]` widens to `Grade | undefined`, so we use an
  // explicit guard rather than `!` — covers both the empty-array case
  // and narrows `primary` to `Grade` for the rest of the function.
  const primary: Grade | undefined =
    cert.grades.find((g) => g.scheme === "psa") ?? cert.grades[0];
  if (!primary) {
    return (
      <section className="rounded-xl border border-zinc-800 p-6 text-zinc-400">
        <h2 className="mb-2 text-base font-medium text-zinc-200">Grade</h2>
        <p className="text-sm">No grade computed for this submission.</p>
      </section>
    );
  }
  return (
    <section>
      <h2 className="mb-3 text-base font-medium text-zinc-200">
        Grade ({primary.scheme.toUpperCase()})
      </h2>
      {primary.final === null && (
        <div className="mb-3 rounded-full border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-xs font-medium text-amber-300">
          Preliminary subgrades — final grade unavailable until corners +
          surface analysis ships.
        </div>
      )}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <GradeStat label="Centering" value={primary.centering} />
        <GradeStat label="Corners" value={primary.corners} />
        <GradeStat label="Edges" value={primary.edges} />
        <GradeStat label="Surface" value={primary.surface} />
        <GradeStat label="Final" value={primary.final} highlight />
      </div>
      <p className="mt-3 text-xs text-zinc-500">
        Confidence: {(primary.confidence * 100).toFixed(0)}%
      </p>
      <DamageHeatmap regions={cert.regions} />
    </section>
  );
}

// --------------------------------------------------------------------------
// Damage-heatmap overlay.
//
// Phase-1 MVP: a card-shaped div with absolutely-positioned cells, one
// per entry in `cert.regions`. Pure CSS — no canvas, no WebGL. The
// overlay is supplementary to the existing per-criterion grades grid;
// it visualizes WHERE on the card the defects sit, not the numeric
// scores (those stay in the grid above).
//
// Layout uses percentages so the card scales with viewport. Aspect
// ratio is fixed at 2.5 / 3.5 (standard trading-card proportions) via
// `aspect-[5/7]` so the corners visually land at the corners of a
// real card rendering.
// --------------------------------------------------------------------------

const SEVERITY_RING_CLASSES: Record<RegionSeverity, string> = {
  ok: "border-lime-500/60 bg-lime-500/15",
  minor: "border-amber-500/60 bg-amber-500/15",
  major: "border-red-500/60 bg-red-500/15",
  unknown: "border-zinc-700 bg-zinc-800/40",
};

function DamageHeatmap({ regions }: { regions: Region[] }) {
  if (regions.length === 0) return null;
  const corners = regions.filter((r) => r.kind === "corner");
  const edges = regions.filter((r) => r.kind === "edge");
  const centering = regions.find((r) => r.kind === "centering");
  const surface = regions.find((r) => r.kind === "surface");

  return (
    <div className="mt-6">
      <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-400">
        Defect map
      </h3>
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:gap-6">
        <div className="relative mx-auto aspect-[5/7] w-48 overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40 sm:mx-0">
          {centering && (
            <div
              className={`pointer-events-none absolute rounded-md border-2 ${SEVERITY_RING_CLASSES[centering.severity]}`}
              style={{ top: "18%", bottom: "18%", left: "12%", right: "12%" }}
              aria-label={`Centering: ${centering.severity}`}
            />
          )}
          {edges.map((e) => (
            <EdgeCell key={`edge-${e.position}`} region={e} />
          ))}
          {corners.map((c) => (
            <CornerCell key={`corner-${c.position}`} region={c} />
          ))}
          {surface && (
            <div
              className={`pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full border px-2 py-1 text-[10px] font-medium ${SEVERITY_RING_CLASSES[surface.severity]} text-zinc-200`}
              aria-label={`Surface: ${surface.severity}`}
            >
              {surface.severity === "unknown"
                ? "surface analysis pending"
                : "surface"}
            </div>
          )}
        </div>
        <DamageHeatmapLegend />
      </div>
      <p className="mt-2 text-xs text-zinc-500">
        Visualization only. Numeric subgrades above are the canonical scores.
      </p>
    </div>
  );
}

function EdgeCell({ region }: { region: Region }) {
  // Edges sit just inside each side of the card frame. Corner regions
  // already cover the corners; the edge rectangles inset 12% on each
  // end so they don't overlap.
  const cls = SEVERITY_RING_CLASSES[region.severity];
  const common = "pointer-events-none absolute rounded-sm border";
  switch (region.position) {
    case "top":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ top: "5%", left: "12%", right: "12%", height: "3%" }}
          aria-label="Edge: top"
        />
      );
    case "bottom":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ bottom: "5%", left: "12%", right: "12%", height: "3%" }}
          aria-label="Edge: bottom"
        />
      );
    case "left":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ left: "5%", top: "12%", bottom: "12%", width: "3%" }}
          aria-label="Edge: left"
        />
      );
    case "right":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ right: "5%", top: "12%", bottom: "12%", width: "3%" }}
          aria-label="Edge: right"
        />
      );
    default:
      return null;
  }
}

function CornerCell({ region }: { region: Region }) {
  const cls = SEVERITY_RING_CLASSES[region.severity];
  const common =
    "pointer-events-none absolute h-5 w-5 rounded-full border-2";
  switch (region.position) {
    case "top_left":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ top: "3%", left: "3%" }}
          aria-label="Corner: top-left"
        />
      );
    case "top_right":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ top: "3%", right: "3%" }}
          aria-label="Corner: top-right"
        />
      );
    case "bottom_left":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ bottom: "3%", left: "3%" }}
          aria-label="Corner: bottom-left"
        />
      );
    case "bottom_right":
      return (
        <div
          className={`${common} ${cls}`}
          style={{ bottom: "3%", right: "3%" }}
          aria-label="Corner: bottom-right"
        />
      );
    default:
      return null;
  }
}

function DamageHeatmapLegend() {
  const items: Array<{ label: string; severity: RegionSeverity }> = [
    { label: "OK", severity: "ok" },
    { label: "Minor", severity: "minor" },
    { label: "Major", severity: "major" },
    { label: "Unknown", severity: "unknown" },
  ];
  return (
    <ul className="flex flex-row flex-wrap gap-3 sm:flex-col sm:gap-2">
      {items.map((it) => (
        <li
          key={it.severity}
          className="flex items-center gap-2 text-xs text-zinc-400"
        >
          <span
            className={`inline-block h-3 w-3 rounded-sm border ${SEVERITY_RING_CLASSES[it.severity]}`}
          />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function GradeStat({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: number | null;
  highlight?: boolean;
}) {
  const display = value === null ? "—" : value.toFixed(1);
  return (
    <div
      className={`rounded-xl border p-4 ${
        highlight
          ? "border-lime-500/40 bg-lime-500/5"
          : "border-zinc-800"
      }`}
    >
      <p className="text-xs uppercase tracking-wide text-zinc-500">{label}</p>
      <p
        className={`mt-1 text-2xl font-semibold ${
          highlight ? "text-lime-400" : "text-zinc-100"
        }`}
      >
        {display}
      </p>
    </div>
  );
}

function AuthenticitySection({ cert }: { cert: Certificate }) {
  const auth = cert.authenticity;
  if (!auth) {
    return (
      <section className="rounded-xl border border-zinc-800 p-6 text-zinc-400">
        <h2 className="mb-2 text-base font-medium text-zinc-200">Authenticity</h2>
        <p className="text-sm">
          The counterfeit-detection pass did not complete for this submission.
        </p>
      </section>
    );
  }

  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-base font-medium text-zinc-200">Authenticity</h2>
        <span className="text-xs text-zinc-500">
          Confidence: {(auth.confidence * 100).toFixed(0)}%
        </span>
      </div>
      <VerdictBadge verdict={auth.verdict} />
      {auth.reasons.length > 0 && (
        <ul className="mt-4 space-y-1 text-sm text-zinc-400">
          {auth.reasons.map((r, i) => (
            <li key={i}>• {r}</li>
          ))}
        </ul>
      )}
      <div className="mt-5 flex flex-col gap-2">
        {auth.detectors.map((d) => (
          <DetectorRow key={d.detector} detector={d} />
        ))}
      </div>
    </section>
  );
}

const VERDICT_LABEL: Record<AuthenticityVerdict, string> = {
  authentic: "Authentic",
  suspicious: "Suspicious — manual review recommended",
  likely_counterfeit: "Likely counterfeit",
  unverified: "Unverified — insufficient signal",
};

const VERDICT_CLASSES: Record<AuthenticityVerdict, string> = {
  authentic: "border-lime-500/40 bg-lime-500/10 text-lime-300",
  suspicious: "border-amber-500/40 bg-amber-500/10 text-amber-300",
  likely_counterfeit: "border-red-500/40 bg-red-500/10 text-red-300",
  unverified: "border-zinc-700 bg-zinc-800/50 text-zinc-400",
};

function VerdictBadge({ verdict }: { verdict: AuthenticityVerdict }) {
  return (
    <div
      className={`rounded-xl border px-4 py-3 text-sm font-medium ${VERDICT_CLASSES[verdict]}`}
    >
      {VERDICT_LABEL[verdict]}
    </div>
  );
}

function DetectorRow({ detector }: { detector: DetectorScore }) {
  const pct = (detector.score * 100).toFixed(0);
  return (
    <div className="flex flex-col gap-1 rounded-xl border border-zinc-800 p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex items-center gap-3">
        <span className="font-mono text-xs uppercase tracking-wide text-zinc-500">
          {detector.detector}
        </span>
        <span className="text-sm text-zinc-200">
          score {pct}%
        </span>
        <span className="text-xs text-zinc-500">
          ({(detector.confidence * 100).toFixed(0)}% conf.)
        </span>
      </div>
      <span
        className={`text-xs font-medium ${VERDICT_CLASSES[detector.verdict]} rounded-full border px-2 py-1`}
      >
        {VERDICT_LABEL[detector.verdict]}
      </span>
    </div>
  );
}

function Footer({ cert }: { cert: Certificate }) {
  const versionSummary = Object.entries(cert.authenticity?.model_versions ?? {})
    .filter(([, v]) => typeof v === "string")
    .map(([k, v]) => `${k}=${v}`)
    .join(" • ");
  return (
    <footer className="border-t border-zinc-800 pt-6 text-xs text-zinc-500">
      <p>
        This certificate is generated by Project Grader’s automated grading +
        counterfeit-detection pipeline. Not affiliated with PSA, BGS, SGC, or
        CGC.
      </p>
      {versionSummary && (
        <p className="mt-2 font-mono text-zinc-600">{versionSummary}</p>
      )}
    </footer>
  );
}
