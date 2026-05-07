/**
 * Public cert page — `/cert/[id]`. Workbench design.
 *
 * Server component that fetches the cert via the public API endpoint
 * and renders Grade + AuthenticityResult + region heatmap. ISR-cached
 * with a 5-minute revalidate window matching the API's Cache-Control.
 *
 * Visual port of docs/design/cert.jsx (Split layout). The mock CERT
 * payload in the design source is shaped against
 * apps/api/grader/schemas/submissions.py::CertificatePublic, so most
 * fields map 1:1; per-detector forensic numbers come from the API's
 * `metadata: Record<string, unknown>` bag rather than the design's
 * inlined typed fields.
 *
 * Interactive bits (copy-toast, QR, mode tabs, detector expand) live
 * in client islands under apps/web/components/cert/.
 */

import { notFound } from "next/navigation";

import {
  type AuthenticityVerdict,
  type CertAuthenticity,
  type Certificate,
  type Grade,
  type IdentifiedCard,
  type PopulationStat,
  type Region,
  type RegionSeverity,
  fetchCertificate,
} from "@/lib/cert";
import AuthenticityDetectors from "@/components/cert/AuthenticityDetectors";
import CardVisionPanel from "@/components/cert/CardVisionPanel";
import CertHeaderActions, {
  CertIdPill,
} from "@/components/cert/CertHeaderActions";
import { resolvePublicBaseUrl } from "@/lib/publicUrl";

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

  const baseUrl = resolvePublicBaseUrl();
  const certUrl = `${baseUrl}/cert/${id}`;
  const printHref = `/cert/${id}/print`;
  const ogImageHref = `/cert/${id}/opengraph-image`;
  const primary: Grade | undefined =
    cert.grades.find((g) => g.scheme === "psa") ?? cert.grades[0];
  const cardName = cert.identified_card?.name ?? "Unidentified card";
  const finalLabel =
    primary && primary.final !== null ? primary.final.toFixed(1) : "—";

  return (
    <main
      style={{
        maxWidth: 1180,
        margin: "32px auto",
        padding: "0 32px 80px",
        display: "flex",
        flexDirection: "column",
        gap: 28,
      }}
    >
      <CertHeader
        cert={cert}
        certUrl={certUrl}
        cardName={cardName}
        finalGradeLabel={finalLabel}
        printHref={printHref}
        ogImageHref={ogImageHref}
      />

      {/* Top band: Card Vision (left, sticky) + Grade Hero / Subgrades / Pop (right) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0, 460px) minmax(0, 1fr)",
          gap: 36,
          marginTop: 4,
        }}
        className="cert-top-band"
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 22,
            position: "sticky",
            top: 80,
            alignSelf: "start",
          }}
        >
          <CardVisionPanel
            images={cert.images}
            regions={cert.regions}
            cardName={cardName}
          />
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 24,
            minWidth: 0,
          }}
        >
          {primary ? (
            <>
              <GradeHero grade={primary} />
              <SubgradeGrid grade={primary} />
            </>
          ) : (
            <NoGradeNotice />
          )}
          {cert.population && <PopulationPanel pop={cert.population} />}
        </div>
      </div>

      {/* Below the band: centering detail + flagged defects */}
      {primary && (
        <div
          className="cert-twocol"
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 36,
          }}
        >
          <CenteringDetail grade={primary} />
          <DefectsPanel regions={cert.regions} />
        </div>
      )}

      <AuthenticityPanel auth={cert.authenticity} />

      <Provenance cert={cert} />

      {/* Responsive collapse for narrow viewports — keeps the design
          desktop-first but folds the two grids to single column when
          the viewport is too narrow to render both side-by-side. */}
      <style>{`
        @media (max-width: 880px) {
          .cert-top-band { grid-template-columns: 1fr !important; }
          .cert-twocol { grid-template-columns: 1fr !important; }
          .cert-top-band > div:first-child { position: static !important; }
        }
      `}</style>
    </main>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Page metadata — preserved from the previous implementation
export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}) {
  const { id } = await params;
  const cert = await fetchCertificate(id).catch(() => null);
  const baseUrl = resolvePublicBaseUrl();
  const url = `${baseUrl}/cert/${id}`;

  const cardName = cert?.identified_card?.name ?? "Card certificate";
  const primary =
    cert?.grades.find((g) => g.scheme === "psa") ?? cert?.grades[0];
  const finalLabel =
    primary && primary.final !== null ? primary.final.toFixed(1) : null;
  const verdict = cert?.authenticity?.verdict ?? null;

  const titleSuffix = finalLabel ? ` — Graded ${finalLabel}` : "";
  const title = `${cardName}${titleSuffix} • Project Grader`;
  const description = finalLabel
    ? `${cardName} graded ${finalLabel}${
        verdict ? ` (${verdict.replace(/_/g, " ")})` : ""
      }. View the full evidence trail and authenticity report.`
    : `${cardName} — AI-graded card with full evidence trail.`;

  return {
    title,
    description,
    metadataBase: new URL(baseUrl),
    openGraph: {
      type: "website" as const,
      url,
      title,
      description,
      siteName: "Project Grader",
      images: [
        {
          url: `/cert/${id}/opengraph-image`,
          width: 1200,
          height: 630,
          alt: `Cert ${id.slice(0, 8)} — ${cardName}`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image" as const,
      title,
      description,
      images: [`/cert/${id}/opengraph-image`],
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// CERT HEADER
function CertHeader({
  cert,
  certUrl,
  cardName,
  finalGradeLabel,
  printHref,
  ogImageHref,
}: {
  cert: Certificate;
  certUrl: string;
  cardName: string;
  finalGradeLabel: string;
  printHref: string;
  ogImageHref: string;
}) {
  const completed = new Date(cert.completed_at);
  const card = cert.identified_card;
  const verdict = cert.authenticity?.verdict ?? "unverified";

  return (
    <header
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 16,
        paddingBottom: 24,
        borderBottom: "1px solid var(--line)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          gap: 24,
          flexWrap: "wrap",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 8,
            flex: 1,
            minWidth: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--lime)",
                letterSpacing: "0.22em",
                textTransform: "uppercase",
              }}
            >
              Project Grader · Certificate
            </span>
            <VerdictPill verdict={verdict} small />
          </div>
          <h1
            style={{
              fontFamily: "var(--font-instrument-serif), serif",
              fontSize: 44,
              fontWeight: 400,
              color: "var(--ink-0)",
              margin: 0,
              letterSpacing: "-0.01em",
              lineHeight: 1.05,
              textWrap: "balance",
            }}
          >
            {cardName}
          </h1>
          {card && <CardMetaLine card={card} />}
        </div>
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            gap: 10,
          }}
        >
          <CertIdPill certId={cert.cert_id} />
          <span
            className="mono"
            style={{
              fontSize: 10,
              color: "var(--ink-3)",
              letterSpacing: "0.04em",
            }}
          >
            graded{" "}
            {completed.toLocaleDateString("en-US", {
              year: "numeric",
              month: "short",
              day: "2-digit",
            })}{" "}
            ·{" "}
            {completed.toLocaleTimeString("en-US", {
              hour: "2-digit",
              minute: "2-digit",
              hour12: false,
            })}{" "}
            UTC
          </span>
        </div>
      </div>

      <CertHeaderActions
        certId={cert.cert_id}
        certUrl={certUrl}
        cardName={cardName}
        finalGradeLabel={finalGradeLabel}
        printHref={printHref}
        ogImageHref={ogImageHref}
      />
    </header>
  );
}

function CardMetaLine({ card }: { card: IdentifiedCard }) {
  const segments: string[] = [];
  if (card.set_code) segments.push(card.set_code);
  if (card.card_number) segments.push(`#${card.card_number}`);
  return (
    <div
      className="mono"
      style={{
        fontSize: 11,
        color: "var(--ink-2)",
        letterSpacing: "0.04em",
        display: "flex",
        gap: 14,
        flexWrap: "wrap",
      }}
    >
      {segments.map((s, i) => (
        <span key={i} style={{ display: "flex", gap: 14, alignItems: "center" }}>
          {i > 0 && <span style={{ color: "var(--ink-5)" }}>·</span>}
          <span>{s}</span>
        </span>
      ))}
      {segments.length > 0 && <span style={{ color: "var(--ink-5)" }}>·</span>}
      <span style={{ color: "var(--ink-3)" }}>
        identification {(card.confidence * 100).toFixed(1)}%
      </span>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GRADE HERO
function GradeHero({ grade }: { grade: Grade }) {
  const final = grade.final;
  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 16 }}>
      <SectionLabel hint={`scheme · ${grade.scheme}`}>Grade</SectionLabel>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "auto minmax(0, 1fr)",
          gap: 44,
          alignItems: "center",
          padding: "28px 32px",
          background:
            "linear-gradient(135deg, rgba(190,242,100,0.04) 0%, transparent 60%), var(--bg-1)",
          border: "1px solid var(--line)",
          borderRadius: 8,
          position: "relative",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            gap: 4,
            flexShrink: 0,
          }}
        >
          <span
            style={{
              fontFamily: "var(--font-instrument-serif), serif",
              fontSize: 128,
              lineHeight: 0.9,
              color: "var(--ink-0)",
              letterSpacing: "-0.04em",
              textShadow: "0 0 80px rgba(190,242,100,0.15)",
            }}
          >
            {final !== null ? final.toFixed(1) : "—"}
          </span>
          <span
            className="mono"
            style={{
              fontSize: 11,
              color: "var(--ink-3)",
              letterSpacing: "0.16em",
              textTransform: "uppercase",
            }}
          >
            of 10.0 · {grade.scheme}
          </span>
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 18,
            minWidth: 0,
          }}
        >
          {final === null ? (
            <PreliminaryBanner />
          ) : (
            <ConfidenceBand finalGrade={final} confidence={grade.confidence} />
          )}
        </div>

        <svg
          aria-hidden
          style={{
            position: "absolute",
            inset: 0,
            width: "100%",
            height: "100%",
            opacity: 0.05,
            pointerEvents: "none",
          }}
        >
          <defs>
            <pattern
              id="grade-grid"
              width={20}
              height={20}
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 20 0 L 0 0 0 20"
                fill="none"
                stroke="var(--lime)"
                strokeWidth="0.5"
              />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grade-grid)" />
        </svg>
      </div>
    </section>
  );
}

function PreliminaryBanner() {
  return (
    <div
      style={{
        padding: "10px 14px",
        background: "rgba(251,191,36,0.06)",
        border: "1px solid rgba(251,191,36,0.4)",
        borderRadius: 6,
        color: "var(--amber)",
        fontSize: 12,
        lineHeight: 1.5,
      }}
    >
      <strong>Preliminary subgrades.</strong>{" "}
      <span style={{ color: "var(--ink-2)" }}>
        Final grade unavailable until the corners + surface trainers ship.
        Centering and edges are the canonical scores.
      </span>
    </div>
  );
}

function ConfidenceBand({
  finalGrade,
  confidence,
}: {
  finalGrade: number;
  confidence: number;
}) {
  const width = (1 - confidence) * 2;
  const lower = Math.max(1, Math.min(10, finalGrade - width));
  const upper = Math.max(1, Math.min(10, finalGrade + width));
  const pct = (confidence * 100).toFixed(0);

  const minPx = ((lower - 1) / 9) * 100;
  const maxPx = ((upper - 1) / 9) * 100;
  const finalPx = ((finalGrade - 1) / 9) * 100;

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: 8,
        minWidth: 0,
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          gap: 12,
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-3)",
            letterSpacing: "0.16em",
            textTransform: "uppercase",
          }}
        >
          Confidence band · {pct}%
        </span>
        <span className="mono" style={{ fontSize: 11, color: "var(--ink-1)" }}>
          likely{" "}
          <span style={{ color: "var(--ink-0)", fontWeight: 600 }}>
            {lower.toFixed(1)}–{upper.toFixed(1)}
          </span>
        </span>
      </div>
      <div
        style={{
          height: 10,
          background: "var(--bg-3)",
          borderRadius: 2,
          position: "relative",
          overflow: "visible",
        }}
      >
        {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
          <div
            key={n}
            style={{
              position: "absolute",
              top: -2,
              bottom: -2,
              left: `${((n - 1) / 9) * 100}%`,
              width: 1,
              background:
                n === Math.floor(finalGrade) ? "var(--ink-3)" : "var(--line)",
            }}
          />
        ))}
        <div
          style={{
            position: "absolute",
            top: 0,
            bottom: 0,
            left: `${minPx}%`,
            width: `${maxPx - minPx}%`,
            background: "rgba(190,242,100,0.2)",
            borderTop: "1px solid var(--lime)",
            borderBottom: "1px solid var(--lime)",
            borderRadius: 1,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: -3,
            bottom: -3,
            left: `calc(${finalPx}% - 1px)`,
            width: 2,
            background: "var(--lime)",
            boxShadow: "0 0 8px var(--lime)",
          }}
        />
      </div>
      <div style={{ display: "flex", justifyContent: "space-between" }}>
        <span className="mono" style={{ fontSize: 9, color: "var(--ink-4)" }}>
          1.0
        </span>
        <span className="mono" style={{ fontSize: 9, color: "var(--ink-4)" }}>
          5.0
        </span>
        <span className="mono" style={{ fontSize: 9, color: "var(--ink-4)" }}>
          10.0
        </span>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// SUBGRADES
function SubgradeGrid({ grade }: { grade: Grade }) {
  const subs: Array<{ k: string; l: string; v: number | null; hint: string }> = [
    { k: "centering", l: "Centering", v: grade.centering, hint: "geometric · 99% conf" },
    { k: "corners", l: "Corners", v: grade.corners, hint: "EfficientNet-V2" },
    { k: "edges", l: "Edges", v: grade.edges, hint: "MAD anomaly" },
    { k: "surface", l: "Surface", v: grade.surface, hint: "paired-flash + rosette" },
  ];
  return (
    <section>
      <SectionLabel hint="per-criterion · independent">Subgrades</SectionLabel>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(4, 1fr)",
          gap: 10,
          marginTop: 12,
        }}
      >
        {subs.map((s) => (
          <div
            key={s.k}
            style={{
              background: "var(--bg-1)",
              border: "1px solid var(--line)",
              borderRadius: 6,
              padding: "16px 18px",
              display: "flex",
              flexDirection: "column",
              gap: 4,
            }}
          >
            <span
              className="mono"
              style={{
                fontSize: 9,
                color: "var(--ink-3)",
                letterSpacing: "0.18em",
                textTransform: "uppercase",
              }}
            >
              {s.l}
            </span>
            <span
              style={{
                fontFamily: "var(--font-instrument-serif), serif",
                fontSize: 40,
                lineHeight: 1,
                color: s.v !== null ? "var(--ink-0)" : "var(--ink-4)",
                letterSpacing: "-0.02em",
              }}
            >
              {s.v !== null ? s.v.toFixed(1) : "—"}
            </span>
            <span
              className="mono"
              style={{
                fontSize: 9,
                color: "var(--ink-4)",
                letterSpacing: "0.04em",
              }}
            >
              {s.hint}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// CENTERING
function CenteringDetail({ grade }: { grade: Grade }) {
  const front =
    grade.front_centering_lr !== null && grade.front_centering_tb !== null
      ? { lr: grade.front_centering_lr, tb: grade.front_centering_tb }
      : null;
  const back =
    grade.back_centering_lr !== null && grade.back_centering_tb !== null
      ? { lr: grade.back_centering_lr, tb: grade.back_centering_tb }
      : null;

  return (
    <section>
      <SectionLabel hint="L/R · T/B · per face">Centering breakdown</SectionLabel>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 10,
          marginTop: 12,
        }}
      >
        <CenteringCard label="Front" data={front} />
        <CenteringCard label="Back" data={back} />
      </div>
      <p
        className="mono"
        style={{
          fontSize: 10,
          color: "var(--ink-4)",
          margin: "10px 0 0",
          letterSpacing: "0.02em",
          lineHeight: 1.5,
        }}
      >
        Worst face dominates the centering subgrade. Perfect = 50/50; PSA 10
        tolerance is ±5%.
      </p>
    </section>
  );
}

function CenteringCard({
  label,
  data,
}: {
  label: string;
  data: { lr: number; tb: number } | null;
}) {
  return (
    <div
      style={{
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 6,
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
          alignItems: "center",
        }}
      >
        <span
          className="mono"
          style={{
            fontSize: 10,
            color: "var(--ink-2)",
            letterSpacing: "0.16em",
            textTransform: "uppercase",
          }}
        >
          {label}
        </span>
        <span className="mono" style={{ fontSize: 10, color: "var(--ink-3)" }}>
          {data ? (
            <>
              L/R{" "}
              <span style={{ color: "var(--ink-1)" }}>{data.lr.toFixed(1)}</span>{" "}
              · T/B{" "}
              <span style={{ color: "var(--ink-1)" }}>{data.tb.toFixed(1)}</span>
            </>
          ) : (
            <span style={{ color: "var(--ink-5)" }}>not captured</span>
          )}
        </span>
      </div>
      <CenteringSchematic data={data} />
    </div>
  );
}

function CenteringSchematic({ data }: { data: { lr: number; tb: number } | null }) {
  const lrLeft = data?.lr ?? 50;
  const tbTop = data?.tb ?? 50;
  return (
    <div
      style={{
        aspectRatio: "5/7",
        background: "var(--bg-2)",
        border: "1px solid var(--line-2)",
        borderRadius: 4,
        position: "relative",
        padding: "10%",
        maxHeight: 140,
        alignSelf: "center",
        width: 100,
      }}
    >
      <div style={{ position: "absolute", inset: 0 }}>
        <div
          style={{
            position: "absolute",
            left: `${(20 * lrLeft) / 100}%`,
            right: `${(20 * (100 - lrLeft)) / 100}%`,
            top: `${(20 * tbTop) / 100}%`,
            bottom: `${(20 * (100 - tbTop)) / 100}%`,
            background: "rgba(190,242,100,0.1)",
            border: "1px solid var(--lime)",
            borderRadius: 2,
            opacity: data ? 1 : 0.3,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: "50%",
            left: "20%",
            right: "20%",
            height: 1,
            background: "var(--ink-5)",
            opacity: 0.4,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: "50%",
            top: "20%",
            bottom: "20%",
            width: 1,
            background: "var(--ink-5)",
            opacity: 0.4,
          }}
        />
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DEFECTS — itemized DINGS-style
const SEVERITY_DOT: Record<RegionSeverity, string> = {
  ok: "var(--lime)",
  minor: "var(--amber)",
  major: "var(--rose)",
  unknown: "var(--ink-5)",
};

const POSITION_LABEL: Record<string, string> = {
  top_left: "Top-left",
  top_right: "Top-right",
  bottom_left: "Bottom-left",
  bottom_right: "Bottom-right",
  top: "Top edge",
  bottom: "Bottom edge",
  left: "Left edge",
  right: "Right edge",
  whole_card: "",
};

function DefectsPanel({ regions }: { regions: Region[] }) {
  const flagged = regions.filter((r) => r.reasons.length > 0);
  const groups: Record<"centering" | "corner" | "edge" | "surface", Region[]> = {
    centering: [],
    corner: [],
    edge: [],
    surface: [],
  };
  flagged.forEach((r) => {
    if (r.kind in groups) {
      groups[r.kind as keyof typeof groups].push(r);
    }
  });
  const kinds: Array<{ k: keyof typeof groups; l: string }> = [
    { k: "centering", l: "Centering" },
    { k: "corner", l: "Corners" },
    { k: "edge", l: "Edges" },
    { k: "surface", l: "Surface" },
  ];

  return (
    <section>
      <SectionLabel hint="DINGS · plain-english rationale">Defects flagged</SectionLabel>
      {flagged.length === 0 ? (
        <p
          style={{
            fontSize: 13,
            color: "var(--ink-3)",
            margin: "12px 0 0",
            fontStyle: "italic",
          }}
        >
          No defects flagged. The card grades clean across all evaluated regions.
        </p>
      ) : (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 16,
            marginTop: 14,
          }}
        >
          {kinds.map(({ k, l }) => {
            if (groups[k].length === 0) return null;
            return (
              <div key={k}>
                <h4
                  className="mono"
                  style={{
                    fontSize: 10,
                    color: "var(--ink-3)",
                    letterSpacing: "0.18em",
                    textTransform: "uppercase",
                    margin: "0 0 8px",
                    fontWeight: 500,
                  }}
                >
                  {l}
                </h4>
                <ul
                  style={{
                    listStyle: "none",
                    margin: 0,
                    padding: 0,
                    display: "flex",
                    flexDirection: "column",
                    gap: 7,
                  }}
                >
                  {groups[k].flatMap((r) =>
                    r.reasons.map((reason, i) => (
                      <li
                        key={`${r.position}-${i}`}
                        style={{
                          display: "flex",
                          gap: 10,
                          alignItems: "baseline",
                          padding: "8px 12px",
                          background: "var(--bg-1)",
                          borderLeft: `2px solid ${SEVERITY_DOT[r.severity]}`,
                          borderRadius: "0 4px 4px 0",
                        }}
                      >
                        <span
                          style={{
                            width: 6,
                            height: 6,
                            borderRadius: 3,
                            background: SEVERITY_DOT[r.severity],
                            flexShrink: 0,
                            transform: "translateY(-1px)",
                          }}
                        />
                        <div
                          style={{
                            display: "flex",
                            flexDirection: "column",
                            gap: 2,
                          }}
                        >
                          <span style={{ fontSize: 13, color: "var(--ink-1)" }}>
                            {reason}
                          </span>
                          <span
                            className="mono"
                            style={{
                              fontSize: 10,
                              color: "var(--ink-4)",
                              letterSpacing: "0.04em",
                            }}
                          >
                            {POSITION_LABEL[r.position] || r.position} ·
                            severity {r.severity} · score{" "}
                            {r.score !== null ? r.score.toFixed(2) : "–"}
                          </span>
                        </div>
                      </li>
                    )),
                  )}
                </ul>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTHENTICITY
const VERDICT_META: Record<
  AuthenticityVerdict,
  { label: string; tone: "lime" | "amber" | "rose" | "ink"; dot: string }
> = {
  authentic: { label: "Authentic", tone: "lime", dot: "var(--lime)" },
  suspicious: {
    label: "Suspicious — manual review",
    tone: "amber",
    dot: "var(--amber)",
  },
  likely_counterfeit: {
    label: "Likely counterfeit",
    tone: "rose",
    dot: "var(--rose)",
  },
  unverified: {
    label: "Unverified — insufficient signal",
    tone: "ink",
    dot: "var(--ink-4)",
  },
};

const VERDICT_COLORS: Record<
  "lime" | "amber" | "rose" | "ink",
  { bg: string; border: string; fg: string }
> = {
  lime: {
    bg: "rgba(190,242,100,0.06)",
    border: "rgba(190,242,100,0.4)",
    fg: "var(--lime)",
  },
  amber: {
    bg: "rgba(251,191,36,0.06)",
    border: "rgba(251,191,36,0.4)",
    fg: "var(--amber)",
  },
  rose: {
    bg: "rgba(251,113,133,0.06)",
    border: "rgba(251,113,133,0.4)",
    fg: "var(--rose)",
  },
  ink: {
    bg: "var(--bg-2)",
    border: "var(--line)",
    fg: "var(--ink-3)",
  },
};

function VerdictPill({
  verdict,
  small = false,
}: {
  verdict: AuthenticityVerdict;
  small?: boolean;
}) {
  const meta = VERDICT_META[verdict];
  const c = VERDICT_COLORS[meta.tone];
  const animated = verdict === "suspicious" || verdict === "likely_counterfeit";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: small ? 5 : 8,
        padding: small ? "3px 8px" : "7px 14px",
        background: c.bg,
        border: `1px solid ${c.border}`,
        borderRadius: 99,
        color: c.fg,
        fontWeight: 500,
        fontSize: small ? 10 : 13,
      }}
    >
      <span
        style={{
          width: small ? 4 : 6,
          height: small ? 4 : 6,
          borderRadius: 4,
          background: meta.dot,
          boxShadow: `0 0 6px ${meta.dot}`,
          animation: animated ? "pulse 1.6s ease-in-out infinite" : "none",
        }}
      />
      {meta.label}
    </span>
  );
}

function AuthenticityPanel({ auth }: { auth: CertAuthenticity | null }) {
  if (!auth) {
    return (
      <section>
        <SectionLabel>Authenticity</SectionLabel>
        <div
          style={{
            marginTop: 12,
            padding: "18px 20px",
            background: "var(--bg-1)",
            border: "1px solid var(--line)",
            borderRadius: 8,
            color: "var(--ink-3)",
            fontSize: 13,
          }}
        >
          The counterfeit-detection pass did not complete for this submission.
        </div>
      </section>
    );
  }

  return (
    <section>
      <SectionLabel hint={`ensemble · ${auth.detectors.length} detectors`}>
        Authenticity
      </SectionLabel>
      <div
        style={{
          marginTop: 12,
          padding: "18px 20px",
          background: "var(--bg-1)",
          border: "1px solid var(--line)",
          borderRadius: 8,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "flex-start",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          <VerdictPill verdict={auth.verdict} />
          <div style={{ textAlign: "right" }}>
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--ink-3)",
                letterSpacing: "0.16em",
                textTransform: "uppercase",
              }}
            >
              ensemble confidence
            </span>
            <div
              style={{
                fontFamily: "var(--font-instrument-serif), serif",
                fontSize: 28,
                color: "var(--ink-0)",
                lineHeight: 1,
                marginTop: 4,
              }}
            >
              {(auth.confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {auth.reasons.length > 0 && (
          <ul
            style={{
              listStyle: "none",
              margin: 0,
              padding: 0,
              display: "flex",
              flexDirection: "column",
              gap: 5,
            }}
          >
            {auth.reasons.map((r, i) => (
              <li
                key={i}
                style={{
                  fontSize: 12,
                  color: "var(--ink-2)",
                  display: "flex",
                  gap: 8,
                  lineHeight: 1.5,
                }}
              >
                <span style={{ color: "var(--lime)", flexShrink: 0 }}>✓</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        )}

        {auth.detectors.length > 0 && (
          <>
            <div style={{ height: 1, background: "var(--line-2)" }} />
            <AuthenticityDetectors detectors={auth.detectors} />
          </>
        )}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// POPULATION
function PopulationPanel({ pop }: { pop: PopulationStat }) {
  const stats = [
    { l: "Population", v: pop.total_graded.toString(), sub: "graded by Project Grader" },
    {
      l: "Rank",
      v: `#${pop.this_rank}`,
      sub: pop.max_grade !== null ? `top score ${pop.max_grade.toFixed(1)}` : "highest scoring",
    },
    {
      l: "Order",
      v: `#${pop.chronological_index}`,
      sub: "graded chronologically",
    },
  ];
  return (
    <section>
      <SectionLabel hint="among identified variants">Population</SectionLabel>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 6,
          marginTop: 12,
        }}
      >
        {stats.map((s) => (
          <div
            key={s.l}
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
              {s.l}
            </div>
            <div
              style={{
                fontFamily: "var(--font-instrument-serif), serif",
                fontSize: 28,
                color: "var(--ink-0)",
                lineHeight: 1,
                marginTop: 4,
                letterSpacing: "-0.02em",
              }}
            >
              {s.v}
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
              {s.sub}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
function NoGradeNotice() {
  return (
    <section
      style={{
        padding: "20px 24px",
        background: "var(--bg-1)",
        border: "1px solid var(--line)",
        borderRadius: 8,
        color: "var(--ink-3)",
        fontSize: 13,
      }}
    >
      No grade computed for this submission.
    </section>
  );
}

function Provenance({ cert }: { cert: Certificate }) {
  const versions = cert.authenticity?.model_versions ?? {};
  const entries = Object.entries(versions).filter(
    ([, v]) => typeof v === "string" || typeof v === "number",
  );
  return (
    <footer
      style={{
        marginTop: 40,
        paddingTop: 24,
        borderTop: "1px solid var(--line-2)",
        display: "flex",
        flexDirection: "column",
        gap: 16,
      }}
    >
      {entries.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
            gap: 6,
          }}
        >
          {entries.map(([k, v]) => (
            <div
              key={k}
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--ink-3)",
                letterSpacing: "0.02em",
                display: "flex",
                justifyContent: "space-between",
                padding: "5px 8px",
                background: "var(--bg-1)",
                borderRadius: 3,
              }}
            >
              <span style={{ color: "var(--ink-4)" }}>{k}</span>
              <span>{String(v)}</span>
            </div>
          ))}
        </div>
      )}
      <p
        style={{
          fontSize: 11,
          color: "var(--ink-4)",
          margin: 0,
          lineHeight: 1.6,
          maxWidth: 720,
        }}
      >
        This certificate is generated by Project Grader&apos;s automated grading
        + counterfeit-detection pipeline. Not affiliated with PSA, BGS, SGC, or
        CGC. Grades are independent assessments and may differ from third-party
        graders. Disagreements are welcome — request a manual review above.
      </p>
    </footer>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
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
