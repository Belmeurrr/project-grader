/**
 * Dynamic OG image for `/cert/[id]`.
 *
 * Next.js's file-based metadata image convention: this module's default
 * export is rendered into a 1200x630 PNG and automatically wired into
 * the page's `<meta property="og:image">` and `<meta name="twitter:image">`
 * tags. No `generateMetadata` plumbing required for the URL itself —
 * Next.js generates `/cert/[id]/opengraph-image` and includes it.
 *
 * This is the differentiator vs TAG Grading's SPA: their cert page is
 * a client-side single-page app, so unfurls on Twitter/Discord show a
 * generic site logo. Ours renders the actual grade + card name + cert
 * ID into the share card. A buyer scrolling Twitter sees the grade
 * before they ever click.
 *
 * Layout: a 1200x630 dark canvas with a left-edge verdict color stripe
 * (lime / amber / red / zinc — same palette as the cert page badges),
 * the identified card name + cert ID + scheme/grade scheme as text,
 * and the final grade in a large lime-tinted badge on the right.
 */
import { ImageResponse } from "next/og";

import { fetchCertificate, type AuthenticityVerdict } from "@/lib/cert";

export const runtime = "nodejs";

export const alt = "Project Grader certificate";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const STRIPE_COLOR: Record<AuthenticityVerdict, string> = {
  authentic: "#84cc16", // lime-500
  suspicious: "#f59e0b", // amber-500
  likely_counterfeit: "#ef4444", // red-500
  unverified: "#52525b", // zinc-600
};

/**
 * Build the data shape passed into <ImageResponse>. Exported so the
 * unit test can assert the layout decisions (final-grade text, verdict
 * stripe color) without rendering an actual PNG inside vitest.
 */
export type CertOgData = {
  certIdShort: string;
  cardName: string;
  finalGrade: string;
  scheme: string;
  stripeColor: string;
  verdictLabel: string;
};

const VERDICT_LABEL: Record<AuthenticityVerdict, string> = {
  authentic: "AUTHENTIC",
  suspicious: "SUSPICIOUS",
  likely_counterfeit: "LIKELY COUNTERFEIT",
  unverified: "UNVERIFIED",
};

export function buildCertOgData(
  cert: Awaited<ReturnType<typeof fetchCertificate>>,
  fallbackId: string,
): CertOgData {
  if (!cert) {
    return {
      certIdShort: `${fallbackId.slice(0, 8)}…`,
      cardName: "Cert not available",
      finalGrade: "—",
      scheme: "PSA",
      stripeColor: STRIPE_COLOR.unverified,
      verdictLabel: VERDICT_LABEL.unverified,
    };
  }
  const primary = cert.grades.find((g) => g.scheme === "psa") ?? cert.grades[0];
  const finalGrade = primary?.final !== null && primary?.final !== undefined
    ? primary.final.toFixed(1)
    : "—";
  const verdict = cert.authenticity?.verdict ?? "unverified";
  return {
    certIdShort: `${cert.cert_id.slice(0, 12)}…`,
    cardName: cert.identified_card?.name ?? "Unidentified card",
    finalGrade,
    scheme: (primary?.scheme ?? "psa").toUpperCase(),
    stripeColor: STRIPE_COLOR[verdict],
    verdictLabel: VERDICT_LABEL[verdict],
  };
}

export default async function Image({ params }: { params: { id: string } }) {
  const cert = await fetchCertificate(params.id).catch(() => null);
  const data = buildCertOgData(cert, params.id);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          background: "#09090b", // zinc-950
          color: "#fafafa",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        {/* Verdict color stripe */}
        <div
          style={{
            width: 24,
            height: "100%",
            background: data.stripeColor,
          }}
        />
        {/* Main content */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            justifyContent: "space-between",
            padding: "64px 72px",
          }}
        >
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div
              style={{
                fontSize: 22,
                letterSpacing: 6,
                color: "#84cc16",
                textTransform: "uppercase",
              }}
            >
              Project Grader • Certificate
            </div>
            <div
              style={{
                fontSize: 64,
                fontWeight: 700,
                lineHeight: 1.05,
                color: "#fafafa",
                maxWidth: 900,
              }}
            >
              {data.cardName}
            </div>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "flex-end",
              justifyContent: "space-between",
              gap: 32,
            }}
          >
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <div
                style={{
                  fontSize: 18,
                  textTransform: "uppercase",
                  letterSpacing: 4,
                  color: "#a1a1aa",
                }}
              >
                Cert ID
              </div>
              <div
                style={{
                  fontSize: 32,
                  fontFamily: "monospace",
                  color: "#e4e4e7",
                }}
              >
                {data.certIdShort}
              </div>
              <div
                style={{
                  marginTop: 12,
                  fontSize: 18,
                  color: data.stripeColor,
                  letterSpacing: 3,
                }}
              >
                {data.verdictLabel}
              </div>
            </div>
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                width: 280,
                height: 220,
                borderRadius: 24,
                border: "2px solid rgba(132, 204, 22, 0.4)",
                background: "rgba(132, 204, 22, 0.08)",
              }}
            >
              <div
                style={{
                  fontSize: 18,
                  letterSpacing: 4,
                  color: "#a1a1aa",
                  textTransform: "uppercase",
                }}
              >
                {data.scheme} Final
              </div>
              <div
                style={{
                  fontSize: 144,
                  fontWeight: 800,
                  lineHeight: 1,
                  color: "#a3e635",
                }}
              >
                {data.finalGrade}
              </div>
            </div>
          </div>
        </div>
      </div>
    ),
    { ...size },
  );
}
