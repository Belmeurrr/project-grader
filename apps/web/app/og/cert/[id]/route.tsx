/**
 * Fallback OG image route — `/og/cert/[id]`.
 *
 * Next.js's file-based metadata image convention (
 * `apps/web/app/cert/[id]/opengraph-image.tsx`) is the canonical path,
 * but some Vercel deployments / self-hosted setups need an explicit
 * route handler with explicit `Content-Type: image/png` headers. This
 * route mirrors the same render so we have a known-good fallback URL
 * we can paste into Twitter's card validator if the metadata route
 * 404s.
 *
 * Implementation reuses `buildCertOgData` from the metadata image
 * file so the layout stays in lockstep — only the framework wiring
 * differs (route handler returning `ImageResponse` vs file-based
 * metadata default export).
 */
import { ImageResponse } from "next/og";

import { fetchCertificate } from "@/lib/cert";

import { buildCertOgData } from "@/app/cert/[id]/opengraph-image";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const { id } = await params;
  const cert = await fetchCertificate(id).catch(() => null);
  const data = buildCertOgData(cert, id);

  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          background: "#09090b",
          color: "#fafafa",
          fontFamily: "system-ui, sans-serif",
        }}
      >
        <div
          style={{
            width: 24,
            height: "100%",
            background: data.stripeColor,
          }}
        />
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
    {
      width: 1200,
      height: 630,
      headers: {
        "content-type": "image/png",
        "cache-control": "public, max-age=300, s-maxage=300",
      },
    },
  );
}
