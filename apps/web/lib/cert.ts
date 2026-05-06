/**
 * Typed client + types for the public GET /cert/:id endpoint.
 *
 * Lives outside the page file so the schema can be reused by future
 * surfaces (mobile app, marketplace listing previews, etc.) without
 * duplicating the response shape.
 *
 * Source of truth for the wire shape is
 * `apps/api/grader/schemas/submissions.py::CertificatePublic`. Keep
 * this in sync — either by hand for now, or via openapi-typescript
 * codegen once we add it.
 */

export type GradingScheme = "psa" | "bgs" | "trugrade";

export type AuthenticityVerdict =
  | "authentic"
  | "suspicious"
  | "likely_counterfeit"
  | "unverified";

export type Grade = {
  scheme: GradingScheme;
  // All subgrades + final are nullable while the corners/edges/surface
  // trainers are skeletons — `compute_psa_final` returns None whenever
  // an input is missing. The cert UI renders missing values as "—" and
  // shows a banner above the grades grid when `final === null`.
  centering: number | null;
  corners: number | null;
  edges: number | null;
  surface: number | null;
  final: number | null;
  confidence: number;
  front_centering_lr: number | null;
  front_centering_tb: number | null;
  back_centering_lr: number | null;
  back_centering_tb: number | null;
};

export type DetectorScore = {
  detector: string;
  score: number;
  verdict: AuthenticityVerdict;
  confidence: number;
  /** Detector-specific forensic data (peak_strength, p95_chroma, etc.) */
  metadata: Record<string, unknown>;
};

export type CertAuthenticity = {
  verdict: AuthenticityVerdict;
  confidence: number;
  reasons: string[];
  model_versions: Record<string, unknown>;
  detectors: DetectorScore[];
};

export type IdentifiedCard = {
  variant_id: string;
  name: string;
  set_code: string;
  card_number: string;
  confidence: number;
};

/**
 * Damage-heatmap region cell. Mirrors `RegionScore` in
 * `apps/api/grader/schemas/submissions.py`. The cert page renders one
 * absolutely-positioned div per entry — corners as small circles in
 * the four corners, edges as long thin rectangles along each side,
 * centering as a centered frame outline, and surface as a centered
 * "analysis pending" badge. Severity (ok / minor / major / unknown)
 * drives the color, not the score directly — keeps the overlay legible
 * for users who don't know what 0.92 means.
 */
export type RegionKind = "centering" | "corner" | "edge" | "surface";
export type RegionPosition =
  | "top_left"
  | "top_right"
  | "bottom_left"
  | "bottom_right"
  | "top"
  | "right"
  | "bottom"
  | "left"
  | "whole_card";
export type RegionSeverity = "ok" | "minor" | "major" | "unknown";

export type Region = {
  kind: RegionKind;
  position: RegionPosition;
  score: number | null;
  severity: RegionSeverity;
};

export type Certificate = {
  cert_id: string;
  completed_at: string; // ISO8601
  identified_card: IdentifiedCard | null;
  grades: Grade[];
  authenticity: CertAuthenticity | null;
  /** Damage-heatmap entries — additive to `grades`, not a replacement. */
  regions: Region[];
};

const apiBaseUrl = (): string => {
  // NEXT_PUBLIC_* is inlined at build time; for server components we
  // can also read the non-public form, but the same URL applies in
  // dev. Default to the conventional dev port to keep `next dev`
  // working out of the box without a .env.
  return process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
};

/**
 * Fetch a public cert. Returns null on 404 (not-found OR not-yet-
 * completed — the API uses opaque 404 for both); throws on any other
 * non-2xx so the page can render an honest error state.
 *
 * Uses Next.js ISR with a 5-minute revalidate window — matches the
 * `Cache-Control: max-age=300` the API sends, and keeps the cert
 * page snappy under spike traffic (one popular card shared on
 * Twitter shouldn't take down the API).
 */
export async function fetchCertificate(certId: string): Promise<Certificate | null> {
  const url = `${apiBaseUrl()}/cert/${encodeURIComponent(certId)}`;
  const res = await fetch(url, {
    next: { revalidate: 300 },
    headers: { Accept: "application/json" },
  });
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`cert fetch failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as Certificate;
}
