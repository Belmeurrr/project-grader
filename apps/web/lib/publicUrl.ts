/**
 * Resolve the canonical public base URL for the deployment.
 *
 * Used when constructing the cert URL that gets baked into QR codes,
 * share intents, and printed footers. We want a stable origin that
 * resolves correctly when scanned from a phone — i.e. NOT
 * `http://localhost:3000` even in dev when someone scans from another
 * machine.
 *
 * Lookup order:
 *   1. NEXT_PUBLIC_PUBLIC_BASE_URL — explicit override (set this in
 *      production deploys).
 *   2. NEXT_PUBLIC_SITE_URL — alternative env name some Next.js
 *      starters use; we accept it for compatibility.
 *   3. VERCEL_URL — auto-set by Vercel previews; we prepend "https://".
 *   4. http://localhost:3000 — dev default; matches `next dev --port 3000`.
 *
 * Returned WITHOUT a trailing slash so callers can append a path
 * directly: `${resolvePublicBaseUrl()}/cert/${id}`.
 */
export function resolvePublicBaseUrl(): string {
  const explicit =
    process.env.NEXT_PUBLIC_PUBLIC_BASE_URL ?? process.env.NEXT_PUBLIC_SITE_URL;
  if (explicit) return stripTrailingSlash(explicit);
  const vercel = process.env.VERCEL_URL;
  if (vercel) return `https://${stripTrailingSlash(vercel)}`;
  return "http://localhost:3000";
}

function stripTrailingSlash(s: string): string {
  return s.endsWith("/") ? s.slice(0, -1) : s;
}
