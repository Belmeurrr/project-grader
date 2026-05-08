/**
 * Same-origin proxy → FastAPI on localhost:8000.
 *
 * iOS Safari can struggle to reach Tailscale-served alt ports (8443) from
 * some networks; routing the API through the same :443 origin sidesteps
 * the issue entirely. The wizard's `NEXT_PUBLIC_API_URL` is set to `/api`
 * and every endpoint helper calls `${apiBase}${path}`, so e.g.
 * `POST /api/submissions` hits this handler, which strips the `/api`
 * prefix and forwards to `http://127.0.0.1:8000/submissions`.
 */

import { NextRequest } from "next/server";

const API_TARGET = "http://127.0.0.1:8000";

async function proxy(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path } = await ctx.params;
  const search = req.nextUrl.search;
  const targetUrl = `${API_TARGET}/${path.join("/")}${search}`;

  const headers = new Headers(req.headers);
  // The FastAPI side validates Origin/Host; let it see the original
  // forwarded ones so CORS + Clerk JWT checks still work.
  headers.delete("host");
  headers.delete("connection");

  const hasBody = !["GET", "HEAD", "OPTIONS"].includes(req.method);

  const init: RequestInit & { duplex?: "half" } = {
    method: req.method,
    headers,
    redirect: "manual",
  };
  if (hasBody) {
    init.body = req.body;
    init.duplex = "half"; // streaming uploads
  }

  const upstream = await fetch(targetUrl, init);
  const responseHeaders = new Headers(upstream.headers);
  // Hop-by-hop headers shouldn't be forwarded.
  responseHeaders.delete("transfer-encoding");
  responseHeaders.delete("connection");

  return new Response(upstream.body, {
    status: upstream.status,
    headers: responseHeaders,
  });
}

export const GET = proxy;
export const POST = proxy;
export const PUT = proxy;
export const PATCH = proxy;
export const DELETE = proxy;
export const OPTIONS = proxy;
export const HEAD = proxy;

export const dynamic = "force-dynamic";
