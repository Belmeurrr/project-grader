/**
 * Same-origin proxy → MinIO on localhost:9000.
 *
 * Presigned S3 POST URLs that the API generates point at this origin's
 * `/storage` prefix instead of `:9443`, so the phone uploads through the
 * same standard :443 port as the page itself. The FastAPI settings'
 * `S3_ENDPOINT_URL` is set to `https://<host>/storage` so boto3's
 * generate_presigned_post produces URLs that hit this handler.
 */

import { NextRequest } from "next/server";

const STORAGE_TARGET = "http://127.0.0.1:9000";

async function proxy(
  req: NextRequest,
  ctx: { params: Promise<{ path: string[] }> },
) {
  const { path } = await ctx.params;
  const search = req.nextUrl.search;
  const targetUrl = `${STORAGE_TARGET}/${path.join("/")}${search}`;

  const headers = new Headers(req.headers);
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
    init.duplex = "half";
  }

  const upstream = await fetch(targetUrl, init);
  const responseHeaders = new Headers(upstream.headers);
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
