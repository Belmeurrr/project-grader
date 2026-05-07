/**
 * Clerk middleware. Gates `/grade/*` (owner-side capture wizard) but
 * leaves `/cert/*` (public share artifact) and the marketing root
 * unauthenticated.
 *
 * The matcher excludes Next internals (`_next`), static assets, and
 * `/cert/*` so the public cert page never runs through Clerk. Inside
 * the middleware we still call `auth.protect()` only for `/grade/*`
 * routes, so adding more public routes later is just a matcher tweak
 * plus a guard here.
 *
 * Dev-mode fallback: when `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is
 * unset, `clerkMiddleware` is a no-op pass-through — the API still
 * accepts `Authorization: Dev <id>` so local dev / CI keep working.
 */

import { NextResponse, type NextRequest } from 'next/server';
import { clerkMiddleware, createRouteMatcher } from '@clerk/nextjs/server';

const isProtectedRoute = createRouteMatcher(['/grade(.*)']);

const hasClerkKeys = Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);

export default hasClerkKeys
  ? clerkMiddleware(async (auth, req) => {
      if (isProtectedRoute(req)) {
        await auth.protect();
      }
    })
  : (_req: NextRequest) => NextResponse.next();

export const config = {
  matcher: [
    // Run on everything except Next internals, static files, and the
    // public cert page. Cert pages must be reachable without auth.
    '/((?!_next/static|_next/image|favicon.ico|cert/).*)',
    // Always run on API routes (none today, but future-proofing).
    '/(api|trpc)(.*)',
  ],
};
