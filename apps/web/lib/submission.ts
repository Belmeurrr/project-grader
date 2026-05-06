/**
 * Typed client for the owner-side submission endpoints.
 *
 * Counterpart to `lib/cert.ts` (which is the public read-side). This
 * file covers the create → upload-url → PUT-to-S3 → register → submit
 * flow that the capture wizard at `/grade/[id]` drives.
 *
 * Source of truth for the wire shapes is
 * `apps/api/grader/schemas/submissions.py`. Keep in sync by hand for
 * now; openapi-typescript codegen is a future task.
 *
 * Auth: production wires Clerk via `useAuthHeader()` (a React hook
 * that wraps Clerk's `useAuth().getToken()`). When
 * `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` is unset (test/dev), the hook
 * falls back to `Authorization: Dev <NEXT_PUBLIC_DEV_CLERK_ID>` which
 * the FastAPI side accepts when `settings.dev_auth_enabled` is true.
 *
 * All endpoint helpers here take a `token: string | null` argument so
 * the call site (a client component) reads the auth header via the
 * hook and passes it down — keeps these functions hook-free and
 * therefore safely callable from `useEffect`, event handlers, etc.
 */
"use client";

import { useAuth } from "@clerk/nextjs";
import { useCallback } from "react";

export type Game =
  | "pokemon"
  | "mtg"
  | "yugioh"
  | "one_piece"
  | "lorcana"
  | "sports"
  | "other";

export type ShotKind =
  | "front_full"
  | "front_full_flash"
  | "back_full"
  | "corner_tl"
  | "corner_tr"
  | "corner_bl"
  | "corner_br"
  | "tilt_30";

export type SubmissionStatus =
  | "pending"
  | "capturing"
  | "processing"
  | "completed"
  | "failed"
  | "rejected_quality";

export type SubmissionOut = {
  id: string;
  status: SubmissionStatus;
  created_at: string;
  completed_at: string | null;
  rejection_reason: string | null;
  // grades + authenticity intentionally omitted at this surface — the
  // /cert/[id] page is where they're rendered. The wizard only cares
  // about status.
};

/**
 * Presigned-POST form payload for direct browser → S3 upload.
 *
 * The shape mirrors AWS's ``generate_presigned_post`` response: the
 * client sends a ``multipart/form-data`` POST against ``url`` with
 * every entry in ``fields`` set verbatim and the file blob appended
 * last under field name ``file``. The server-side policy embedded in
 * ``fields.policy`` includes a ``content-length-range`` condition,
 * so S3 enforces the max image size during upload — a malicious
 * client can't push a 5 GB blob through a presigned URL meant for
 * phone-photo-sized uploads.
 */
export type ShotUploadUrlResponse = {
  shot_id: string;
  url: string;
  fields: Record<string, string>;
  s3_key: string;
  expires_at: string;
};

export type ShotOut = {
  id: string;
  kind: ShotKind;
  s3_key: string;
  width_px: number;
  height_px: number;
  quality_passed: boolean;
  /** Detector-specific keys; freeform on purpose so the UI can show
   * blur/glare/perspective values without each metric requiring a
   * schema bump. */
  quality_metadata: Record<string, unknown>;
};

export type SubmitResponse = {
  submission_id: string;
  status: SubmissionStatus;
  task_id: string | null;
};

const apiBaseUrl = (): string =>
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

/**
 * Dev-mode auth header. Reads NEXT_PUBLIC_DEV_CLERK_ID (defaulting to
 * "dev-user" so a fresh checkout works without env config). The
 * FastAPI side accepts this when `settings.dev_auth_enabled = true`.
 */
const getDevAuthHeader = (): string => {
  const id = process.env.NEXT_PUBLIC_DEV_CLERK_ID ?? "dev-user";
  return `Dev ${id}`;
};

/**
 * Whether Clerk is configured at all. Used to decide between
 * `useAuth().getToken()` and the dev-mode fallback. Reading
 * `process.env.NEXT_PUBLIC_*` at module scope is fine — Next inlines
 * it at build time.
 */
const clerkConfigured = (): boolean =>
  Boolean(process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY);

/**
 * React hook that returns an `authedFetch` bound to the current
 * Clerk session (or the dev-mode fallback). Must be called from a
 * client component; the returned `fetch` wrapper itself is plain
 * async, so passing it down to event handlers / effects is fine.
 *
 * In dev/test (no `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`), `useAuth()`
 * is still safe to call — `<ClerkProvider>` renders a no-op when
 * unconfigured — but we skip `getToken()` and emit the `Dev <id>`
 * header instead. This keeps tests/local dev working without
 * provisioning Clerk keys.
 */
export function useAuthedFetch(): (
  path: string,
  init?: RequestInit,
) => Promise<Response> {
  // `useAuth` is always called (rules of hooks). When Clerk is
  // unconfigured, `getToken` returns null and we drop into the
  // dev-mode branch.
  const { getToken } = useAuth();

  return useCallback(
    async (path: string, init: RequestInit = {}) => {
      const url = `${apiBaseUrl()}${path}`;
      const headers = new Headers(init.headers);

      let authHeader: string;
      if (clerkConfigured()) {
        const token = await getToken();
        authHeader = token ? `Bearer ${token}` : getDevAuthHeader();
      } else {
        authHeader = getDevAuthHeader();
      }
      headers.set("Authorization", authHeader);

      if (init.body && !headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }
      headers.set("Accept", "application/json");
      return fetch(url, { ...init, headers });
    },
    [getToken],
  );
}

export type AuthedFetch = ReturnType<typeof useAuthedFetch>;

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(`api error ${status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: unknown;
    try {
      detail = await res.json();
    } catch {
      detail = await res.text();
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

// --------------------------------------------------------------------------
// Submission lifecycle
// --------------------------------------------------------------------------

export async function createSubmission(
  authedFetch: AuthedFetch,
  opts: { game_hint?: Game } = {},
): Promise<SubmissionOut> {
  const res = await authedFetch("/submissions", {
    method: "POST",
    body: JSON.stringify({ game_hint: opts.game_hint ?? null }),
  });
  return asJson<SubmissionOut>(res);
}

export async function getSubmission(
  authedFetch: AuthedFetch,
  id: string,
): Promise<SubmissionOut | null> {
  const res = await authedFetch(`/submissions/${encodeURIComponent(id)}`);
  if (res.status === 404) return null;
  return asJson<SubmissionOut>(res);
}

// --------------------------------------------------------------------------
// Per-shot upload flow
// --------------------------------------------------------------------------

export async function requestShotUploadUrl(
  authedFetch: AuthedFetch,
  submissionId: string,
  kind: ShotKind,
  contentType: string,
): Promise<ShotUploadUrlResponse> {
  const res = await authedFetch(
    `/submissions/${encodeURIComponent(submissionId)}/shots/upload-url`,
    {
      method: "POST",
      body: JSON.stringify({ kind, content_type: contentType }),
    },
  );
  return asJson<ShotUploadUrlResponse>(res);
}

/**
 * Direct multipart POST to the presigned S3 URL. The server hands us
 * a ``url`` + ``fields`` pair (the AWS presigned-POST shape); we
 * append every field verbatim — order matters for the policy
 * signature — then append the file blob last under field name
 * ``file``.
 *
 * Why POST not PUT: the presigned-POST form embeds a signed policy
 * with a ``content-length-range`` condition, so S3 server-side
 * rejects uploads that exceed our configured cap. A presigned PUT
 * URL has no such enforcement.
 *
 * IMPORTANT: do NOT set ``Content-Type`` on the request — the
 * browser must compute the multipart boundary header itself, and
 * setting Content-Type explicitly clobbers that.
 */
export async function uploadShotToS3(
  url: string,
  blob: Blob,
  fields: Record<string, string>,
): Promise<void> {
  const form = new FormData();
  // Fields must be appended before the file blob — S3 reads them in
  // order to validate the signed policy.
  for (const [name, value] of Object.entries(fields)) {
    form.append(name, value);
  }
  form.append("file", blob);
  const res = await fetch(url, { method: "POST", body: form });
  if (!res.ok) {
    throw new Error(`s3 upload failed: ${res.status} ${res.statusText}`);
  }
}

export async function registerShot(
  authedFetch: AuthedFetch,
  submissionId: string,
  shotId: string,
  s3Key: string,
  kind: ShotKind,
): Promise<ShotOut> {
  const res = await authedFetch(
    `/submissions/${encodeURIComponent(submissionId)}/shots`,
    {
      method: "POST",
      // ``kind`` is sent in the body so the server can verify it
      // matches the kind embedded in ``s3Key`` — defends against a
      // client mixing shot_id/s3_key from two different presigns to
      // mislabel an upload's kind. See server-side
      // ``register_shot``.
      body: JSON.stringify({ shot_id: shotId, s3_key: s3Key, kind }),
    },
  );
  return asJson<ShotOut>(res);
}

export async function submitForGrading(
  authedFetch: AuthedFetch,
  submissionId: string,
): Promise<SubmitResponse> {
  const res = await authedFetch(
    `/submissions/${encodeURIComponent(submissionId)}/submit`,
    { method: "POST" },
  );
  return asJson<SubmitResponse>(res);
}

// --------------------------------------------------------------------------
// Convenience: full per-shot flow
// --------------------------------------------------------------------------

/**
 * Three-step shot upload: presign → multipart POST → register.
 * Returns the server-side ShotOut so the caller can decide whether
 * quality passed and either advance the wizard or prompt for retake.
 *
 * Errors propagate; the caller is expected to surface them to the
 * user. We don't retry automatically — quality failures are the
 * common case and the user needs to physically re-shoot, not the
 * client to re-POST.
 */
export async function uploadShot(
  authedFetch: AuthedFetch,
  submissionId: string,
  kind: ShotKind,
  blob: Blob,
): Promise<ShotOut> {
  const contentType = blob.type || "image/jpeg";
  const presigned = await requestShotUploadUrl(
    authedFetch,
    submissionId,
    kind,
    contentType,
  );
  await uploadShotToS3(presigned.url, blob, presigned.fields);
  return registerShot(
    authedFetch,
    submissionId,
    presigned.shot_id,
    presigned.s3_key,
    kind,
  );
}

export { ApiError };
