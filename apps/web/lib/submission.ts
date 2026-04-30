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
 * Auth: dev mode (settings.dev_auth_enabled) accepts
 * `Authorization: Dev <clerk_id>`. Production wires Clerk's React SDK
 * and swaps `getDevAuthHeader` for the SDK's session-token getter.
 * The capture wizard hits this client through `authedFetch`, so the
 * swap is one function and one call site.
 */

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

export type ShotUploadUrlResponse = {
  shot_id: string;
  upload_url: string;
  s3_key: string;
  expires_at: string;
  required_headers: Record<string, string>;
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
 * "dev-user" so a fresh checkout works without env config). When real
 * Clerk lands, replace this call site with the Clerk session-token
 * getter — the wizard never reaches into the auth scheme directly.
 */
const getDevAuthHeader = (): string => {
  const id = process.env.NEXT_PUBLIC_DEV_CLERK_ID ?? "dev-user";
  return `Dev ${id}`;
};

class ApiError extends Error {
  constructor(
    public status: number,
    public detail: unknown,
  ) {
    super(`api error ${status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
}

async function authedFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const url = `${apiBaseUrl()}${path}`;
  const headers = new Headers(init.headers);
  headers.set("Authorization", getDevAuthHeader());
  if (init.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");
  return fetch(url, { ...init, headers });
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

export async function createSubmission(opts: {
  game_hint?: Game;
} = {}): Promise<SubmissionOut> {
  const res = await authedFetch("/submissions", {
    method: "POST",
    body: JSON.stringify({ game_hint: opts.game_hint ?? null }),
  });
  return asJson<SubmissionOut>(res);
}

export async function getSubmission(id: string): Promise<SubmissionOut | null> {
  const res = await authedFetch(`/submissions/${encodeURIComponent(id)}`);
  if (res.status === 404) return null;
  return asJson<SubmissionOut>(res);
}

// --------------------------------------------------------------------------
// Per-shot upload flow
// --------------------------------------------------------------------------

export async function requestShotUploadUrl(
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
 * Direct PUT to the presigned S3 URL. Pass through the
 * `required_headers` from the upload-url response verbatim — S3
 * signature validation hashes Content-Type and any x-amz-* headers
 * exactly as listed.
 */
export async function uploadShotToS3(
  url: string,
  blob: Blob,
  requiredHeaders: Record<string, string>,
): Promise<void> {
  const res = await fetch(url, {
    method: "PUT",
    headers: requiredHeaders,
    body: blob,
  });
  if (!res.ok) {
    throw new Error(`s3 upload failed: ${res.status} ${res.statusText}`);
  }
}

export async function registerShot(
  submissionId: string,
  shotId: string,
  s3Key: string,
): Promise<ShotOut> {
  const res = await authedFetch(
    `/submissions/${encodeURIComponent(submissionId)}/shots`,
    {
      method: "POST",
      body: JSON.stringify({ shot_id: shotId, s3_key: s3Key }),
    },
  );
  return asJson<ShotOut>(res);
}

export async function submitForGrading(
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
 * Three-step shot upload: presign → PUT → register. Returns the
 * server-side ShotOut so the caller can decide whether quality passed
 * and either advance the wizard or prompt for retake.
 *
 * Errors propagate; the caller is expected to surface them to the
 * user. We don't retry automatically — quality failures are the
 * common case and the user needs to physically re-shoot, not the
 * client to re-PUT.
 */
export async function uploadShot(
  submissionId: string,
  kind: ShotKind,
  blob: Blob,
): Promise<ShotOut> {
  const contentType = blob.type || "image/jpeg";
  const presigned = await requestShotUploadUrl(submissionId, kind, contentType);
  await uploadShotToS3(presigned.upload_url, blob, presigned.required_headers);
  return registerShot(submissionId, presigned.shot_id, presigned.s3_key);
}

export { ApiError };
