/**
 * `/grade` — entry point for a new grading submission.
 *
 * Single button that creates a Submission via the API and redirects
 * to `/grade/[id]` where the capture wizard runs. We don't try to
 * lazily create-on-first-shot because the submission UUID is needed
 * up-front to scope the per-shot S3 keys.
 *
 * Game hint is omitted at v1 — detection runs server-side. A future
 * enhancement: a "I'm grading a Pokemon card" preselector that skips
 * the detection pass and saves a few hundred ms.
 */

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { ApiError, createSubmission } from "@/lib/submission";

export default function GradeStartPage() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const onStart = async () => {
    setBusy(true);
    setError(null);
    try {
      const sub = await createSubmission();
      router.push(`/grade/${sub.id}`);
    } catch (e) {
      setBusy(false);
      if (e instanceof ApiError && e.status === 401) {
        setError(
          "Authentication failed. In dev, set NEXT_PUBLIC_DEV_CLERK_ID and ensure the API is running with dev_auth_enabled.",
        );
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    }
  };

  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-8 px-6 py-20">
      <header className="flex flex-col gap-3">
        <p className="text-sm uppercase tracking-widest text-lime-500">
          New submission
        </p>
        <h1 className="text-4xl font-semibold leading-tight">
          Grade a card
        </h1>
        <p className="max-w-xl text-zinc-400">
          You&apos;ll capture six shots: front, back, and the four corners.
          Each shot is quality-checked the moment you upload it — bad shots
          are rejected so the grader sees only usable evidence. Front is
          required to grade; the rest sharpen the result.
        </p>
      </header>

      <section className="flex flex-col gap-3">
        <button
          type="button"
          onClick={onStart}
          disabled={busy}
          className="self-start rounded-xl border border-lime-500/40 bg-lime-500/10 px-5 py-3 text-sm font-medium text-lime-300 transition-colors hover:bg-lime-500/20 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? "Starting…" : "Start capture"}
        </button>
        {error && (
          <p className="rounded-xl border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-300">
            {error}
          </p>
        )}
      </section>
    </main>
  );
}
