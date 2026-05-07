/**
 * Smoke test for the capture wizard at `/grade/[id]`.
 *
 * The default page export wires `next/navigation`'s `useRouter` and
 * the `use(params)` Suspense unwrap into `<GradePage/>`; mocking that
 * thicket out for a behaviour test is more work than it's worth. We
 * import the named `Body` export instead — same component the
 * default page renders, minus the routing shell — and hand it a
 * deterministic `SubmissionOut` and a stub `authedFetch` that never
 * fires (no shots have been captured yet, so the wizard renders
 * idle rows).
 *
 * What this test pins:
 *   - `Body` switches into the wizard branch when status is "pending"
 *     (rather than "processing" or terminal).
 *   - All eight wizard rows render with their labels.
 *   - "Submit for grading" is disabled before any front shot passes
 *     (frontPassed=false gate). This is the most-likely-to-regress
 *     piece of the wizard's logic — if a future refactor flips the
 *     boolean or drops the `disabled` prop, the entire authenticate
 *     -before-submit invariant breaks.
 *   - The graceful-degrade path on a browser without
 *     `navigator.mediaDevices`. We don't activate any row here (so
 *     the CaptureSurface doesn't mount), but verify nothing in the
 *     idle render path accidentally touches mediaDevices.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

// --- Mocks ---------------------------------------------------------------

// Bypass Clerk: the wizard calls `useAuth().getToken()` indirectly via
// `useAuthedFetch`. The submission module's hook must be rerouted to a
// no-op so the test render tree never touches a real Clerk provider.
vi.mock("@clerk/nextjs", () => ({
  useAuth: () => ({ getToken: async () => null, isSignedIn: true }),
}));

// We import the page module — pull `Body` and a stable submission
// module reference out of it. Mock only the parts of submission.ts
// that would otherwise fire network calls if the test ever activates
// a row. Keep `useAuthedFetch` intact because Body itself doesn't
// call it; the parent (GradePage) does, and we pass our own
// authedFetch stub straight into Body.
vi.mock("@/lib/submission", async () => {
  const actual = await vi.importActual<
    typeof import("@/lib/submission")
  >("@/lib/submission");
  return {
    ...actual,
    // Deterministic getSubmission: returns a pending submission. Body
    // doesn't call this directly (it's invoked by GradePage), but the
    // mock guarantees no fetch ever escapes if a future refactor
    // moves the call site.
    getSubmission: vi.fn(async () => ({
      id: "test-submission",
      status: "pending" as const,
      created_at: "2026-04-29T00:00:00Z",
      completed_at: null,
      rejection_reason: null,
    })),
    submitForGrading: vi.fn(async () => ({
      submission_id: "test-submission",
      status: "processing" as const,
      task_id: "task-1",
    })),
    uploadShot: vi.fn(),
  };
});

// Stub navigator.mediaDevices to undefined to assert the wizard's
// idle render doesn't blow up on a headless browser. The
// CaptureSurface's getUserMedia path only fires when a row is
// activated (which this happy-path test never does).
beforeEach(() => {
  // jsdom doesn't ship navigator.mediaDevices; assert it stays undefined.
  // @ts-expect-error — deliberately simulating an unsupported browser
  delete (globalThis.navigator as Navigator & { mediaDevices?: unknown })
    .mediaDevices;
});

// --- Test ----------------------------------------------------------------

import { Body } from "@/components/grade/Wizard";
import type { SubmissionOut } from "@/lib/submission";

describe("grade wizard happy path", () => {
  it("renders all 8 shot rows and disables submit until front passes", () => {
    const submission: SubmissionOut = {
      id: "test-submission",
      status: "pending",
      created_at: "2026-04-29T00:00:00Z",
      completed_at: null,
      rejection_reason: null,
    };

    const authedFetch = vi.fn(
      async () => new Response("{}", { status: 200 }),
    ) as unknown as ReturnType<
      typeof import("@/lib/submission").useAuthedFetch
    >;

    render(
      <Body
        submission={submission}
        authedFetch={authedFetch}
        onShotsChanged={async () => {}}
        onSubmitted={async () => {}}
      />,
    );

    // 1. All eight wizard rows render in the ShotRail. Labels come
    // from WIZARD_SHOTS in `apps/web/components/grade/shots.ts` —
    // if anyone reorders or drops a shot, the assertion lights up.
    // The active shot's label also renders in the viewfinder header,
    // so we use getAllByText and assert ≥1 match.
    const expectedLabels = [
      "Front",
      "Back",
      "Flash front",
      "Tilt 30°",
      "Corner TL",
      "Corner TR",
      "Corner BL",
      "Corner BR",
    ];
    for (const label of expectedLabels) {
      expect(screen.getAllByText(label).length).toBeGreaterThanOrEqual(1);
    }

    // 2. Submit button is disabled — front shot hasn't passed yet.
    const submit = screen.getByRole("button", {
      name: /submit for grading/i,
    });
    expect(submit).toBeDisabled();

    // 3. Helper text explains why submit is disabled (this is the
    // user-facing contract of frontPassed=false; if it's ever removed,
    // the submit button just looks broken).
    expect(
      screen.getByText(/the front shot must pass quality before grading/i),
    ).toBeInTheDocument();

    // 4. authedFetch was never called by an idle render — nothing
    // should hit the network just from mounting Body in pending state.
    expect(authedFetch).not.toHaveBeenCalled();
  });
});
