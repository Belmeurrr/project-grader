/**
 * Unit tests for the OG image data-builder.
 *
 * We don't render a real PNG inside vitest (next/og's wasm Satori
 * pipeline is heavy + not happy in jsdom). Instead, we exercise
 * `buildCertOgData` — the pure function that turns a Certificate (or
 * null) into the strings + colors the JSX template uses. If the layout
 * changes, this test pins the data shape; if data ever drifts, the
 * unfurl image silently breaks and this catches it.
 */
import { describe, it, expect, vi } from "vitest";

// next/og imports `@vercel/og` which doesn't load cleanly under jsdom.
// We don't render in this test — only invoke the pure data builder —
// so stub the module.
vi.mock("next/og", () => ({
  ImageResponse: class {},
}));

// Avoid hitting the real cert API when the module imports
// fetchCertificate. The default export is what would call it; we
// only invoke `buildCertOgData` here, but the import is still
// evaluated.
vi.mock("@/lib/cert", async () => {
  const actual =
    await vi.importActual<typeof import("@/lib/cert")>("@/lib/cert");
  return {
    ...actual,
    fetchCertificate: vi.fn(async () => null),
  };
});

import { buildCertOgData } from "@/app/cert/[id]/opengraph-image";
import type { Certificate } from "@/lib/cert";

const FIXTURE: Certificate = {
  cert_id: "cert_abc123def456",
  completed_at: "2026-04-01T00:00:00Z",
  identified_card: {
    variant_id: "v1",
    name: "Charizard 1st Edition",
    set_code: "BS",
    card_number: "4",
    confidence: 0.97,
  },
  grades: [
    {
      scheme: "psa",
      centering: 9,
      corners: 8.5,
      edges: 8,
      surface: 9,
      final: 8.5,
      confidence: 0.85,
      front_centering_lr: 50,
      front_centering_tb: 50,
      back_centering_lr: 50,
      back_centering_tb: 50,
    },
  ],
  authenticity: {
    verdict: "authentic",
    confidence: 0.94,
    reasons: [],
    model_versions: {},
    detectors: [],
  },
  regions: [],
  population: null,
};

describe("buildCertOgData", () => {
  it("renders card name + final grade + authentic stripe color", () => {
    const data = buildCertOgData(FIXTURE, FIXTURE.cert_id);
    expect(data.cardName).toBe("Charizard 1st Edition");
    expect(data.finalGrade).toBe("8.5");
    expect(data.scheme).toBe("PSA");
    expect(data.verdictLabel).toBe("AUTHENTIC");
    // lime-500 — the color that signals "authentic" in the cert palette.
    expect(data.stripeColor.toLowerCase()).toBe("#84cc16");
  });

  it("falls back gracefully when cert is null", () => {
    const data = buildCertOgData(null, "cert_xxx");
    expect(data.cardName).toBe("Cert not available");
    expect(data.finalGrade).toBe("—");
    expect(data.verdictLabel).toBe("UNVERIFIED");
  });

  it("uses unverified stripe when authenticity is missing", () => {
    const cert: Certificate = {
      ...FIXTURE,
      authenticity: null,
    };
    const data = buildCertOgData(cert, cert.cert_id);
    expect(data.verdictLabel).toBe("UNVERIFIED");
    // zinc-600 — the unverified palette.
    expect(data.stripeColor.toLowerCase()).toBe("#52525b");
  });

  it("renders em-dash when final grade is null (skeleton trainer state)", () => {
    const cert: Certificate = {
      ...FIXTURE,
      grades: [{ ...FIXTURE.grades[0]!, final: null }],
    };
    const data = buildCertOgData(cert, cert.cert_id);
    expect(data.finalGrade).toBe("—");
  });
});
