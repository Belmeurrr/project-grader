/**
 * Tests for <DefectList>.
 *
 * The component is the textual rationale list TAG Grading calls
 * "DINGS" — sibling of <DamageHeatmap>. We pin the two highest-value
 * behaviors:
 *   1. Empty state renders the subtle "No defects flagged." line when
 *      every region's reasons[] is empty.
 *   2. Populated state renders one bullet per (region, reason) pair,
 *      grouped by kind heading (Centering / Corners / Edges / Surface),
 *      with the right "Region label — reason" string per item.
 *
 * We deliberately don't snapshot the markup — the indicator dot's
 * exact pixel size and the section ordering can shift without a
 * regression. The assertions target the user-visible text + structure.
 */
import { describe, it, expect } from "vitest";
import { render, screen, within, cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

import DefectList from "@/components/cert/DefectList";
import type { Region } from "@/lib/cert";

afterEach(() => {
  cleanup();
});

function makeRegion(overrides: Partial<Region>): Region {
  return {
    kind: "centering",
    position: "whole_card",
    score: null,
    severity: "ok",
    reasons: [],
    ...overrides,
  };
}

// A canonical 10-region payload matching what `_build_regions_for_grade`
// emits (1 centering + 4 edges + 4 corners + 1 surface). Tweaks per
// test override individual entries.
function baseRegions(): Region[] {
  return [
    makeRegion({ kind: "centering", position: "whole_card" }),
    makeRegion({ kind: "edge", position: "top" }),
    makeRegion({ kind: "edge", position: "right" }),
    makeRegion({ kind: "edge", position: "bottom" }),
    makeRegion({ kind: "edge", position: "left" }),
    makeRegion({ kind: "corner", position: "top_left" }),
    makeRegion({ kind: "corner", position: "top_right" }),
    makeRegion({ kind: "corner", position: "bottom_left" }),
    makeRegion({ kind: "corner", position: "bottom_right" }),
    makeRegion({ kind: "surface", position: "whole_card" }),
  ];
}

describe("<DefectList>", () => {
  it("renders the empty state when no region has any reason", () => {
    render(<DefectList regions={baseRegions()} />);
    expect(screen.getByText(/No defects flagged\./i)).toBeInTheDocument();
    // The "Defects flagged" heading still renders (it's the section
    // label); but no kind sub-headings should appear.
    expect(screen.queryByText(/^Centering$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Corners$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Edges$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Surface$/)).not.toBeInTheDocument();
  });

  it("renders an empty state for a fully empty regions array", () => {
    render(<DefectList regions={[]} />);
    expect(screen.getByText(/No defects flagged\./i)).toBeInTheDocument();
  });

  it("renders bullets grouped by kind for populated reasons", () => {
    const regions: Region[] = [
      makeRegion({
        kind: "centering",
        position: "whole_card",
        severity: "major",
        reasons: ["Significant off-center crop"],
      }),
      makeRegion({
        kind: "edge",
        position: "top",
        severity: "minor",
        reasons: ["Minor edge wear"],
      }),
      makeRegion({
        kind: "edge",
        position: "right",
        severity: "minor",
        reasons: ["Minor edge wear"],
      }),
      makeRegion({
        kind: "corner",
        position: "top_left",
        severity: "major",
        reasons: ["Visible corner damage"],
      }),
      makeRegion({
        kind: "corner",
        position: "bottom_right",
        severity: "ok",
        reasons: [],
      }),
      makeRegion({
        kind: "surface",
        position: "whole_card",
        severity: "unknown",
        reasons: ["Analysis pending"],
      }),
    ];
    render(<DefectList regions={regions} />);

    // No empty-state copy.
    expect(screen.queryByText(/No defects flagged\./i)).not.toBeInTheDocument();

    // Kind sub-headings present for kinds that have ≥1 flagged region.
    expect(screen.getByRole("heading", { name: "Centering" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Corners" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Edges" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Surface" })).toBeInTheDocument();

    // Per-row content: region label + reason.
    expect(
      screen.getByText(/Significant off-center crop/),
    ).toBeInTheDocument();
    expect(screen.getByText(/Top-left corner/)).toBeInTheDocument();
    expect(screen.getByText(/Visible corner damage/)).toBeInTheDocument();
    expect(screen.getByText(/Analysis pending/)).toBeInTheDocument();

    // Both edge bullets render — same reason string, two distinct
    // positions ("Top edge" / "Right edge").
    expect(screen.getByText(/Top edge/)).toBeInTheDocument();
    expect(screen.getByText(/Right edge/)).toBeInTheDocument();
    const minorEdgeMatches = screen.getAllByText(/Minor edge wear/);
    expect(minorEdgeMatches).toHaveLength(2);

    // Bullet count: one <li> per (region, reason) — 1 centering + 2
    // edges + 1 corner + 1 surface = 5 list items.
    const items = screen.getAllByRole("listitem");
    expect(items).toHaveLength(5);
  });

  it("hides kind sections when only ok-severity regions exist for that kind", () => {
    const regions: Region[] = [
      makeRegion({
        kind: "centering",
        position: "whole_card",
        severity: "minor",
        reasons: ["Minor off-center"],
      }),
      // Corners all ok → no Corners section should render.
      makeRegion({ kind: "corner", position: "top_left", severity: "ok" }),
      makeRegion({ kind: "corner", position: "top_right", severity: "ok" }),
    ];
    render(<DefectList regions={regions} />);
    expect(
      screen.getByRole("heading", { name: "Centering" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("heading", { name: "Corners" }),
    ).not.toBeInTheDocument();
  });
});
