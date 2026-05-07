/**
 * <DefectList> — DINGS-style itemized defect rationale.
 *
 * Renders beneath <DamageHeatmap> on the public cert page. Where the
 * heatmap shows WHERE on the card the defects sit (via positioned
 * cells), this list shows WHAT each one is in plain English — TAG
 * Grading's textual rationale is the single biggest perceived-
 * transparency feature TAG has that we don't, so this surfaces every
 * region's `reasons[]` as a short, color-dotted bullet list grouped by
 * kind (centering / corners / edges / surface).
 *
 * The strings themselves are heuristic stand-ins keyed off (kind,
 * severity) on the API side — see
 * `apps/api/grader/routers/cert.py::_reasons_for`. Future trainer
 * outputs (measured pixel locations, per-defect bounding boxes,
 * textual classifications) will populate `Region.reasons` directly so
 * this component renders verbatim without the API helper.
 *
 * Empty state: when no region carries any reason (a clean card, or
 * every kind in the "ok" bucket), shows a subtle grey "No defects
 * flagged." string rather than an empty section. That's a positive
 * outcome to surface, not a missing feature.
 *
 * Stays a sibling of <DamageHeatmap> rather than folding into it: the
 * heatmap shape stays unchanged (additive), and a future redesign that
 * promotes the textual list above the visual map can swap their order
 * without touching either component.
 */
import type { Region, RegionKind, RegionPosition, RegionSeverity } from "@/lib/cert";

// Keep the dot palette in sync with SEVERITY_RING_CLASSES on the cert
// page — same lime/amber/red/zinc semantics, just smaller and solid
// rather than ringed-with-fill so the bullet list reads as text-first
// with a colour accent.
const SEVERITY_DOT_CLASSES: Record<RegionSeverity, string> = {
  ok: "bg-lime-500",
  minor: "bg-amber-500",
  major: "bg-red-500",
  unknown: "bg-zinc-500/40",
};

const KIND_LABEL: Record<RegionKind, string> = {
  centering: "Centering",
  corner: "Corners",
  edge: "Edges",
  surface: "Surface",
};

const KIND_ORDER: RegionKind[] = ["centering", "corner", "edge", "surface"];

const POSITION_LABEL: Record<RegionPosition, string> = {
  top_left: "Top-left corner",
  top_right: "Top-right corner",
  bottom_left: "Bottom-left corner",
  bottom_right: "Bottom-right corner",
  top: "Top edge",
  right: "Right edge",
  bottom: "Bottom edge",
  left: "Left edge",
  whole_card: "",
};

function regionLabel(region: Region): string {
  // Centering + surface use whole_card position; their "label" is the
  // kind itself ("Centering" / "Surface") rather than a positional
  // prefix. The other kinds get their position name verbatim.
  if (region.kind === "centering") return "Centering";
  if (region.kind === "surface") return "Surface";
  return POSITION_LABEL[region.position] || KIND_LABEL[region.kind];
}

export type DefectListProps = {
  regions: Region[];
};

export default function DefectList({ regions }: DefectListProps) {
  // Filter to entries that actually have something to say. A region
  // with severity=ok / empty reasons gets dropped — we don't want to
  // render every corner just to say "fine".
  const flagged = regions.filter((r) => r.reasons.length > 0);

  return (
    <section className="mt-6" aria-labelledby="defects-heading">
      <h3
        id="defects-heading"
        className="mb-2 text-xs font-medium uppercase tracking-wide text-zinc-400"
      >
        Defects flagged
      </h3>
      {flagged.length === 0 ? (
        <p className="text-sm text-zinc-500">No defects flagged.</p>
      ) : (
        <div className="flex flex-col gap-4">
          {KIND_ORDER.map((kind) => {
            const items = flagged.filter((r) => r.kind === kind);
            if (items.length === 0) return null;
            return (
              <DefectGroup key={kind} kind={kind} items={items} />
            );
          })}
        </div>
      )}
    </section>
  );
}

function DefectGroup({
  kind,
  items,
}: {
  kind: RegionKind;
  items: Region[];
}) {
  return (
    <div>
      <h4 className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
        {KIND_LABEL[kind]}
      </h4>
      <ul className="flex flex-col gap-1.5">
        {items.flatMap((region) =>
          region.reasons.map((reason, i) => (
            <li
              key={`${region.kind}-${region.position}-${i}`}
              className="flex items-start gap-2 text-sm text-zinc-300"
            >
              <span
                className={`mt-1.5 inline-block h-1.5 w-1.5 flex-none rounded-full ${SEVERITY_DOT_CLASSES[region.severity]}`}
                aria-hidden="true"
              />
              <span>
                <span className="text-zinc-200">{regionLabel(region)}</span>
                <span className="text-zinc-500"> — </span>
                <span className="text-zinc-400">{reason}</span>
              </span>
            </li>
          )),
        )}
      </ul>
    </div>
  );
}
