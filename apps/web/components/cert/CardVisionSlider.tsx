"use client";

/**
 * <CardVisionSlider> — TAG Grading-style "Card Vision" opacity slider.
 *
 * Crossfades between the regular front capture (standard light) and
 * the flash capture (raking flash light) using a draggable opacity
 * slider. The flash shot reveals surface defects — scratches, print
 * lines, holographic interference — that wash out under flat lighting,
 * so the slider is a direct, evidence-first replacement for the kind
 * of "before/after defect overlay" headline visual that grading shops
 * use to demonstrate authenticity rigor.
 *
 * Why a Client Component:
 *   `<input type="range">` change events are interactive UI; React
 *   Server Components can't subscribe to DOM events. The slider
 *   itself is the only piece that needs hydration — the cert page
 *   embeds this component inside an otherwise-static RSC tree.
 *
 * Degradation modes:
 *   - both URLs present → slider with crossfade
 *   - only frontUrl     → static image, no slider (the flash isn't
 *     available so there's nothing to fade to)
 *   - both null         → render nothing; the parent falls through to
 *     the existing placeholder rectangle
 *
 * Accessibility:
 *   The range input gets an explicit aria-label + valuetext so
 *   screen readers narrate the position as a percentage. Keyboard
 *   users can tab to the slider and use arrow keys / Home / End to
 *   drag it (default range-input behavior). Both <img> elements
 *   carry descriptive alt text.
 */

import { useId, useState } from "react";

export type CardVisionSliderProps = {
  /** Presigned URL for the standard-light canonical front shot. */
  frontUrl: string | null;
  /** Presigned URL for the flash-lit canonical front shot. */
  flashUrl: string | null;
  /**
   * Optional URL for the 30-degree tilt shot. Reserved for a future
   * "show holographic parallax" surface; not crossfaded today (the
   * geometry differs from the straight-on canonicals, so a naive
   * opacity blend would just look like ghosting).
   */
  tiltUrl?: string | null;
  /** Optional alt-text prefix, e.g. the identified card's name. */
  altPrefix?: string;
};

export default function CardVisionSlider({
  frontUrl,
  flashUrl,
  altPrefix = "Card",
}: CardVisionSliderProps) {
  const sliderId = useId();
  const [opacity, setOpacity] = useState(0);

  // Both URLs absent → caller is responsible for the placeholder.
  // Returning null lets the cert page keep the existing
  // `<DamageHeatmap>` rectangle as-is.
  if (!frontUrl && !flashUrl) return null;

  // Only one URL present → render a static image with no slider.
  // Crossfading needs both halves; without the flash there's nothing
  // to fade to. We pick whichever is available so the cert page still
  // shows the actual card.
  const onlyOne =
    (!frontUrl && flashUrl) || (frontUrl && !flashUrl) ? true : false;

  if (onlyOne) {
    const url = frontUrl ?? flashUrl;
    if (!url) return null;
    return (
      <div className="flex flex-col items-center gap-2">
        <div className="relative aspect-[5/7] w-full overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40">
          <img
            src={url}
            alt={`${altPrefix} — front scan`}
            className="absolute inset-0 h-full w-full object-cover"
          />
        </div>
      </div>
    );
  }

  // Both present — full slider experience.
  // ``opacity`` 0 = standard light fully visible, 1 = flash fully visible.
  return (
    <div className="flex flex-col items-center gap-3">
      <div
        className="relative aspect-[5/7] w-full overflow-hidden rounded-xl border border-zinc-800 bg-zinc-900/40"
        data-testid="card-vision-stack"
      >
        <img
          src={frontUrl ?? undefined}
          alt={`${altPrefix} — standard light`}
          className="absolute inset-0 h-full w-full object-cover"
        />
        <img
          src={flashUrl ?? undefined}
          alt={`${altPrefix} — flash light, surface defects highlighted`}
          className="absolute inset-0 h-full w-full object-cover transition-opacity duration-100"
          style={{ opacity }}
          data-testid="card-vision-flash"
        />
      </div>
      <div className="flex w-full flex-col gap-1">
        <input
          id={sliderId}
          type="range"
          min={0}
          max={100}
          value={Math.round(opacity * 100)}
          onChange={(e) => setOpacity(Number(e.target.value) / 100)}
          aria-label="Card Vision: crossfade between standard light and flash"
          aria-valuetext={`${Math.round(opacity * 100)}% flash`}
          className="w-full accent-lime-500"
          data-testid="card-vision-slider"
        />
        <label
          htmlFor={sliderId}
          className="flex justify-between text-[10px] uppercase tracking-wide text-zinc-500"
        >
          <span>← Standard light</span>
          <span>Flash light →</span>
        </label>
      </div>
    </div>
  );
}
