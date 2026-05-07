/**
 * Shared wizard-shot config.
 *
 * Single source of truth for the 8-shot capture order, labels, hints,
 * required flag, pose (drives the viewfinder guide overlay), and
 * lighting tag (drives the lighting pill in the viewfinder header).
 *
 * The order matches the design source (`docs/design/capture.jsx`):
 *   front → back → front_full_flash → tilt_30 → corners (TL, TR, BL, BR).
 *
 * `required` is kept tight against the actual API gate: only
 * `front_full` blocks `submitForGrading` (REQUIRED_SHOTS in
 * pipeline_runner.py). Marking back as visually-required would mislead
 * users into thinking the API would reject a front-only submission.
 */
import type { ShotKind } from "@/lib/submission";

export type WizardPose =
  | "flat"
  | "tilt"
  | "corner-tl"
  | "corner-tr"
  | "corner-bl"
  | "corner-br";

export type WizardLight = "ambient" | "flash" | "macro";

export type WizardShot = {
  kind: ShotKind;
  label: string;
  hint: string;
  required: boolean;
  pose: WizardPose;
  light: WizardLight;
  description: string;
  tips: string[];
};

const FLAT_TIPS = [
  "Card flat on a neutral surface (matte gray works best)",
  "Phone parallel to the card · use a tripod or table edge",
  "Diffuse light from two sides · no overhead glare",
  "Card fills > 70% of the frame; gutters of equal width",
];

const TILT_TIPS = [
  "Tilt the top edge of the card ~30° toward the camera",
  "Keep the card roughly centered in frame",
  "Same lighting as the front shot · do not move the lights",
  "Holo parallax detector needs visible foil shift",
];

const CORNER_TIPS = [
  "Switch to macro mode on your phone",
  "Fill the inner ring with the card corner",
  "Card flat, lit evenly · no shadows over the corner",
];

const FLASH_TIPS = [
  "Same framing as the front shot",
  "Flash on; keep white-balance steady",
  "Used for substrate b* and glare-pair detectors",
  "If you see a single hot spot, diffuse with a tissue",
];

export const WIZARD_SHOTS: WizardShot[] = [
  {
    kind: "front_full",
    label: "Front",
    hint: "Whole front, fill the frame, edges parallel.",
    required: true,
    pose: "flat",
    light: "ambient",
    description:
      "Whole card, face up. No flash. Fill the guide rectangle edge-to-edge.",
    tips: FLAT_TIPS,
  },
  {
    kind: "back_full",
    label: "Back",
    hint: "Whole back, same framing as the front.",
    required: false,
    pose: "flat",
    light: "ambient",
    description: "Whole card, face down. Same lighting as the front shot.",
    tips: FLAT_TIPS,
  },
  {
    kind: "front_full_flash",
    label: "Flash front",
    hint:
      "Front under flash; whole card visible. Helps the substrate detector.",
    required: false,
    pose: "flat",
    light: "flash",
    description:
      "Same framing as front. Flash on. Used for substrate b* and glare-pair detectors.",
    tips: FLASH_TIPS,
  },
  {
    kind: "tilt_30",
    label: "Tilt 30°",
    hint: "Tilt the card ~30° so light catches the holo.",
    required: false,
    pose: "tilt",
    light: "ambient",
    description: "Front of card tilted ~30°. Used for holo parallax.",
    tips: TILT_TIPS,
  },
  {
    kind: "corner_tl",
    label: "Corner TL",
    hint: "Macro shot of the top-left corner; corner centered, sharp.",
    required: false,
    pose: "corner-tl",
    light: "macro",
    description: "Macro shot of the top-left corner. Fill the inner ring.",
    tips: CORNER_TIPS,
  },
  {
    kind: "corner_tr",
    label: "Corner TR",
    hint: "Macro shot of the top-right corner.",
    required: false,
    pose: "corner-tr",
    light: "macro",
    description: "Macro shot of the top-right corner.",
    tips: CORNER_TIPS,
  },
  {
    kind: "corner_bl",
    label: "Corner BL",
    hint: "Macro shot of the bottom-left corner.",
    required: false,
    pose: "corner-bl",
    light: "macro",
    description: "Macro shot of the bottom-left corner.",
    tips: CORNER_TIPS,
  },
  {
    kind: "corner_br",
    label: "Corner BR",
    hint: "Macro shot of the bottom-right corner.",
    required: false,
    pose: "corner-br",
    light: "macro",
    description: "Macro shot of the bottom-right corner.",
    tips: CORNER_TIPS,
  },
];

export const REQUIRED_KINDS: ShotKind[] = WIZARD_SHOTS.filter(
  (s) => s.required,
).map((s) => s.kind);
