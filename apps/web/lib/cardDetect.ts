/**
 * Browser-side card-quad detector — Canny + findContours + approxPolyDP via
 * OpenCV.js. Replaces the Sobel-bbox heuristic with true 4-corner quad
 * detection so phone shadows, paper-towel patterns, and table edges (which
 * produce rectangular blobs but not closed quadrilaterals) get rejected.
 *
 * OpenCV.js is ~8MB and lazy-loaded on first use. We run on the main thread
 * (no Worker) — Canny + contours on a 320×240 downsample is ~20-40ms on a
 * mid-range phone, well under our 300ms tick budget.
 */

declare global {
  interface Window {
    cv?: any;
  }
}

let cvLoadPromise: Promise<any> | null = null;
let cvReady = false;

// Vendored locally so the page doesn't depend on a public CDN being
// reachable from the user's network. Lives in apps/web/public/, served
// from the same origin as the page (no CORS, no DNS hops).
const OPENCV_SRC = "/opencv-4.10.0.js";

export function loadOpenCV(): Promise<any> {
  if (cvReady && typeof window !== "undefined" && window.cv) {
    return Promise.resolve(window.cv);
  }
  if (cvLoadPromise) return cvLoadPromise;
  cvLoadPromise = new Promise((resolve, reject) => {
    if (typeof window === "undefined") {
      reject(new Error("loadOpenCV called outside browser"));
      return;
    }
    if (window.cv && (window.cv as any).Mat) {
      cvReady = true;
      resolve(window.cv);
      return;
    }
    const existing = document.querySelector(
      `script[data-opencv]`,
    ) as HTMLScriptElement | null;
    const script = existing ?? document.createElement("script");
    if (!existing) {
      script.src = OPENCV_SRC;
      script.async = true;
      script.dataset.opencv = "1";
      script.onerror = () =>
        reject(new Error("Failed to load OpenCV.js from CDN"));
      document.head.appendChild(script);
    }
    // Unified poll loop. Don't rely on onRuntimeInitialized — by the time
    // we'd register that callback, the runtime may already have finished
    // and the callback would never fire. Polling cv.Mat is bulletproof.
    const startedAt = Date.now();
    const TIMEOUT_MS = 60000;
    const wait = () => {
      if (window.cv && (window.cv as any).Mat) {
        cvReady = true;
        resolve(window.cv);
        return;
      }
      if (Date.now() - startedAt > TIMEOUT_MS) {
        reject(new Error("OpenCV.js failed to initialize within 60s"));
        return;
      }
      setTimeout(wait, 150);
    };
    wait();
    if (!existing) script.onload = wait;
  });
  return cvLoadPromise;
}

export function isOpenCVReady(): boolean {
  return cvReady && typeof window !== "undefined" && !!window.cv;
}

export type Point = { x: number; y: number };
export type Quad = [Point, Point, Point, Point]; // TL, TR, BR, BL

/**
 * Detect a card-like quadrilateral in the given ImageData. Returns 4
 * corners ordered TL-TR-BR-BL in image coordinates, or null if no
 * acceptable quad is found.
 *
 * Validation thresholds:
 *  - 4 points after approxPolyDP
 *  - convex
 *  - area between 12% and 95% of frame
 *  - aspect ratio after corner ordering ∈ [0.55, 0.85]
 *  - all interior angles within 75–105°
 *  - opposite-side length ratio within 0.7–1.43
 */
export function detectCardQuad(image: ImageData): Quad | null {
  if (!isOpenCVReady()) return null;
  const cv = window.cv as any;
  const W = image.width;
  const H = image.height;
  const total = W * H;

  const src = cv.matFromImageData(image);
  const gray = new cv.Mat();
  const edges = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const contours = new cv.MatVector();
  const hier = new cv.Mat();
  // CLAHE: contrast-limited adaptive histogram equalization. Flattens
  // local lighting variation (foil glare, uneven shadow on a textured
  // surface) so Canny picks the card edge instead of glossy highlights.
  // ~1-2ms on a 320x240 frame.
  const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));

  let best: Quad | null = null;
  let bestScore = 0;
  let bestRaw: Point[] | null = null;

  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    clahe.apply(gray, gray);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);
    cv.Canny(gray, edges, 75, 200);
    cv.dilate(edges, edges, kernel);
    cv.findContours(
      edges,
      contours,
      hier,
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE,
    );

    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const approx = new cv.Mat();
      try {
        const peri = cv.arcLength(c, true);
        cv.approxPolyDP(c, approx, 0.02 * peri, true);
        if (approx.rows !== 4) continue;
        const area = Math.abs(cv.contourArea(approx));
        if (area < 0.12 * total || area > 0.95 * total) continue;
        if (!cv.isContourConvex(approx)) continue;

        const raw: Point[] = [];
        for (let j = 0; j < 4; j++) {
          raw.push({ x: approx.intPtr(j, 0)[0], y: approx.intPtr(j, 0)[1] });
        }
        const ordered = orderQuad(raw);
        if (!quadGeometryOk(ordered)) continue;
        if (area > bestScore) {
          bestScore = area;
          best = ordered;
          bestRaw = raw;
        }
      } finally {
        approx.delete();
        c.delete();
      }
    }

    // Sub-pixel corner refinement: approxPolyDP snaps to vertices of
    // the simplified contour, which can sit 1-3px off the actual edge.
    // cornerSubPix re-fits using the local intensity gradient. Cheap
    // (<1ms on this frame size) and gives a noticeably smoother polygon.
    if (best) {
      const cornerMat = cv.matFromArray(
        4,
        1,
        cv.CV_32FC2,
        // eslint-disable-next-line prefer-spread
        ([] as number[]).concat(...best.map((p) => [p.x, p.y])),
      );
      try {
        const winSize = new cv.Size(5, 5);
        const zeroZone = new cv.Size(-1, -1);
        const criteria = new cv.TermCriteria(
          cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
          30,
          0.01,
        );
        cv.cornerSubPix(gray, cornerMat, winSize, zeroZone, criteria);
        const refined: Point[] = [];
        const data = cornerMat.data32F;
        for (let j = 0; j < 4; j++) {
          refined.push({ x: data[j * 2]!, y: data[j * 2 + 1]! });
        }
        // cornerSubPix preserves point order, so we can map directly.
        best = refined as Quad;
      } catch {
        // If sub-pixel refinement throws (degenerate region), fall back
        // to the integer corners — still good enough.
      } finally {
        cornerMat.delete();
      }
    }
  } finally {
    src.delete();
    gray.delete();
    edges.delete();
    kernel.delete();
    contours.delete();
    hier.delete();
    clahe.delete();
  }
  // Suppress lint warning for bestRaw being assigned but not directly
  // returned — it's used to short-circuit re-ordering above. Comment-out
  // anti-pattern would lose the intent so keep the noop reference.
  void bestRaw;
  return best;
}

// Order 4 points TL, TR, BR, BL using the canonical sum/diff trick:
// TL has min(x+y), BR has max(x+y); TR has max(x-y), BL has min(x-y).
function orderQuad(pts: Point[]): Quad {
  // Caller guarantees pts.length === 4. Initialize from pts[0] which we
  // know exists; if it doesn't, the upstream contour was malformed and
  // the caller should never have invoked this.
  const seed = pts[0]!;
  let tl: Point = seed;
  let br: Point = seed;
  let tr: Point = seed;
  let bl: Point = seed;
  let minSum = Infinity;
  let maxSum = -Infinity;
  let minDiff = Infinity;
  let maxDiff = -Infinity;
  for (const p of pts) {
    const s = p.x + p.y;
    const d = p.x - p.y;
    if (s < minSum) {
      minSum = s;
      tl = p;
    }
    if (s > maxSum) {
      maxSum = s;
      br = p;
    }
    if (d > maxDiff) {
      maxDiff = d;
      tr = p;
    }
    if (d < minDiff) {
      minDiff = d;
      bl = p;
    }
  }
  return [tl, tr, br, bl];
}

function quadGeometryOk(q: Quad): boolean {
  const [tl, tr, br, bl] = q;
  const top = dist(tl, tr);
  const right = dist(tr, br);
  const bottom = dist(br, bl);
  const left = dist(bl, tl);
  if (top < 1 || right < 1 || bottom < 1 || left < 1) return false;
  // Opposite-side ratio close to 1
  const lr = Math.min(top, bottom) / Math.max(top, bottom);
  const tb = Math.min(left, right) / Math.max(left, right);
  if (lr < 0.70 || tb < 0.70) return false;
  // Aspect (orientation-agnostic)
  const sideAvgH = (top + bottom) / 2;
  const sideAvgV = (left + right) / 2;
  const aspect = Math.min(sideAvgH, sideAvgV) / Math.max(sideAvgH, sideAvgV);
  if (aspect < 0.55 || aspect > 0.85) return false;
  // Interior angles 75–105°
  const triples: Array<[Point, Point, Point]> = [
    [bl, tl, tr],
    [tl, tr, br],
    [tr, br, bl],
    [br, bl, tl],
  ];
  for (const [a, b, c] of triples) {
    const ang = angleAt(a, b, c);
    if (ang < 75 || ang > 105) return false;
  }
  return true;
}

function dist(a: Point, b: Point): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

function angleAt(a: Point, vertex: Point, c: Point): number {
  const v1x = a.x - vertex.x;
  const v1y = a.y - vertex.y;
  const v2x = c.x - vertex.x;
  const v2y = c.y - vertex.y;
  const dot = v1x * v2x + v1y * v2y;
  const m1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const m2 = Math.sqrt(v2x * v2x + v2y * v2y);
  if (m1 === 0 || m2 === 0) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

/** Axis-aligned bounding box of a quad, normalized 0..1 against given dims. */
export function quadBounds(
  q: Quad,
  W: number,
  H: number,
): { x: number; y: number; w: number; h: number } {
  const xs = q.map((p) => p.x);
  const ys = q.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  return {
    x: x0 / W,
    y: y0 / H,
    w: (x1 - x0) / W,
    h: (y1 - y0) / H,
  };
}

/**
 * Per-corner Euclidean distance between two quads (in image pixels), used
 * to decide whether the current detection is "the same card" as the prior
 * one or a fresh detection that should reset the smoother.
 */
export function quadDistance(a: Quad, b: Quad): number {
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    const dx = a[i]!.x - b[i]!.x;
    const dy = a[i]!.y - b[i]!.y;
    sum += Math.sqrt(dx * dx + dy * dy);
  }
  return sum / 4;
}

/**
 * Average a list of quads corner-wise. Caller passes already-ordered quads
 * (TL, TR, BR, BL) so corners line up.
 */
export function averageQuads(quads: Quad[]): Quad {
  const acc: Point[] = [
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
    { x: 0, y: 0 },
  ];
  for (const q of quads) {
    for (let i = 0; i < 4; i++) {
      acc[i]!.x += q[i]!.x;
      acc[i]!.y += q[i]!.y;
    }
  }
  return acc.map((p) => ({
    x: p.x / quads.length,
    y: p.y / quads.length,
  })) as unknown as Quad;
}

/**
 * Maximum corner deviation of any quad in `quads` from `avg` (in pixels).
 * Used as a "how jittery is the detection" metric — small means the
 * detector keeps finding the same card in the same place, which is what
 * we need to call the lock "stable".
 */
export function quadMaxDeviation(quads: Quad[], avg: Quad): number {
  let max = 0;
  for (const q of quads) {
    for (let i = 0; i < 4; i++) {
      const dx = q[i]!.x - avg[i]!.x;
      const dy = q[i]!.y - avg[i]!.y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > max) max = d;
    }
  }
  return max;
}

/**
 * Rolling-window quad smoother + lock detector. Emits a smoothed quad for
 * the polygon overlay so it doesn't visibly jitter, plus a `locked` flag
 * that signals "the same quad has been here for the whole window with low
 * deviation" — that's the trigger condition for auto-capture.
 */
export class QuadLockTracker {
  private readonly windowSize: number;
  private readonly resetDistanceThreshold: number;
  private readonly stableDeviationThreshold: number;
  private buffer: Quad[] = [];

  constructor(opts?: {
    windowSize?: number;
    resetDistance?: number;
    stableDeviation?: number;
  }) {
    this.windowSize = opts?.windowSize ?? 4;
    // If the new quad is more than this many pixels off the last, treat
    // it as a different card and reset history (avoids smoothing across
    // big jumps when the user moves the phone).
    this.resetDistanceThreshold = opts?.resetDistance ?? 30;
    // Locked when every recent quad's corners are within this many pixels
    // of the rolling average — true "card sitting still under the camera".
    this.stableDeviationThreshold = opts?.stableDeviation ?? 6;
  }

  push(quad: Quad | null): {
    smoothed: Quad | null;
    locked: boolean;
    progress: number; // 0..1 fill of the rolling window
    deviation: number;
  } {
    if (!quad) {
      this.buffer = [];
      return { smoothed: null, locked: false, progress: 0, deviation: 0 };
    }
    const last = this.buffer[this.buffer.length - 1];
    if (last && quadDistance(quad, last) > this.resetDistanceThreshold) {
      this.buffer = [quad];
    } else {
      this.buffer.push(quad);
      if (this.buffer.length > this.windowSize) this.buffer.shift();
    }
    const smoothed = averageQuads(this.buffer);
    const deviation = quadMaxDeviation(this.buffer, smoothed);
    const filled = this.buffer.length >= this.windowSize;
    const locked = filled && deviation <= this.stableDeviationThreshold;
    const progress = Math.min(1, this.buffer.length / this.windowSize);
    return { smoothed, locked, progress, deviation };
  }

  reset() {
    this.buffer = [];
  }
}
