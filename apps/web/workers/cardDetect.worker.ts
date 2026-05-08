/**
 * Web Worker that hosts OpenCV.js + the card-quad detector.
 *
 * Why off the main thread: the OpenCV.js bundle is ~10 MB and its WASM
 * module compiles synchronously when the script first runs. On a phone
 * that can stall the main thread for several seconds — long enough that
 * user taps queue up and the page feels broken. Running OpenCV inside a
 * dedicated worker means the WASM compile cost is paid in the worker's
 * thread; the main thread keeps painting and handling input the whole
 * time.
 *
 * Protocol:
 *   Worker posts { type: "ready" } once OpenCV's runtime is initialized.
 *   Main thread posts { type: "detect", id, imageData } to request a
 *     detection. Worker replies with { type: "result", id, quad } where
 *     `quad` is the 4 ordered corners (TL, TR, BR, BL) in image coords
 *     or null if no card-like quadrilateral was found.
 */

/// <reference lib="webworker" />

interface Point {
  x: number;
  y: number;
}
type Quad = [Point, Point, Point, Point];

interface DetectMessage {
  type: "detect";
  id: number;
  imageData: ImageData;
}

interface ReadyMessage {
  type: "ready";
}
interface ResultMessage {
  type: "result";
  id: number;
  quad: Quad | null;
}
interface ErrorMessage {
  type: "error";
  id?: number;
  message: string;
}

type WorkerOut = ReadyMessage | ResultMessage | ErrorMessage;

const ctx = self as unknown as DedicatedWorkerGlobalScope;
let cvReady = false;

// OpenCV.js's UMD wrapper only sets `root.cv = factory()` when
// `typeof window === 'object'` — workers have no window, so without a
// workaround the factory is never called and `cv` is never created.
//
// The classic fix: alias `self.window = self` BEFORE importScripts.
// The UMD wrapper sees window as truthy, runs the browser-globals
// branch, and the worker's global scope ends up with `self.cv` populated
// (initially without Mat etc., until the WASM runtime initializes).
(ctx as unknown as { window: unknown }).window = ctx;

try {
  ctx.importScripts("/opencv-4.10.0.js");
} catch (e) {
  ctx.postMessage({
    type: "error",
    message: `importScripts failed: ${e instanceof Error ? e.message : String(e)}`,
  } satisfies ErrorMessage);
}

// After the script runs, `self.cv` exists (a partial Module object).
// `cv.Mat` and the rest only become callable once the WASM runtime
// has initialized. Set the onRuntimeInitialized callback if available,
// and poll as a fallback in case it's already past that point.
const cvAtLoad = (ctx as unknown as { cv?: { onRuntimeInitialized?: () => void; Mat?: unknown } }).cv;
if (cvAtLoad) {
  const cb = () => {
    cvReady = true;
    ctx.postMessage({ type: "ready" } satisfies ReadyMessage);
  };
  if (cvAtLoad.Mat) {
    cb();
  } else {
    cvAtLoad.onRuntimeInitialized = cb;
  }
}
function waitForRuntime() {
  if (cvReady) return;
  const cv = (ctx as unknown as { cv?: { Mat?: unknown } }).cv;
  if (cv && cv.Mat) {
    cvReady = true;
    ctx.postMessage({ type: "ready" } satisfies ReadyMessage);
    return;
  }
  setTimeout(waitForRuntime, 80);
}
waitForRuntime();

ctx.addEventListener("message", (ev: MessageEvent<DetectMessage>) => {
  const msg = ev.data;
  if (msg.type !== "detect") return;
  if (!cvReady) {
    // Detector not ready yet; respond with null so the caller can
    // fall back to full-frame upload without blocking.
    const out: ResultMessage = { type: "result", id: msg.id, quad: null };
    ctx.postMessage(out);
    return;
  }
  try {
    const quad = detect(msg.imageData);
    const out: ResultMessage = { type: "result", id: msg.id, quad };
    ctx.postMessage(out);
  } catch (e) {
    const out: ErrorMessage = {
      type: "error",
      id: msg.id,
      message: e instanceof Error ? e.message : String(e),
    };
    ctx.postMessage(out);
  }
});

function detect(image: ImageData): Quad | null {
  const cv = (ctx as unknown as { cv: any }).cv;
  const W = image.width;
  const H = image.height;
  const total = W * H;

  const src = cv.matFromImageData(image);
  const gray = new cv.Mat();
  const edges = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const contours = new cv.MatVector();
  const hier = new cv.Mat();
  const clahe = new cv.CLAHE(2.0, new cv.Size(8, 8));

  let best: Quad | null = null;
  let bestArea = 0;

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
          raw.push({
            x: approx.intPtr(j, 0)[0],
            y: approx.intPtr(j, 0)[1],
          });
        }
        const ordered = orderQuad(raw);
        if (!quadGeometryOk(ordered)) continue;
        if (area > bestArea) {
          bestArea = area;
          best = ordered;
        }
      } finally {
        approx.delete();
        c.delete();
      }
    }

    // Sub-pixel corner refinement: approxPolyDP snaps to vertices of
    // the simplified contour, which can sit 1-3 px off the actual edge.
    if (best) {
      const cornerMat = cv.matFromArray(
        4,
        1,
        cv.CV_32FC2,
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
          refined.push({ x: data[j * 2], y: data[j * 2 + 1] });
        }
        best = refined as Quad;
      } catch {
        // sub-pixel failed; integer corners are fine.
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
  return best;
}

function orderQuad(pts: Point[]): Quad {
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
  const lr = Math.min(top, bottom) / Math.max(top, bottom);
  const tb = Math.min(left, right) / Math.max(left, right);
  if (lr < 0.7 || tb < 0.7) return false;
  const sideAvgH = (top + bottom) / 2;
  const sideAvgV = (left + right) / 2;
  const aspect = Math.min(sideAvgH, sideAvgV) / Math.max(sideAvgH, sideAvgV);
  if (aspect < 0.55 || aspect > 0.85) return false;
  for (const [a, b, c] of [
    [bl, tl, tr],
    [tl, tr, br],
    [tr, br, bl],
    [br, bl, tl],
  ] as Array<[Point, Point, Point]>) {
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

// Fool TypeScript's "all paths must export" check (this file isn't a
// module — it's a worker entry — but tsc parses it the same).
export type {};

// Intentional reference so the type is considered used.
type _Out = WorkerOut;
const _: _Out | null = null;
void _;
