/**
 * Main-thread wrapper around the card-detection Web Worker.
 *
 * The worker compiles OpenCV.js's WASM off-main-thread, so the page
 * stays responsive even on a phone while the 10 MB module is being
 * parsed and instantiated. The wrapper exposes one async function:
 *
 *   const { detect, ready } = createCardDetectWorker();
 *   await ready;                   // wait for the worker to initialize
 *   const quad = await detect(im); // 4 corners or null
 *
 * If the worker can't be created at all (very old browser) or fails to
 * initialize, `detect` resolves to null forever and `ready` rejects —
 * the calling code should fall through to "ship the full frame".
 */

interface Point {
  x: number;
  y: number;
}
type Quad = [Point, Point, Point, Point];

export interface CardDetectWorker {
  ready: Promise<void>;
  detect: (image: ImageData) => Promise<Quad | null>;
  destroy: () => void;
}

export function createCardDetectWorker(): CardDetectWorker {
  let worker: Worker | null = null;
  try {
    // The new-Worker(new URL(..., import.meta.url)) form is the Next.js
    // / webpack pattern that gets the worker bundled and served.
    worker = new Worker(
      new URL("../workers/cardDetect.worker.ts", import.meta.url),
      { type: "module" },
    );
  } catch (e) {
    return makeNoopWorker(
      `Worker construction failed: ${e instanceof Error ? e.message : String(e)}`,
    );
  }

  let nextId = 1;
  const pending = new Map<number, (q: Quad | null) => void>();
  let readyResolve: () => void;
  let readyReject: (e: Error) => void;
  const ready = new Promise<void>((res, rej) => {
    readyResolve = res;
    readyReject = rej;
  });

  worker.addEventListener("message", (ev) => {
    const msg = ev.data as
      | { type: "ready" }
      | { type: "result"; id: number; quad: Quad | null }
      | { type: "error"; id?: number; message: string };
    if (msg.type === "ready") {
      readyResolve();
      return;
    }
    if (msg.type === "result") {
      const cb = pending.get(msg.id);
      if (cb) {
        pending.delete(msg.id);
        cb(msg.quad);
      }
      return;
    }
    if (msg.type === "error") {
      if (msg.id !== undefined) {
        const cb = pending.get(msg.id);
        if (cb) {
          pending.delete(msg.id);
          cb(null); // detection failed → null, caller falls back
        }
      } else {
        // Init-time error.
        readyReject(new Error(msg.message));
      }
      return;
    }
  });

  worker.addEventListener("error", (ev) => {
    readyReject(new Error(`Worker error: ${ev.message ?? "unknown"}`));
  });

  return {
    ready,
    detect: (image) =>
      new Promise<Quad | null>((resolve) => {
        if (!worker) {
          resolve(null);
          return;
        }
        const id = nextId++;
        pending.set(id, resolve);
        worker.postMessage({ type: "detect", id, imageData: image });
      }),
    destroy: () => {
      worker?.terminate();
      worker = null;
      for (const cb of pending.values()) cb(null);
      pending.clear();
    },
  };
}

function makeNoopWorker(reason: string): CardDetectWorker {
  return {
    ready: Promise.reject(new Error(reason)),
    detect: () => Promise.resolve(null),
    destroy: () => {},
  };
}
