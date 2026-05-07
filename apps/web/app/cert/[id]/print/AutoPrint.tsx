"use client";

/**
 * Tiny client component that triggers `window.print()` once on mount.
 *
 * Buyers/auction houses opening `/cert/[id]/print` expect the browser
 * print dialog to come up immediately — same UX as Stripe receipts and
 * Square's customer-facing receipts. We delay 250ms so the QR + cert
 * data has a chance to paint before the browser snapshots the page.
 */
import { useEffect } from "react";

export default function AutoPrint() {
  useEffect(() => {
    const t = window.setTimeout(() => {
      window.print();
    }, 250);
    return () => window.clearTimeout(t);
  }, []);
  return null;
}
