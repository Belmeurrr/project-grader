"use client";

/**
 * <CertHeaderActions> — share/print/copy/QR/manual-review row at the
 * top of the cert page, ported from docs/design/cert.jsx CertHeader.
 *
 * Browser-only because:
 *   - navigator.clipboard / navigator.share are runtime APIs
 *   - useState drives the copy-toast and QR popover
 *
 * Uses qrcode.react for the QR popover (same library the print
 * one-pager uses).
 */

import { useState } from "react";
import { QRCodeSVG } from "qrcode.react";

const REVIEW_EMAIL = "reviews@grader.app";

type Props = {
  certId: string;
  certUrl: string;
  cardName: string;
  finalGradeLabel: string;
  printHref: string;
  ogImageHref: string;
};

export default function CertHeaderActions({
  certId,
  certUrl,
  cardName,
  finalGradeLabel,
  printHref,
  ogImageHref,
}: Props) {
  const [copied, setCopied] = useState<"id" | "url" | null>(null);
  const [qrOpen, setQrOpen] = useState(false);
  const shareText = `${cardName} — graded ${finalGradeLabel} via Project Grader`;

  async function copyText(value: string, kind: "id" | "url") {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(kind);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      setCopied(null);
    }
  }

  async function onShare() {
    if (typeof navigator !== "undefined" && "share" in navigator) {
      try {
        await navigator.share({
          title: `Cert ${certId}`,
          text: shareText,
          url: certUrl,
        });
        return;
      } catch {
        /* user cancelled or share unavailable — fall through */
      }
    }
    // No native share → copy URL as a graceful fallback.
    await copyText(certUrl, "url");
  }

  const reviewMailto = `mailto:${REVIEW_EMAIL}?subject=${encodeURIComponent(
    `Manual review request — ${certId}`,
  )}&body=${encodeURIComponent(
    `Cert: ${certUrl}\nCard: ${cardName}\nFinal grade: ${finalGradeLabel}\n\nReason for review:\n`,
  )}`;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <button className="pg-btn pg-btn-ghost" onClick={onShare}>
          <Icon kind="share" /> Share
        </button>
        <button
          className="pg-btn pg-btn-ghost"
          onClick={() => copyText(certUrl, "url")}
        >
          <Icon kind="link" /> {copied === "url" ? "Copied" : "Copy URL"}
        </button>
        <button
          className="pg-btn pg-btn-ghost"
          onClick={() => copyText(certId, "id")}
        >
          <Icon kind="copy" /> {copied === "id" ? "Copied" : "Copy ID"}
        </button>
        <a
          className="pg-btn pg-btn-ghost"
          href={printHref}
          target="_blank"
          rel="noopener"
        >
          <Icon kind="print" /> Print / PDF
        </a>
        <a
          className="pg-btn pg-btn-ghost"
          href={ogImageHref}
          download={`cert-${certId}.png`}
        >
          <Icon kind="image" /> Save image
        </a>
        <div style={{ flex: 1 }} />
        <button
          className="pg-btn pg-btn-ghost"
          onClick={() => setQrOpen((v) => !v)}
          aria-expanded={qrOpen}
        >
          <Icon kind="qr" /> QR
        </button>
        <a className="pg-btn pg-btn-ghost" href={reviewMailto}>
          Request manual review
        </a>
      </div>

      {qrOpen && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 16,
            padding: 14,
            background: "var(--bg-1)",
            border: "1px solid var(--line)",
            borderRadius: 6,
            alignSelf: "flex-end",
          }}
        >
          <div
            style={{
              padding: 8,
              background: "var(--ink-0)",
              borderRadius: 4,
            }}
          >
            <QRCodeSVG value={certUrl} size={120} bgColor="#f4f4f5" fgColor="#0a0a0a" level="M" />
          </div>
          <div
            style={{ display: "flex", flexDirection: "column", gap: 4, fontSize: 11 }}
          >
            <span
              className="mono"
              style={{
                fontSize: 10,
                color: "var(--ink-3)",
                letterSpacing: "0.16em",
                textTransform: "uppercase",
              }}
            >
              Scan to verify
            </span>
            <span style={{ color: "var(--ink-2)" }}>
              QR encodes the canonical cert URL.
            </span>
            <span
              className="mono"
              style={{ fontSize: 10, color: "var(--ink-4)", marginTop: 4 }}
            >
              {certUrl}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

function Icon({ kind }: { kind: "share" | "link" | "copy" | "print" | "image" | "qr" }) {
  const paths: Record<typeof kind, string> = {
    share: "M4 12v7h12v-7 M10 4l3-3 3 3 M13 1v11",
    link:
      "M9 13a4 4 0 0 0 5.66 0l3-3a4 4 0 0 0-5.66-5.66l-1 1 M11 7a4 4 0 0 0-5.66 0l-3 3a4 4 0 1 0 5.66 5.66l1-1",
    copy: "M7 4h9v12H7z M4 8v9h9",
    print: "M5 7V3h10v4 M5 14h10v6H5z M3 7h14v7H3z M16 10h.01",
    image: "M3 3h14v14H3z M3 13l4-4 4 4 3-3 3 3 M11 7a1 1 0 1 1-2 0 1 1 0 0 1 2 0z",
    qr: "M3 3h5v5H3z M12 3h5v5h-5z M3 12h5v5H3z M12 12h2v2h-2z M16 12h1v1 M12 16h1v1 M16 16h1v1",
  } as const;
  return (
    <svg
      width="13"
      height="13"
      viewBox="0 0 20 20"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d={paths[kind]} />
    </svg>
  );
}

/**
 * Inline cert-id pill — copy on click, mono font, low-contrast surface.
 * Re-exported from the same client component because it shares the
 * copy-toast machinery.
 */
export function CertIdPill({ certId }: { certId: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(certId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  };
  return (
    <button
      onClick={onCopy}
      title="Copy cert ID"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "5px 10px",
        borderRadius: 4,
        background: "var(--bg-2)",
        border: "1px solid var(--line)",
        color: "inherit",
        fontFamily: "inherit",
        cursor: "pointer",
        transition: "border-color 120ms",
      }}
    >
      <span
        className="mono"
        style={{ fontSize: 11, color: "var(--ink-1)", letterSpacing: "0.04em" }}
      >
        {certId}
      </span>
      <span
        className="mono"
        style={{
          fontSize: 9,
          color: copied ? "var(--lime)" : "var(--ink-4)",
          letterSpacing: "0.1em",
          textTransform: "uppercase",
        }}
      >
        {copied ? "COPIED" : "COPY"}
      </span>
    </button>
  );
}
