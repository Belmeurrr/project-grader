"use client";

/**
 * QR for the print one-pager. Same QRCodeSVG used in <CertShareCard>,
 * but sized larger (160px) and rendered black-on-white for monochrome
 * printers. Wrapped in its own client component because the `qrcode.react`
 * library is client-only (uses canvas/svg APIs).
 */
import { QRCodeSVG } from "qrcode.react";

export default function PrintQR({ url }: { url: string }) {
  return (
    <div className="shrink-0 border border-black bg-white p-2">
      <QRCodeSVG
        value={url}
        size={140}
        level="M"
        bgColor="#ffffff"
        fgColor="#000000"
        aria-label="QR code linking to cert URL"
      />
    </div>
  );
}
