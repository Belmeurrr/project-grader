/**
 * Tests for <CertShareCard>.
 *
 * Covers the three behaviors most likely to regress:
 *   1. The QR code renders, with the canonical cert URL as its value
 *      attribute (so the QR genuinely encodes the right link).
 *   2. The "copy cert URL" button calls `navigator.clipboard.writeText`.
 *   3. The "Share" button calls `navigator.share` when present, and
 *      otherwise opens the dropdown with Twitter/Email/Copy options.
 *
 * We don't try to scan the rendered SVG — that's a job for an
 * integration test. Pinning the `value` prop on the underlying
 * <QRCodeSVG> is enough to catch the most likely regression (someone
 * passes the bare cert ID instead of the canonical URL).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, cleanup } from "@testing-library/react";

// Mock qrcode.react to a simple <svg data-value=...> so we can assert
// the value prop without running the real QR encoder. Keeps the test
// fast + deterministic + free of canvas/SVG arithmetic.
vi.mock("qrcode.react", () => ({
  QRCodeSVG: ({ value }: { value: string }) => (
    <svg data-testid="qr" data-value={value} />
  ),
}));

import CertShareCard from "@/components/cert/CertShareCard";

const PROPS = {
  certId: "cert_abc123",
  certUrl: "https://grader.app/cert/cert_abc123",
  cardName: "Charizard 1st Edition",
  finalGradeLabel: "8.5",
};

describe("<CertShareCard>", () => {
  let writeTextMock: ReturnType<typeof vi.fn>;
  let originalClipboard: typeof navigator.clipboard | undefined;
  let originalShare: typeof navigator.share | undefined;

  beforeEach(() => {
    writeTextMock = vi.fn(async () => undefined);
    originalClipboard = navigator.clipboard;
    originalShare = (navigator as Navigator & { share?: typeof navigator.share })
      .share;
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText: writeTextMock },
    });
  });

  afterEach(() => {
    // Vitest with `globals: false` doesn't auto-wire RTL's afterEach
    // cleanup, so previous renders stick around in the DOM and turn
    // `getByRole` into multiple-match errors. Call cleanup explicitly.
    cleanup();
    if (originalClipboard !== undefined) {
      Object.defineProperty(navigator, "clipboard", {
        configurable: true,
        value: originalClipboard,
      });
    }
    if (originalShare !== undefined) {
      Object.defineProperty(navigator, "share", {
        configurable: true,
        value: originalShare,
      });
    } else {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      delete (navigator as unknown as { share?: unknown }).share;
    }
    vi.restoreAllMocks();
  });

  it("renders the QR with the canonical cert URL", () => {
    render(<CertShareCard {...PROPS} />);
    const qr = screen.getByTestId("qr");
    expect(qr.getAttribute("data-value")).toBe(PROPS.certUrl);
  });

  it("renders the cert ID and copies it on click", async () => {
    render(<CertShareCard {...PROPS} />);
    const idButton = screen.getByRole("button", { name: /copy cert id/i });
    expect(idButton.textContent).toContain(PROPS.certId);
    fireEvent.click(idButton);
    // Microtask boundary — copyText awaits the clipboard write.
    await Promise.resolve();
    expect(writeTextMock).toHaveBeenCalledWith(PROPS.certId);
  });

  it("copies the URL via the Copy cert URL button", async () => {
    render(<CertShareCard {...PROPS} />);
    fireEvent.click(screen.getByRole("button", { name: /copy cert url/i }));
    await Promise.resolve();
    expect(writeTextMock).toHaveBeenCalledWith(PROPS.certUrl);
  });

  it("calls navigator.share when present", async () => {
    const shareMock = vi.fn(async () => undefined);
    Object.defineProperty(navigator, "share", {
      configurable: true,
      value: shareMock,
    });
    render(<CertShareCard {...PROPS} />);
    fireEvent.click(screen.getByRole("button", { name: /^share$/i }));
    await Promise.resolve();
    await Promise.resolve();
    expect(shareMock).toHaveBeenCalledTimes(1);
    const calls = shareMock.mock.calls as unknown as Array<[ShareData]>;
    const arg = calls[0]?.[0];
    expect(arg).toBeDefined();
    expect(arg!.url).toBe(PROPS.certUrl);
    expect(arg!.text).toContain(PROPS.cardName);
    expect(arg!.text).toContain(PROPS.finalGradeLabel);
  });

  it("opens the fallback dropdown when navigator.share is missing", async () => {
    // Simulate a browser without Web Share API.
    delete (navigator as unknown as { share?: unknown }).share;
    render(<CertShareCard {...PROPS} />);
    expect(screen.queryByRole("menu")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: /^share$/i }));
    // Dropdown should open with the three menu items.
    const menu = await screen.findByRole("menu");
    expect(menu).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /twitter/i })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /email|discord/i })).toBeInTheDocument();
    expect(screen.getByRole("menuitem", { name: /copy link/i })).toBeInTheDocument();
  });
});
