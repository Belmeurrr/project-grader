"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

type Route = {
  href: string;
  label: string;
  /** Match against pathname prefix; undefined → exact match on href */
  match?: string;
  disabled?: boolean;
};

const ROUTES: Route[] = [
  { href: "/", label: "Workbench" },
  { href: "/grade", label: "Capture", match: "/grade" },
  { href: "#", label: "Cert", disabled: true },
  { href: "#", label: "Card", disabled: true },
  { href: "#", label: "Queue", disabled: true },
  { href: "#", label: "Insights", disabled: true },
  { href: "#", label: "System", disabled: true },
];

function NavLink({ route, pathname }: { route: Route; pathname: string }) {
  const active = route.match
    ? pathname.startsWith(route.match)
    : pathname === route.href;
  const className = `pg-nav-link${active ? " active" : ""}${route.disabled ? " disabled" : ""}`;

  if (route.disabled) {
    return (
      <span className={className} title="Mock pending">
        {route.label}
        <span
          className="mono"
          style={{
            fontSize: 8,
            color: "var(--ink-4)",
            marginLeft: 4,
            letterSpacing: "0.1em",
          }}
        >
          SOON
        </span>
      </span>
    );
  }

  return (
    <Link href={route.href} className={className}>
      {route.label}
    </Link>
  );
}

export default function Nav({ subtitle = "The Workbench" }: { subtitle?: string }) {
  const pathname = usePathname();
  // Hide on print routes — the print one-pager is meant to be a clean
  // standalone artifact (no app chrome) when users open it for the
  // browser's print dialog.
  if (pathname.endsWith("/print")) return null;
  return (
    <nav className="pg-nav">
      <div className="pg-nav-brand">
        <Link href="/" style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span className="pg-nav-mark">◰</span>
          <span style={{ display: "flex", flexDirection: "column", lineHeight: 1.1 }}>
            <span className="pg-nav-title">Project Grader</span>
            <span className="pg-nav-sub">{subtitle}</span>
          </span>
        </Link>
      </div>
      <div className="pg-nav-links">
        {ROUTES.map((r) => (
          <NavLink key={r.label} route={r} pathname={pathname} />
        ))}
      </div>
      <div className="pg-nav-actions">
        <span
          className="mono"
          style={{ fontSize: 10, color: "var(--ink-3)" }}
        >
          belme · personal
        </span>
        <span
          style={{
            width: 6,
            height: 6,
            borderRadius: 3,
            background: "var(--lime)",
            boxShadow: "0 0 6px var(--lime)",
          }}
        />
      </div>
    </nav>
  );
}
