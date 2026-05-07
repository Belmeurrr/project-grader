/**
 * Print-friendly cert view at `/cert/[id]/print`.
 *
 * Auction houses + dealers want a one-pager they can staple to a
 * packing slip. This route re-renders the same cert data with a
 * monochrome-friendly layout, large QR + cert ID at the top, no nav,
 * and an auto-trigger of `window.print()` so opening the URL
 * immediately surfaces the browser print dialog.
 *
 * The auto-print bit is in a small client component (`<AutoPrint/>`).
 * The rest of the page is a server component so we can reuse the same
 * `fetchCertificate` call shape as the main `/cert/[id]` page.
 */
import { notFound } from "next/navigation";

import {
  fetchCertificate,
  type Certificate,
  type Grade,
} from "@/lib/cert";

import AutoPrint from "./AutoPrint";
import PrintQR from "./PrintQR";
import { resolvePublicBaseUrl } from "@/lib/publicUrl";

export const revalidate = 300;

type Params = { id: string };

export default async function CertPrintPage({
  params,
}: {
  params: Promise<Params>;
}) {
  const { id } = await params;
  const cert = await fetchCertificate(id);
  if (!cert) notFound();

  const baseUrl = resolvePublicBaseUrl();
  const certUrl = `${baseUrl}/cert/${id}`;
  const primary: Grade | undefined =
    cert.grades.find((g) => g.scheme === "psa") ?? cert.grades[0];
  const finalLabel =
    primary && primary.final !== null ? primary.final.toFixed(1) : "—";

  return (
    <main className="mx-auto max-w-3xl px-8 py-10 text-black print:px-4 print:py-2">
      <AutoPrint />
      {/* Print stylesheet — keep it inline so this page is self-contained
          and survives any future global CSS reshuffles. */}
      <style>{`
        @media print {
          body { background: white !important; color: black !important; }
          .no-print { display: none !important; }
          .print-page-break { page-break-after: always; }
          a { color: black !important; text-decoration: none !important; }
        }
        .cert-print-card {
          color: black;
          background: white;
        }
      `}</style>
      <div className="cert-print-card">
        <header className="flex flex-row items-start justify-between gap-6 border-b-2 border-black pb-4">
          <div>
            <p className="text-[11px] uppercase tracking-widest">
              Project Grader • Certificate
            </p>
            <h1 className="mt-1 break-all font-mono text-base">
              {cert.cert_id}
            </h1>
            <p className="mt-1 text-[11px]">
              Completed {new Date(cert.completed_at).toLocaleString()}
            </p>
            {cert.identified_card && (
              <p className="mt-2 text-lg font-semibold">
                {cert.identified_card.name}{" "}
                <span className="text-sm font-normal">
                  ({cert.identified_card.set_code} #
                  {cert.identified_card.card_number})
                </span>
              </p>
            )}
          </div>
          <PrintQR url={certUrl} />
        </header>

        <section className="mt-6 grid grid-cols-5 gap-2 text-center">
          <PrintGradeStat label="Centering" value={primary?.centering ?? null} />
          <PrintGradeStat label="Corners" value={primary?.corners ?? null} />
          <PrintGradeStat label="Edges" value={primary?.edges ?? null} />
          <PrintGradeStat label="Surface" value={primary?.surface ?? null} />
          <div className="rounded-md border-2 border-black p-2">
            <p className="text-[10px] uppercase tracking-wide">Final</p>
            <p className="text-3xl font-bold">{finalLabel}</p>
          </div>
        </section>

        <section className="mt-6 border-t border-black pt-4 text-sm">
          <h2 className="text-base font-semibold">Authenticity</h2>
          <p className="mt-1">
            Verdict:{" "}
            <span className="font-semibold uppercase">
              {cert.authenticity?.verdict ?? "unverified"}
            </span>
          </p>
          {cert.authenticity?.reasons && cert.authenticity.reasons.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-xs">
              {cert.authenticity.reasons.map((r, i) => (
                <li key={i}>{r}</li>
              ))}
            </ul>
          )}
        </section>

        <FooterPrint cert={cert} certUrl={certUrl} />
      </div>
    </main>
  );
}

function PrintGradeStat({
  label,
  value,
}: {
  label: string;
  value: number | null;
}) {
  return (
    <div className="rounded-md border border-black p-2">
      <p className="text-[10px] uppercase tracking-wide">{label}</p>
      <p className="text-2xl font-semibold">
        {value === null ? "—" : value.toFixed(1)}
      </p>
    </div>
  );
}

function FooterPrint({
  cert,
  certUrl,
}: {
  cert: Certificate;
  certUrl: string;
}) {
  return (
    <footer className="mt-8 border-t border-black pt-4 text-[11px]">
      <p>
        Verify online: <span className="font-mono">{certUrl}</span>
      </p>
      <p className="mt-1">
        Generated by Project Grader’s automated grading + counterfeit-
        detection pipeline. Not affiliated with PSA, BGS, SGC, or CGC.
      </p>
      {cert.authenticity?.model_versions && (
        <p className="mt-1 font-mono">
          {Object.entries(cert.authenticity.model_versions)
            .filter(([, v]) => typeof v === "string")
            .map(([k, v]) => `${k}=${v}`)
            .join(" • ")}
        </p>
      )}
    </footer>
  );
}

export async function generateMetadata({
  params,
}: {
  params: Promise<Params>;
}) {
  const { id } = await params;
  return {
    title: `Cert ${id.slice(0, 8)}… (Print) — Project Grader`,
    description: "Printable cert one-pager.",
    robots: { index: false, follow: false },
  };
}
