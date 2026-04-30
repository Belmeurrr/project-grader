import Link from "next/link";

export default function CertNotFound() {
  return (
    <main className="mx-auto flex max-w-2xl flex-col gap-6 px-6 py-20 text-center">
      <p className="text-xs uppercase tracking-widest text-lime-500">
        Project Grader • Certificate
      </p>
      <h1 className="text-3xl font-semibold">Cert not found</h1>
      <p className="text-zinc-400">
        Either this certificate ID doesn’t exist, or the submission isn’t
        complete yet. If you just submitted a card, give the pipeline a
        moment and refresh.
      </p>
      <Link
        href="/"
        className="mx-auto rounded-xl border border-zinc-700 px-4 py-2 text-sm text-zinc-200 hover:border-zinc-500"
      >
        ← Back to home
      </Link>
    </main>
  );
}
