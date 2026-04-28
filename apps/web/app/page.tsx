export default function HomePage() {
  return (
    <main className="mx-auto flex max-w-3xl flex-col gap-8 px-6 py-20">
      <header className="flex flex-col gap-3">
        <p className="text-sm uppercase tracking-widest text-lime-500">Project Grader</p>
        <h1 className="text-5xl font-semibold leading-tight">
          AI grading and counterfeit detection for trading cards.
        </h1>
        <p className="max-w-xl text-lg text-zinc-400">
          Higher accuracy than the 95% AI-grader ceiling, plus first-class authenticity verification.
          Pokemon, MTG, sports, and more.
        </p>
      </header>

      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <FeatureCard
          title="Per-criterion grading"
          body="Centering, corners, edges, surface — graded independently with calibrated uncertainty."
        />
        <FeatureCard
          title="Authenticity ensemble"
          body="Print-rosette FFT, holo parallax, color profile, typography, paper substrate."
        />
        <FeatureCard
          title="Multi-shot capture"
          body="Guidance system rejects bad captures rather than guessing."
        />
        <FeatureCard
          title="Public cert pages"
          body="Every grade gets a shareable URL with full evidence trail."
        />
      </section>

      <footer className="mt-12 text-sm text-zinc-500">v0.0.1 — scaffolding in progress.</footer>
    </main>
  );
}

function FeatureCard({ title, body }: { title: string; body: string }) {
  return (
    <article className="rounded-xl border border-zinc-800 p-5">
      <h3 className="mb-2 text-base font-medium">{title}</h3>
      <p className="text-sm text-zinc-400">{body}</p>
    </article>
  );
}
