import { useNavigate } from 'react-router-dom';

export function About() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen">
      {/* Running head */}
      <header className="flex items-center justify-between gap-3 border-b border-wine-glow/40 bg-wine-deep/40 py-3 pl-16 pr-4 md:px-8 md:py-4 lg:pl-10">
        <div className="flex min-w-0 items-baseline gap-2 font-sans text-[10.5px] uppercase tracking-widest-caps text-parchment-mute md:gap-4">
          <button onClick={() => navigate('/')} className="shrink-0 hover:text-parchment">
            Home
          </button>
          <span className="text-copper">◆</span>
          <span className="truncate">About the Author</span>
        </div>
      </header>

      <article className="mx-auto max-w-[760px] px-6 py-12 md:px-10 md:py-16 lg:px-12 lg:py-20">
        <div className="eyebrow mb-5">§ Biographical Note</div>

        <h1
          className="font-display text-[52px] font-semibold leading-[0.95] text-parchment-ink sm:text-[64px] md:text-[72px] lg:text-[80px]"
          style={{
            fontVariationSettings: "'opsz' 144, 'SOFT' 100",
            letterSpacing: '-0.02em',
          }}
        >
          Who am <span className="font-ital italic font-normal text-gold">I</span>
          <span className="text-copper">?</span>
        </h1>

        <p className="mt-6 font-ital text-[18px] italic leading-snug text-parchment-dim md:text-[22px]">
          A short note from the person who wrote this field guide.
        </p>

        <div className="mt-10 h-px w-20 bg-copper" />

        <div className="prose-editorial drop-cap mt-10">
          <p>
            My name is <strong>Asaf Gardin</strong>. I'm an{' '}
            <em>Inference Engineer</em> at <em>AI21 Labs</em>. I'm endlessly
            curious, and the kind of person who likes to learn new things
            down to their core.
          </p>

          <p>
            I built Kernel Academy because learning Triton, frankly, was never
            particularly convenient. The reference material is excellent if
            you already know what you're looking at: official tutorials, dense
            research papers, scattered notebooks, production kernels buried in
            a dozen repositories. None of it adds up to a place where a curious
            engineer can sit down and <em>just start writing kernels</em>,{' '}
            watch them execute, and build intuition chapter by chapter.
          </p>

          <h2>So I made one.</h2>

          <p>
            Kernel Academy is the resource I wish had existed when I started.
            It's opinionated: it begins with program IDs and masks and works
            its way up to the patterns that actually show up in modern LLM
            inference: fused activations, online softmax, quantised
            dequantisation. Every example runs in the browser on a NumPy-backed
            simulator, so you can learn without owning a GPU.
          </p>

          <blockquote>
            The best way to learn a systems language is to write one small,
            correct thing, and then make it faster.
          </blockquote>

          <h2>If you'd like to reach me</h2>

          <p>
            Corrections and pull requests are welcome. If something here is
            wrong, unclear, or could be explained better, please tell me.
          </p>
        </div>

        {/* Colophon / links */}
        <div className="mt-20 border-t border-wine-glow/40 pt-10">
          <div className="grid grid-cols-2 gap-10 md:grid-cols-3">
            <ColophonCol label="LinkedIn">
              <a
                href="https://www.linkedin.com/in/joseph-asaf-gardin/"
                target="_blank"
                rel="noreferrer"
                className="font-display text-[16px] text-parchment underline decoration-copper/40
                           underline-offset-4 transition-colors hover:text-gold hover:decoration-gold/70"
              >
                Asaf Gardin
              </a>
            </ColophonCol>

            <ColophonCol label="GitHub">
              <a
                href="https://github.com/Josephasafg"
                target="_blank"
                rel="noreferrer"
                className="font-display text-[16px] text-parchment underline decoration-copper/40
                           underline-offset-4 transition-colors hover:text-gold hover:decoration-gold/70"
              >
                @Josephasafg
              </a>
            </ColophonCol>

            <ColophonCol label="Correspondence">
              <a
                href="mailto:asaf.j.gardin@gmail.com"
                className="font-display text-[16px] text-parchment underline decoration-copper/40
                           underline-offset-4 transition-colors hover:text-gold hover:decoration-gold/70"
              >
                asaf.j.gardin@gmail.com
              </a>
            </ColophonCol>
          </div>
        </div>

        {/* Return to contents */}
        <div className="mt-16">
          <button
            onClick={() => navigate('/')}
            className="group flex items-baseline gap-3 font-display text-[18px] text-parchment-dim transition-colors hover:text-gold"
          >
            <span className="font-ital italic text-copper transition-transform group-hover:-translate-x-1">
              ←
            </span>
            Return to the contents
          </button>
        </div>
      </article>
    </div>
  );
}

function ColophonCol({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="eyebrow mb-2">{label}</div>
      {children}
    </div>
  );
}
