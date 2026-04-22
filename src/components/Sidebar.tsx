import { Link, useLocation, useNavigate } from 'react-router-dom';
import { curriculum } from '../content/curriculum';
import { useProgress } from '../store/progress';
import { toRoman } from '../utils/roman';
import { Check } from 'lucide-react';

export function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const { completedLessons, solvedPuzzles, quizScores } = useProgress();

  const totalLessons = curriculum.reduce((s, m) => s + m.lessons.length, 0);
  const totalPuzzles = curriculum.reduce((s, m) => s + m.puzzles.length, 0);
  const totalQuizzes = curriculum.length;
  const completedCount = completedLessons.length;
  const solvedCount = solvedPuzzles.length;
  const quizCount = Object.keys(quizScores).length;

  return (
    <aside className="flex h-full flex-col border-r border-wine-glow/40 bg-wine-deep/60 backdrop-blur-sm">
      {/* Masthead */}
      <Link
        to="/"
        className="block border-b border-wine-glow/40 px-7 pb-6 pt-8 transition-colors hover:bg-wine-glow/10"
      >
        <div className="eyebrow mb-2">MMXXVI</div>
        <h1 className="font-display text-[24px] font-semibold leading-none text-parchment-ink"
            style={{ fontVariationSettings: "'opsz' 48, 'SOFT' 100" }}>
          Kernel
          <br />
          <span className="font-ital text-[28px] font-normal text-gold">Academy</span>
        </h1>
        <p className="mt-3 font-sans text-[10px] uppercase tracking-caps text-copper/80">
          Triton for the Curious Engineer
        </p>
      </Link>

      {/* Ledger stats */}
      <div className="border-b border-wine-glow/40 px-7 py-5">
        <div className="eyebrow mb-3">Ledger</div>
        <div className="space-y-1.5 font-sans text-[12px]">
          <LedgerRow label="Tutorials" done={completedCount} total={totalLessons} />
          <LedgerRow label="Puzzles" done={solvedCount} total={totalPuzzles} />
          <LedgerRow label="Examinations" done={quizCount} total={totalQuizzes} />
        </div>
      </div>

      {/* Table of contents */}
      <nav className="flex-1 overflow-y-auto px-7 py-5">
        <div className="eyebrow mb-4">Table of Contents</div>
        <ol className="space-y-6">
          {curriculum.map((mod, idx) => {
            const moduleHasActive = location.pathname.includes(`/module/${mod.id}/`);
            return (
              <li key={mod.id}>
                <button
                  onClick={() =>
                    navigate(`/module/${mod.id}/lesson/${mod.lessons[0]?.id}`)
                  }
                  className={`group block w-full text-left ${
                    moduleHasActive ? 'text-parchment' : 'text-parchment/75'
                  }`}
                >
                  <div className="mb-1 flex items-baseline gap-3">
                    <span className={`font-display text-[11px] font-medium tracking-widest-caps ${
                      moduleHasActive ? 'text-copper' : 'text-copper/60'
                    }`}>
                      {toRoman(idx + 1)}
                    </span>
                    <span className="font-display text-[14.5px] font-semibold leading-tight
                                     transition-colors group-hover:text-gold"
                          style={{ fontVariationSettings: "'opsz' 18, 'SOFT' 60" }}>
                      {mod.title}
                    </span>
                  </div>
                </button>

                <ul className="mt-2 space-y-[5px] border-l border-wine-glow/30 pl-4">
                  {mod.lessons.map((lesson) => {
                    const path = `/module/${mod.id}/lesson/${lesson.id}`;
                    const isActive = location.pathname === path;
                    const isDone = completedLessons.includes(lesson.id);
                    return (
                      <li key={lesson.id}>
                        <button
                          onClick={() => navigate(path)}
                          className={`flex w-full items-center gap-2 py-0.5 pr-2 text-left font-serif text-[13px] leading-snug
                                      transition-colors ${
                                        isActive
                                          ? 'text-gold'
                                          : 'text-parchment/60 hover:text-parchment'
                                      }`}
                        >
                          {isDone && <Check size={10} className="shrink-0 text-sage" />}
                          <span className="truncate">{lesson.title}</span>
                        </button>
                      </li>
                    );
                  })}

                  {mod.puzzles.map((puzzle) => {
                    const path = `/module/${mod.id}/puzzle/${puzzle.id}`;
                    const isActive = location.pathname === path;
                    const isDone = solvedPuzzles.includes(puzzle.id);
                    return (
                      <li key={puzzle.id}>
                        <button
                          onClick={() => navigate(path)}
                          className={`flex w-full items-center gap-2 py-0.5 pr-2 text-left font-serif text-[13px] italic leading-snug
                                      transition-colors ${
                                        isActive
                                          ? 'text-gold'
                                          : 'text-parchment/55 hover:text-parchment'
                                      }`}
                        >
                          {isDone && <Check size={10} className="shrink-0 text-sage" />}
                          <span className="truncate">
                            <span className={`mr-1.5 not-italic ${isActive ? 'text-bordeaux' : 'text-bordeaux/70'}`}>◆</span>
                            {puzzle.title.replace(/^Puzzle:\s*/, '')}
                          </span>
                        </button>
                      </li>
                    );
                  })}

                  <li>
                    <button
                      onClick={() => navigate(`/module/${mod.id}/quiz`)}
                      className={`flex w-full items-center gap-2 py-0.5 text-left font-sans text-[10.5px] uppercase tracking-caps
                                  transition-colors ${
                                    location.pathname === `/module/${mod.id}/quiz`
                                      ? 'text-gold'
                                      : 'text-parchment-mute hover:text-parchment-dim'
                                  }`}
                    >
                      {quizScores[mod.id] !== undefined ? (
                        <Check size={10} className="shrink-0 text-sage" />
                      ) : null}
                      <span>Examination</span>
                      {quizScores[mod.id] !== undefined && (
                        <span className="ml-auto font-serif text-[11px] normal-case tracking-normal text-sage">
                          {quizScores[mod.id]}
                        </span>
                      )}
                    </button>
                  </li>
                </ul>
              </li>
            );
          })}
        </ol>
      </nav>

      {/* Colophon */}
      <div className="border-t border-wine-glow/40 px-7 py-5">
        <button
          onClick={() => navigate('/about')}
          className={`flex items-baseline gap-2 font-sans text-[10.5px] uppercase tracking-widest-caps
                      transition-colors ${
                        location.pathname === '/about'
                          ? 'text-gold'
                          : 'text-copper hover:text-gold'
                      }`}
        >
          <span>§</span> About the Author
        </button>
      </div>
    </aside>
  );
}

function LedgerRow({ label, done, total }: { label: string; done: number; total: number }) {
  return (
    <div className="flex items-baseline justify-between gap-2">
      <span className="text-parchment-mute">{label}</span>
      <span className="flex-1 mx-2 border-b border-dotted border-wine-glow/60" />
      <span className="font-display text-parchment">
        <span className="numeral-lining">{done}</span>
        <span className="mx-1 text-parchment-mute">/</span>
        <span className="numeral-lining text-parchment-dim">{total}</span>
      </span>
    </div>
  );
}
