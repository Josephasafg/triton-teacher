import { useParams, useNavigate } from 'react-router-dom';
import { curriculum } from '../content/curriculum';
import { useProgress } from '../store/progress';
import { CodePlayground } from './CodePlayground';
import { toRoman } from '../utils/roman';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useState, useCallback } from 'react';
import { Check } from 'lucide-react';

export function PuzzleView() {
  const { moduleId, puzzleId } = useParams();
  const navigate = useNavigate();
  const { solvePuzzle, isPuzzleSolved } = useProgress();

  const [showHints, setShowHints] = useState(false);
  const [hintLevel, setHintLevel] = useState(0);
  const [showSolution, setShowSolution] = useState(false);
  const [solved, setSolved] = useState(false);

  const mod = curriculum.find((m) => m.id === moduleId);
  const puzzle = mod?.puzzles.find((p) => p.id === puzzleId);

  const handleSuccess = useCallback(() => {
    setSolved(true);
    if (puzzle) solvePuzzle(puzzle.id);
  }, [puzzle, solvePuzzle]);

  if (!mod || !puzzle) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="font-display italic text-parchment-mute">
          The requested exercise could not be found.
        </p>
      </div>
    );
  }

  const modIdx = curriculum.findIndex((m) => m.id === mod.id);
  const puzzleIdx = mod.puzzles.findIndex((p) => p.id === puzzleId);
  const alreadySolved = isPuzzleSolved(puzzle.id);
  const nextPuzzle = puzzleIdx < mod.puzzles.length - 1 ? mod.puzzles[puzzleIdx + 1] : null;

  const diffTint: Record<string, string> = {
    easy: 'text-sage',
    medium: 'text-gold',
    hard: 'text-bordeaux',
  };

  const cleanTitle = puzzle.title.replace(/^Puzzle:\s*/, '');

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-wine-glow/40 bg-wine-deep/40 px-10 py-4">
        <div className="flex items-baseline gap-4 font-sans text-[10.5px] uppercase tracking-widest-caps text-parchment-mute">
          <button onClick={() => navigate('/')} className="hover:text-parchment">
            Kernel Academy
          </button>
          <span className="text-copper">◆</span>
          <span>
            Ch. {toRoman(modIdx + 1)} &nbsp;·&nbsp;{' '}
            <span className="font-serif normal-case tracking-normal text-parchment-dim">
              {mod.title}
            </span>
          </span>
        </div>

        <div className="flex items-center gap-4">
          <span className={`diff-pill ${diffTint[puzzle.difficulty]}`}>
            {puzzle.difficulty}
          </span>
          {(solved || alreadySolved) && (
            <span className="diff-pill flex items-center gap-1.5 text-sage">
              <Check size={11} /> Solved
            </span>
          )}
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        {/* Problem column */}
        <div className="w-[42%] overflow-y-auto border-r border-wine-glow/40">
          <div className="mx-auto max-w-[520px] px-12 py-14">
            <div className="eyebrow mb-4">
              Exercise {toRoman(puzzleIdx + 1)}
            </div>
            <h1 className="font-display text-[38px] font-semibold leading-[1.08] text-parchment-ink"
                style={{ fontVariationSettings: "'opsz' 48, 'SOFT' 100", letterSpacing: '-0.015em' }}>
              {cleanTitle}
            </h1>
            <div className="mt-5 h-px w-16 bg-copper" />

            <div className="prose-editorial mt-8">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {puzzle.description}
              </ReactMarkdown>
            </div>

            {/* Hints: styled as marginalia */}
            {puzzle.hints.length > 0 && (
              <div className="mt-10">
                <button
                  onClick={() => setShowHints(!showHints)}
                  className="btn-ghost !text-copper"
                >
                  {showHints ? 'Withdraw hints' : 'A hint, if needed'}
                </button>

                {showHints && (
                  <div className="mt-4 space-y-3 border-l border-copper/40 pl-5">
                    {puzzle.hints.slice(0, hintLevel + 1).map((hint, i) => (
                      <div key={i}>
                        <div className="eyebrow mb-1 !text-copper/80">
                          Hint {toRoman(i + 1)}
                        </div>
                        <p className="font-ital text-[15px] italic leading-relaxed text-parchment/85">
                          {hint}
                        </p>
                      </div>
                    ))}
                    {hintLevel < puzzle.hints.length - 1 && (
                      <button
                        onClick={() => setHintLevel((h) => h + 1)}
                        className="btn-ghost !text-copper/70"
                      >
                        Another hint →
                      </button>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Solution reveal */}
            <div className="mt-8">
              <button
                onClick={() => setShowSolution(!showSolution)}
                className="btn-ghost !text-parchment-mute hover:!text-bordeaux"
              >
                {showSolution ? 'Hide the answer' : 'Reveal the answer'}
              </button>

              {showSolution && (
                <div className="mt-4 border border-wine-glow/50 bg-wine-deep/80 p-5">
                  <div className="eyebrow mb-3 !text-bordeaux">Answer Key</div>
                  <pre className="overflow-x-auto font-mono text-[12.5px] leading-relaxed text-parchment/90">
                    {puzzle.solution}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Code lab */}
        <div className="flex w-[58%] flex-col overflow-hidden p-5">
          <div className="mb-3 flex items-baseline justify-between">
            <div className="eyebrow">Your Attempt</div>
            <span className="font-sans text-[11px] text-parchment-mute">
              Running the kernel executes the tests below.
            </span>
          </div>
          <div className="flex-1 min-h-0">
            <CodePlayground
              key={puzzle.id}
              initialCode={puzzle.starterCode}
              storageKey={puzzle.id}
              testCode={puzzle.testCode}
              onSuccess={handleSuccess}
            />
          </div>
        </div>
      </div>

      {/* Solved banner */}
      {(solved || alreadySolved) && (
        <div className="border-t border-sage/30 bg-sage/[0.08] px-10 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-baseline gap-3 font-display text-[16px] text-sage">
              <span className="font-ital italic text-[18px]">◆</span>
              <em className="italic">Quod erat demonstrandum.</em>
              <span className="text-parchment-dim">The exercise is solved.</span>
            </div>
            {nextPuzzle ? (
              <button
                onClick={() =>
                  navigate(`/module/${mod.id}/puzzle/${nextPuzzle.id}`)
                }
                className="btn-primary !border-sage/40 !text-sage hover:!text-gold"
              >
                Next Exercise →
              </button>
            ) : (
              <button
                onClick={() => navigate(`/module/${mod.id}/quiz`)}
                className="btn-primary !border-sage/40 !text-sage hover:!text-gold"
              >
                To the Examination →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
