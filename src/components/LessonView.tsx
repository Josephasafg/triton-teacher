import { useParams, useNavigate } from 'react-router-dom';
import { curriculum } from '../content/curriculum';
import { useProgress } from '../store/progress';
import { CodePlayground } from './CodePlayground';
import { toRoman } from '../utils/roman';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Check } from 'lucide-react';

export function LessonView() {
  const { moduleId, lessonId } = useParams();
  const navigate = useNavigate();
  const { completeLesson, isLessonComplete } = useProgress();

  const mod = curriculum.find((m) => m.id === moduleId);
  const lesson = mod?.lessons.find((l) => l.id === lessonId);

  if (!mod || !lesson) {
    return (
      <div className="flex h-full items-center justify-center">
        <p className="font-display italic text-parchment-mute">
          The requested lesson could not be found.
        </p>
      </div>
    );
  }

  const modIdx = curriculum.findIndex((m) => m.id === mod.id);
  const lessonIdx = mod.lessons.findIndex((l) => l.id === lessonId);
  const completed = isLessonComplete(lesson.id);

  const prev = lessonIdx > 0 ? mod.lessons[lessonIdx - 1] : null;
  const next = lessonIdx < mod.lessons.length - 1 ? mod.lessons[lessonIdx + 1] : null;
  const nextIsPuzzle = !next && mod.puzzles.length > 0;

  return (
    <div className="flex min-h-full flex-col lg:h-full">
      {/* Running head */}
      <header className="flex items-center justify-between gap-3 border-b border-wine-glow/40 bg-wine-deep/40 py-3 pl-16 pr-4 md:px-8 md:py-4 lg:pl-10">
        <div className="flex min-w-0 items-baseline gap-2 font-sans text-[10.5px] uppercase tracking-widest-caps text-parchment-mute md:gap-4">
          <button onClick={() => navigate('/')} className="shrink-0 hover:text-parchment">
            Home
          </button>
          <span className="text-copper">◆</span>
          <span className="min-w-0 truncate">
            <span className="hidden sm:inline">Ch. {toRoman(modIdx + 1)} &nbsp;·&nbsp;{' '}</span>
            <span className="font-serif normal-case tracking-normal text-parchment-dim">
              {mod.title}
            </span>
          </span>
        </div>

        <button
          onClick={() => completeLesson(lesson.id)}
          className={`shrink-0 ${completed ? 'btn-secondary !border-sage/40 !text-sage' : 'btn-secondary'}`}
        >
          {completed ? (
            <>
              <Check size={12} /> Read
            </>
          ) : (
            <>Mark read</>
          )}
        </button>
      </header>

      {/* Two-column reading spread */}
      <div className="flex min-h-0 flex-1 flex-col lg:flex-row">
        {/* Text column */}
        <div className="w-full border-b border-wine-glow/40 lg:w-[55%] lg:overflow-y-auto lg:border-b-0 lg:border-r">
          <article className="mx-auto max-w-[620px] px-6 py-10 md:px-10 md:py-14 lg:px-14 lg:py-16">
            <div className="mb-8 md:mb-10">
              <div className="eyebrow mb-4">
                § {toRoman(modIdx + 1)}.{toRoman(lessonIdx + 1)} &nbsp;·&nbsp;{' '}
                Lesson {lessonIdx + 1} of {mod.lessons.length}
              </div>
              <h1 className="font-display text-[32px] font-semibold leading-[1.05] text-parchment-ink md:text-[38px] lg:text-[44px]"
                  style={{ fontVariationSettings: "'opsz' 72, 'SOFT' 100", letterSpacing: '-0.015em' }}>
                {lesson.title}
              </h1>
              <div className="mt-6 h-px w-20 bg-copper" />
            </div>

            <div className="prose-editorial drop-cap">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {lesson.content}
              </ReactMarkdown>
            </div>

            {/* Continue navigation — desktop: inline at bottom of text column */}
            <div className="mt-16 hidden border-t border-wine-glow/30 pt-8 lg:block">
              <LessonNav
                prev={prev}
                next={next}
                nextIsPuzzle={nextIsPuzzle}
                moduleId={mod.id}
                puzzles={mod.puzzles}
                onNavigate={navigate}
              />
            </div>
          </article>
        </div>

        {/* Code lab column */}
        <div className="flex w-full flex-col p-4 md:p-5 lg:w-[45%] lg:overflow-hidden">
          <div className="mb-3">
            <div className="eyebrow">The Laboratory</div>
          </div>
          <div className="h-[480px] lg:h-[calc(100%-32px)]">
            <CodePlayground
              key={lesson.id}
              initialCode={lesson.code}
              storageKey={lesson.id}
              readOnly={false}
            />
          </div>
        </div>
      </div>

      {/* Mobile navigation footer */}
      <div className="border-t border-wine-glow/30 bg-wine-deep/40 px-6 py-8 lg:hidden">
        <LessonNav
          prev={prev}
          next={next}
          nextIsPuzzle={nextIsPuzzle}
          moduleId={mod.id}
          puzzles={mod.puzzles}
          onNavigate={navigate}
        />
      </div>
    </div>
  );
}

type NavLesson = { id: string; title: string };
type NavPuzzle = { id: string; title: string };

function LessonNav({
  prev,
  next,
  nextIsPuzzle,
  moduleId,
  puzzles,
  onNavigate,
}: {
  prev: NavLesson | null;
  next: NavLesson | null;
  nextIsPuzzle: boolean;
  moduleId: string;
  puzzles: NavPuzzle[];
  onNavigate: (path: string) => void;
}) {
  return (
    <div className="flex items-center justify-between gap-4">
      {prev ? (
        <button
          onClick={() => onNavigate(`/module/${moduleId}/lesson/${prev.id}`)}
          className="group text-left"
        >
          <div className="eyebrow mb-1">Previous</div>
          <div className="font-display text-[17px] text-parchment/80 transition-colors group-hover:text-gold">
            <span className="mr-2 inline-block font-ital italic text-copper transition-transform group-hover:-translate-x-1">
              ←
            </span>
            {prev.title}
          </div>
        </button>
      ) : (
        <div />
      )}

      {next ? (
        <button
          onClick={() => onNavigate(`/module/${moduleId}/lesson/${next.id}`)}
          className="group text-right"
        >
          <div className="eyebrow mb-1">Next</div>
          <div className="font-display text-[17px] text-parchment/80 transition-colors group-hover:text-gold">
            {next.title}
            <span className="ml-2 inline-block font-ital italic text-copper transition-transform group-hover:translate-x-1">
              →
            </span>
          </div>
        </button>
      ) : nextIsPuzzle ? (
        <button
          onClick={() => onNavigate(`/module/${moduleId}/puzzle/${puzzles[0].id}`)}
          className="group text-right"
        >
          <div className="eyebrow mb-1">Exercise</div>
          <div className="font-display text-[17px] italic text-parchment/80 transition-colors group-hover:text-gold">
            {puzzles[0].title.replace(/^Puzzle:\s*/, '')}
            <span className="ml-2 inline-block font-ital not-italic text-copper transition-transform group-hover:translate-x-1">
              →
            </span>
          </div>
        </button>
      ) : (
        <button
          onClick={() => onNavigate(`/module/${moduleId}/quiz`)}
          className="group text-right"
        >
          <div className="eyebrow mb-1">Examination</div>
          <div className="font-display text-[17px] text-parchment/80 transition-colors group-hover:text-gold">
            Take the exam
            <span className="ml-2 inline-block font-ital text-copper transition-transform group-hover:translate-x-1">
              →
            </span>
          </div>
        </button>
      )}
    </div>
  );
}
