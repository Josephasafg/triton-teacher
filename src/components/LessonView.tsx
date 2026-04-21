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
    <div className="flex h-full flex-col">
      {/* Running head */}
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

        <button
          onClick={() => completeLesson(lesson.id)}
          className={completed ? 'btn-secondary !border-sage/40 !text-sage' : 'btn-secondary'}
        >
          {completed ? (
            <>
              <Check size={12} /> Read
            </>
          ) : (
            <>Mark as read</>
          )}
        </button>
      </header>

      {/* Two-column reading spread */}
      <div className="flex flex-1 min-h-0">
        {/* Text column */}
        <div className="w-[55%] overflow-y-auto border-r border-wine-glow/40">
          <article className="mx-auto max-w-[620px] px-14 py-16">
            <div className="mb-10">
              <div className="eyebrow mb-4">
                § {toRoman(modIdx + 1)}.{toRoman(lessonIdx + 1)} &nbsp;·&nbsp;{' '}
                Lesson {lessonIdx + 1} of {mod.lessons.length}
              </div>
              <h1 className="font-display text-[44px] font-semibold leading-[1.05] text-parchment-ink"
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

            {/* Continue navigation at end of text */}
            <div className="mt-16 border-t border-wine-glow/30 pt-8">
              <div className="flex items-center justify-between gap-4">
                {prev ? (
                  <button
                    onClick={() => navigate(`/module/${mod.id}/lesson/${prev.id}`)}
                    className="group text-left"
                  >
                    <div className="eyebrow mb-1">Previous</div>
                    <div className="font-display text-[17px] text-parchment/80 transition-colors group-hover:text-gold">
                      <span className="mr-2 font-ital italic text-copper
                                        transition-transform group-hover:-translate-x-1 inline-block">
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
                    onClick={() => navigate(`/module/${mod.id}/lesson/${next.id}`)}
                    className="group text-right"
                  >
                    <div className="eyebrow mb-1">Next</div>
                    <div className="font-display text-[17px] text-parchment/80 transition-colors group-hover:text-gold">
                      {next.title}
                      <span className="ml-2 font-ital italic text-copper
                                         transition-transform group-hover:translate-x-1 inline-block">
                        →
                      </span>
                    </div>
                  </button>
                ) : nextIsPuzzle ? (
                  <button
                    onClick={() =>
                      navigate(`/module/${mod.id}/puzzle/${mod.puzzles[0].id}`)
                    }
                    className="group text-right"
                  >
                    <div className="eyebrow mb-1">Exercise</div>
                    <div className="font-display text-[17px] italic text-parchment/80 transition-colors group-hover:text-gold">
                      {mod.puzzles[0].title.replace(/^Puzzle:\s*/, '')}
                      <span className="ml-2 not-italic font-ital text-copper
                                         transition-transform group-hover:translate-x-1 inline-block">
                        →
                      </span>
                    </div>
                  </button>
                ) : (
                  <button
                    onClick={() => navigate(`/module/${mod.id}/quiz`)}
                    className="group text-right"
                  >
                    <div className="eyebrow mb-1">Examination</div>
                    <div className="font-display text-[17px] text-parchment/80 transition-colors group-hover:text-gold">
                      Take the exam
                      <span className="ml-2 font-ital text-copper
                                         transition-transform group-hover:translate-x-1 inline-block">
                        →
                      </span>
                    </div>
                  </button>
                )}
              </div>
            </div>
          </article>
        </div>

        {/* Code lab column */}
        <div className="w-[45%] overflow-hidden p-5">
          <div className="mb-3">
            <div className="eyebrow">The Laboratory</div>
          </div>
          <div className="h-[calc(100%-32px)]">
            <CodePlayground
              key={lesson.id}
              initialCode={lesson.code}
              storageKey={lesson.id}
              readOnly={false}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
