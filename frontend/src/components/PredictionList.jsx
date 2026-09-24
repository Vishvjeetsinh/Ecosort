import React, { useCallback, useMemo, useRef } from 'react';
import { formatPercent } from '../lib/format.js';
import { categoryById } from '../lib/categories.js';
import PredictionRow from './PredictionRow.jsx';

function WarningIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-5 w-5 shrink-0" fill="currentColor">
      <path d="M12 2.8 1.6 20.2h20.8L12 2.8Zm0 5.6a1 1 0 0 1 1 1v4.8a1 1 0 1 1-2 0V9.4a1 1 0 0 1 1-1Zm0 8.2a1.2 1.2 0 1 1 0 2.4 1.2 1.2 0 0 1 0-2.4Z" />
    </svg>
  );
}

/**
 * The top-3 predictions as an ARIA radiogroup: exactly one category drives the
 * guidance card, and arrow keys move focus and selection together the way radios do.
 */
export function PredictionList({
  predictions,
  categories,
  selectedId,
  onSelect,
  lowConfidence = false,
  unmatchedMass = 0,
  engineKind = 'fallback',
}) {
  const items = Array.isArray(predictions) ? predictions : [];
  const buttonRefs = useRef([]);

  const activeIndex = useMemo(() => {
    const i = items.findIndex((p) => p.category === selectedId);
    return i >= 0 ? i : 0;
  }, [items, selectedId]);

  const moveTo = useCallback(
    (nextIndex) => {
      if (items.length === 0) return;
      const wrapped = (nextIndex + items.length) % items.length;
      const target = items[wrapped];
      if (target) onSelect?.(target.category);
      const node = buttonRefs.current[wrapped];
      if (node) node.focus();
    },
    [items, onSelect],
  );

  const handleKeyDown = useCallback(
    (event, index) => {
      switch (event.key) {
        case 'ArrowDown':
        case 'ArrowRight':
          event.preventDefault();
          moveTo(index + 1);
          break;
        case 'ArrowUp':
        case 'ArrowLeft':
          event.preventDefault();
          moveTo(index - 1);
          break;
        case 'Home':
          event.preventDefault();
          moveTo(0);
          break;
        case 'End':
          event.preventDefault();
          moveTo(items.length - 1);
          break;
        default:
          break;
      }
    },
    [items.length, moveTo],
  );

  if (items.length === 0) return null;

  const unmatched = Number.isFinite(Number(unmatchedMass)) ? Number(unmatchedMass) : 0;
  const isFallback = engineKind === 'fallback';

  return (
    <section aria-labelledby="prediction-list-heading" className="space-y-3">
      <h3
        id="prediction-list-heading"
        className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400"
      >
        Top matches
      </h3>

      <div
        role="radiogroup"
        aria-labelledby="prediction-list-heading"
        aria-describedby={lowConfidence ? 'prediction-low-confidence' : undefined}
        className="space-y-2"
      >
        {items.map((prediction, index) => (
          <PredictionRow
            key={`${prediction.category}-${index}`}
            prediction={prediction}
            category={categoryById(categories, prediction.category)}
            rank={index + 1}
            selected={index === activeIndex}
            onSelect={onSelect}
            tabIndex={index === activeIndex ? 0 : -1}
            onKeyDown={(event) => handleKeyDown(event, index)}
            elementRef={(el) => {
              buttonRefs.current[index] = el;
            }}
          />
        ))}
      </div>

      {lowConfidence ? (
        <div
          id="prediction-low-confidence"
          role="note"
          className="flex gap-3 rounded-xl border border-amber-300 bg-amber-50 p-3 text-amber-900 dark:border-amber-500/60 dark:bg-amber-500/10 dark:text-amber-100"
        >
          <WarningIcon />
          <div className="space-y-2 text-sm leading-6">
            <p className="font-semibold">Low confidence — treat this as a guess.</p>
            {isFallback ? (
              <p>
                No purpose-trained model is installed, so EcoSort is running the pretrained
                MobileNetV2 fallback: it recognises everyday ImageNet objects and maps them onto
                waste streams. That mapping is coarse by design and is weakest on packaging, mixed
                materials and close-ups.
              </p>
            ) : (
              <p>
                The trained model found no category it is sure about. The item may sit outside the
                categories it was taught, or the shot may be hard to read.
              </p>
            )}
            <ul className="list-disc space-y-1 pl-5">
              <li>Retake the photo against a plain background, with the item well lit.</li>
              <li>Fill the frame with the single item — no hands, no clutter, no other objects.</li>
              <li>
                {isFallback
                  ? 'For dependable results, train a model on your own photos: make train.'
                  : 'If this keeps happening for an item you care about, add examples of it and retrain: make train.'}
              </li>
            </ul>
            <p>
              You can still correct the result below — pick the right category and EcoSort stores
              the correction with this classification.
            </p>
          </div>
        </div>
      ) : null}

      {unmatched > 0.5 ? (
        <p className="rounded-xl border border-slate-200 bg-slate-50 p-3 text-sm leading-6 text-slate-700 dark:border-slate-700 dark:bg-slate-800/60 dark:text-slate-200">
          Most of what the model saw — {formatPercent(unmatched)} of its probability — is not a
          recognised waste item at all. Expect the ranking above to be unreliable until the item
          fills the frame on its own.
        </p>
      ) : null}
    </section>
  );
}

export default PredictionList;
