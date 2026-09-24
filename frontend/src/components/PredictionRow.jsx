import React from 'react';
import { formatConfidence } from '../lib/format.js';
import ConfidenceBar, { withAlpha } from './ConfidenceBar.jsx';

/**
 * One prediction inside PredictionList's radiogroup.
 *
 * `tabIndex` / `onKeyDown` / `elementRef` are supplied by the list so it can run
 * roving focus; App never renders this component directly.
 */
export function PredictionRow({
  prediction,
  category,
  rank,
  selected = false,
  onSelect,
  tabIndex = -1,
  onKeyDown,
  elementRef,
}) {
  if (!prediction) return null;

  const colorHex = category?.colorHex || '#64748b';
  const textColorHex = category?.textColorHex || '#ffffff';
  const categoryLabelText = category?.label || prediction.category || 'Unknown';
  const icon = category?.icon || categoryLabelText.slice(0, 1).toUpperCase();
  const confidence = Number.isFinite(Number(prediction.confidence)) ? Number(prediction.confidence) : 0;

  return (
    <button
      type="button"
      role="radio"
      aria-checked={selected}
      tabIndex={tabIndex}
      ref={elementRef}
      onClick={() => onSelect?.(prediction.category)}
      onKeyDown={onKeyDown}
      className={[
        'flex w-full items-center gap-3 rounded-xl border p-3 text-left motion-safe:transition-colors',
        selected
          ? 'border-transparent ring-2 ring-brand-500 dark:ring-brand-400'
          : 'border-slate-200 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:hover:border-slate-600 dark:hover:bg-slate-800/60',
      ].join(' ')}
      style={selected ? { backgroundColor: withAlpha(colorHex, 0.14) } : undefined}
    >
      <span className="w-6 shrink-0 text-xs font-semibold tabular-nums text-slate-400 dark:text-slate-500">
        #{rank}
      </span>

      <span
        aria-hidden="true"
        className="grid h-10 w-10 shrink-0 place-items-center rounded-lg text-lg font-bold"
        style={{ backgroundColor: colorHex, color: textColorHex }}
      >
        {icon}
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-sm font-semibold text-slate-900 dark:text-slate-50">
            {prediction.label || categoryLabelText}
          </span>
          <span className="shrink-0 text-sm font-semibold tabular-nums text-slate-700 dark:text-slate-200">
            {formatConfidence(confidence)}
          </span>
        </div>
        <div className="mt-2">
          <ConfidenceBar
            value={confidence}
            colorHex={colorHex}
            label={`${categoryLabelText} confidence`}
            height={8}
          />
        </div>
      </div>
    </button>
  );
}

export default PredictionRow;
