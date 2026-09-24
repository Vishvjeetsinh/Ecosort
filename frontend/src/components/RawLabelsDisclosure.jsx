import React from 'react';
import { formatConfidence } from '../lib/format.js';
import ConfidenceBar from './ConfidenceBar.jsx';

/**
 * The fallback engine's own vocabulary, shown verbatim. Seeing "pop bottle, soda
 * bottle" at 61% explains a plastic verdict far better than the aggregated score does.
 */
export function RawLabelsDisclosure({ rawLabels, engineKind = 'fallback' }) {
  const labels = Array.isArray(rawLabels) ? rawLabels.filter(Boolean).slice(0, 10) : [];
  if (labels.length === 0) return null;

  return (
    <details className="card p-4">
      <summary className="cursor-pointer list-none text-sm font-semibold text-slate-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-500 focus-visible:ring-offset-2 dark:text-slate-100 dark:focus-visible:ring-offset-slate-900">
        <span aria-hidden="true" className="mr-2 inline-block text-slate-400">▸</span>
        What the model actually saw
      </summary>

      <p className="mt-3 text-xs leading-5 text-slate-500 dark:text-slate-400">
        These are raw ImageNet-1k classes — the 1000 everyday objects MobileNetV2 was trained on.
        EcoSort maps them onto waste streams, so a class it has no mapping for contributes nothing
        to the ranking above.
        {engineKind === 'fallback'
          ? ' The purpose-trained model is not installed, so this mapping is what produced your result.'
          : ' Your custom model produced the result above; these labels are shown for reference only.'}
      </p>

      <ol className="mt-3 space-y-2">
        {labels.map((entry, i) => (
          <li key={`${entry.index ?? i}-${entry.label}`} className="grid grid-cols-[1.5rem_1fr_3.5rem] items-center gap-2">
            <span className="text-xs tabular-nums text-slate-400 dark:text-slate-500">{i + 1}.</span>
            <div className="min-w-0">
              <span className="block truncate text-sm text-slate-700 dark:text-slate-200" title={entry.label}>
                {entry.label}
              </span>
              <div className="mt-1">
                <ConfidenceBar
                  value={entry.confidence}
                  colorHex="#64748b"
                  label={`${entry.label} probability`}
                  height={4}
                />
              </div>
            </div>
            <span className="text-right text-xs font-medium tabular-nums text-slate-600 dark:text-slate-300">
              {formatConfidence(entry.confidence)}
            </span>
          </li>
        ))}
      </ol>
    </details>
  );
}

export default RawLabelsDisclosure;
