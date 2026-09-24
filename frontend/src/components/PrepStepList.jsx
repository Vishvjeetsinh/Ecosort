import React from 'react';

/** Numbered preparation steps ("Empty any liquid", "Rinse", ...). */
export function PrepStepList({ steps }) {
  const list = Array.isArray(steps) ? steps.filter((s) => typeof s === 'string' && s.trim()) : [];
  if (list.length === 0) return null;

  return (
    <ol className="space-y-2">
      {list.map((step, i) => (
        <li key={`${i}-${step}`} className="flex items-start gap-3">
          <span
            aria-hidden="true"
            className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full bg-slate-200 text-xs font-semibold tabular-nums text-slate-700 dark:bg-slate-700 dark:text-slate-100"
          >
            {i + 1}
          </span>
          <span className="text-sm leading-6 text-slate-700 dark:text-slate-200">{step}</span>
        </li>
      ))}
    </ol>
  );
}

export default PrepStepList;
