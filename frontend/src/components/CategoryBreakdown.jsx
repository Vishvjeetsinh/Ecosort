import React from 'react';
import { formatCount, formatPercent, titleCase } from '../lib/format.js';
import { indexById, FALLBACK_CATEGORIES } from '../lib/categories.js';

function toCount(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

/**
 * Share of every waste category in the current window: one stacked bar for the mix,
 * then a ranked list for the exact numbers.
 */
export function CategoryBreakdown({ byCategory, total, categories }) {
  const catalogue = Array.isArray(categories) && categories.length ? categories : FALLBACK_CATEGORIES;
  const byId = indexById(catalogue);

  const rows = (Array.isArray(byCategory) ? byCategory : [])
    .filter((row) => row && typeof row === 'object' && row.category)
    .map((row) => {
      const meta = byId[row.category] ?? null;
      const count = toCount(row.count);
      return {
        id: row.category,
        label: row.label || meta?.label || titleCase(String(row.category)),
        icon: meta?.icon || '♻️',
        colorHex: row.colorHex || meta?.colorHex || '#64748b',
        textColorHex: meta?.textColorHex || '#ffffff',
        count,
        share: Number.isFinite(Number(row.share)) ? Number(row.share) : null,
      };
    });

  const summed = rows.reduce((sum, row) => sum + row.count, 0);
  // Prefer the server's `total`; fall back to the row sum so the bar is never blank
  // just because a caller passed the counts without the total.
  const denominator = toCount(total) || summed;

  const shareOf = (row) => {
    if (row.share !== null) return row.share;
    return denominator > 0 ? row.count / denominator : 0;
  };

  const seen = rows.filter((row) => row.count > 0).sort((a, b) => b.count - a.count);
  const unseen = rows.filter((row) => row.count === 0);
  const maxCount = seen.length ? seen[0].count : 0;

  return (
    <section aria-label="Category breakdown" className="space-y-4">
      <div>
        <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
          {seen.map((row) => (
            <div
              key={row.id}
              className="h-full first:rounded-l-full last:rounded-r-full"
              style={{ width: `${shareOf(row) * 100}%`, backgroundColor: row.colorHex }}
              title={`${row.label}: ${formatCount(row.count)} (${formatPercent(shareOf(row))})`}
            />
          ))}
        </div>
        {seen.length === 0 ? (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            Nothing classified yet in this window.
          </p>
        ) : null}
      </div>

      {seen.length > 0 ? (
        <ol className="space-y-2">
          {seen.map((row, index) => (
            <li key={row.id} className="flex items-center gap-3">
              <span className="w-4 shrink-0 text-right text-xs tabular-nums text-slate-400 dark:text-slate-500">
                {index + 1}
              </span>
              <span
                aria-hidden="true"
                className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-sm"
                style={{ backgroundColor: row.colorHex, color: row.textColorHex }}
              >
                {row.icon}
              </span>
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline justify-between gap-2">
                  <span className="truncate text-sm font-medium text-slate-800 dark:text-slate-100">
                    {row.label}
                  </span>
                  <span className="shrink-0 text-xs tabular-nums text-slate-500 dark:text-slate-400">
                    {formatCount(row.count)} · {formatPercent(shareOf(row))}
                  </span>
                </span>
                <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                  <span
                    className="block h-full rounded-full"
                    style={{
                      width: `${maxCount > 0 ? (row.count / maxCount) * 100 : 0}%`,
                      backgroundColor: row.colorHex,
                    }}
                  />
                </span>
              </span>
            </li>
          ))}
        </ol>
      ) : null}

      {unseen.length > 0 ? (
        <div className="border-t border-slate-200 pt-3 dark:border-slate-800">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400 dark:text-slate-500">
            Not seen yet ({unseen.length})
          </p>
          <ul className="mt-2 flex flex-wrap gap-1.5">
            {unseen.map((row) => (
              <li
                key={row.id}
                className="flex items-center gap-1.5 rounded-full border border-slate-200 px-2 py-0.5 text-xs text-slate-400 dark:border-slate-700 dark:text-slate-500"
              >
                <span
                  aria-hidden="true"
                  className="inline-block h-2 w-2 rounded-full opacity-50"
                  style={{ backgroundColor: row.colorHex }}
                />
                {row.label}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export default CategoryBreakdown;
