import React, { useMemo, useState } from 'react';
import { categoriesOrFallback, categoryById } from '../lib/categories.js';
import { withAlpha } from './ConfidenceBar.jsx';

function haystack(values) {
  return values
    .flat()
    .filter((v) => typeof v === 'string')
    .join(' ')
    .toLowerCase();
}

function Swatch({ colorHex, textColorHex, colorName, size = 'md' }) {
  const box = size === 'sm' ? 'h-6 w-6 text-[10px]' : 'h-10 w-10 text-xs';
  return (
    <span className="inline-flex items-center gap-2">
      <span
        aria-hidden="true"
        className={`grid shrink-0 place-items-center rounded-md font-bold ring-1 ring-black/10 dark:ring-white/10 ${box}`}
        style={{ backgroundColor: colorHex || '#64748b', color: textColorHex || '#ffffff' }}
      >
        ●
      </span>
      {/* The colour NAME is always rendered as text: colour alone must never carry meaning. */}
      <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{colorName || 'Unnamed'}</span>
    </span>
  );
}

/**
 * The standalone bin-colour guide tab: every bin in the selected region, every
 * category's destination, and a search that spans categories, bins and examples.
 */
export function BinColorGuide({ rules, categories, query, onQueryChange }) {
  const [internalQuery, setInternalQuery] = useState('');
  const value = typeof query === 'string' ? query : internalQuery;

  const setQuery = (next) => {
    setInternalQuery(next);
    onQueryChange?.(next);
  };

  const categoryList = useMemo(() => categoriesOrFallback(categories), [categories]);

  const bins = Array.isArray(rules?.bins) ? rules.bins : [];
  const guidanceById = rules?.categories && typeof rules.categories === 'object' ? rules.categories : {};
  const region = rules?.region ?? null;

  const q = value.trim().toLowerCase();

  const categoryMatches = useMemo(() => {
    const map = new Map();
    for (const category of categoryList) {
      const guidance = guidanceById[category.id] || null;
      const text = haystack([
        category.id,
        category.label,
        category.shortLabel,
        category.description,
        category.examples || [],
        guidance?.binName,
        guidance?.colorName,
        guidance?.disposal,
        guidance?.notes,
        guidance?.dropOff,
        guidance?.acceptedExamples || [],
        guidance?.rejectedExamples || [],
        guidance?.prepSteps || [],
      ]);
      map.set(category.id, !q || text.includes(q));
    }
    return map;
  }, [categoryList, guidanceById, q]);

  const visibleCategories = categoryList.filter((c) => categoryMatches.get(c.id));

  const visibleBins = bins.filter((bin) => {
    if (!q) return true;
    const own = haystack([bin.id, bin.name, bin.colorName, bin.description]);
    if (own.includes(q)) return true;
    return (Array.isArray(bin.accepts) ? bin.accepts : []).some((id) => categoryMatches.get(id));
  });

  // categoryList always has the ten categories, so "no rules" must be judged on the region payload.
  const hasRules = bins.length > 0 || Object.keys(guidanceById).length > 0;
  if (!hasRules) {
    return (
      <div className="card p-6 text-center">
        <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">No recycling rules loaded</p>
        <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
          Pick a region in the header — EcoSort loads its bin colours and rules from the backend at
          <code className="mx-1 rounded bg-slate-100 px-1 py-0.5 text-xs dark:bg-slate-800">/api/rules</code>.
        </p>
      </div>
    );
  }

  const nothingMatches = q !== '' && visibleBins.length === 0 && visibleCategories.length === 0;

  return (
    <section className="space-y-6" aria-label="Bin colour guide">
      <div className="card space-y-3 p-4">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">
            Bin colour guide{region?.name ? ` — ${region.name}` : ''}
          </h2>
          {region?.authority || region?.updated ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {region?.authority}
              {region?.authority && region?.updated ? ' · ' : ''}
              {region?.updated ? `updated ${region.updated}` : ''}
            </p>
          ) : null}
        </div>

        <div>
          <label htmlFor="bin-guide-search" className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-300">
            Search a category, a bin, or an item
          </label>
          <div className="flex gap-2">
            <input
              id="bin-guide-search"
              type="search"
              className="input flex-1"
              placeholder="e.g. glass, blue bin, pizza box, battery"
              value={value}
              onChange={(event) => setQuery(event.target.value)}
              autoComplete="off"
            />
            {value ? (
              <button type="button" className="btn btn-ghost" onClick={() => setQuery('')}>
                Clear
              </button>
            ) : null}
          </div>
          <p className="mt-1 text-xs text-slate-500 dark:text-slate-400" aria-live="polite">
            {q
              ? `${visibleBins.length} bin${visibleBins.length === 1 ? '' : 's'} and ${visibleCategories.length} categor${visibleCategories.length === 1 ? 'y' : 'ies'} match “${value.trim()}”.`
              : `${bins.length} bin${bins.length === 1 ? '' : 's'}, ${categoryList.length} categories.`}
          </p>
        </div>
      </div>

      {nothingMatches ? (
        <div className="card p-6 text-center">
          <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">
            Nothing matches “{value.trim()}”
          </p>
          <p className="mt-1 text-sm text-slate-500 dark:text-slate-400">
            Try a material (glass, metal), a bin colour (blue, green) or an item (battery, pizza box).
          </p>
          <button type="button" className="btn btn-primary mt-3" onClick={() => setQuery('')}>
            Clear the search
          </button>
        </div>
      ) : null}

      {visibleBins.length > 0 ? (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Bins in this region
          </h3>
          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {visibleBins.map((bin) => {
              const accepted = (Array.isArray(bin.accepts) ? bin.accepts : [])
                .map((id) => categoryById(categoryList, id))
                .filter(Boolean);
              return (
                <article key={bin.id} className="card flex flex-col gap-3 p-4">
                  <div
                    className="flex items-center justify-between gap-2 rounded-xl p-3"
                    style={{ backgroundColor: bin.colorHex || '#64748b', color: bin.textColorHex || '#ffffff' }}
                  >
                    <div className="min-w-0">
                      <p className="truncate text-base font-bold">{bin.name}</p>
                      <p className="truncate text-xs opacity-90">{bin.colorName} bin</p>
                    </div>
                    <span className="shrink-0 rounded-full bg-black/20 px-2 py-0.5 text-[11px] font-semibold">
                      {accepted.length} type{accepted.length === 1 ? '' : 's'}
                    </span>
                  </div>

                  {bin.description ? (
                    <p className="text-sm leading-6 text-slate-600 dark:text-slate-300">{bin.description}</p>
                  ) : null}

                  <div className="mt-auto">
                    <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      Accepts
                    </h4>
                    {accepted.length > 0 ? (
                      <ul className="flex flex-wrap gap-1.5">
                        {accepted.map((category) => (
                          <li key={category.id}>
                            <span
                              className="chip text-slate-800 dark:text-slate-100"
                              style={{ backgroundColor: withAlpha(category.colorHex, 0.18) }}
                            >
                              <span aria-hidden="true">{category.icon || '•'}</span>
                              {category.shortLabel || category.label}
                            </span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <p className="text-sm text-slate-500 dark:text-slate-400">
                        Nothing is routed to this bin in the current rules.
                      </p>
                    )}
                  </div>
                </article>
              );
            })}
          </div>
        </div>
      ) : null}

      {visibleCategories.length > 0 ? (
        <div>
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Where each category goes
          </h3>
          <div className="card overflow-x-auto p-0">
            <table className="w-full min-w-[40rem] border-collapse text-left text-sm">
              <caption className="sr-only">
                Every waste category with its bin, whether it is recyclable, and the first preparation steps.
              </caption>
              <thead className="bg-slate-50 text-xs uppercase tracking-wide text-slate-500 dark:bg-slate-800 dark:text-slate-400">
                <tr>
                  <th scope="col" className="px-4 py-2 font-semibold">Category</th>
                  <th scope="col" className="px-4 py-2 font-semibold">Bin</th>
                  <th scope="col" className="px-4 py-2 font-semibold">Recyclable</th>
                  <th scope="col" className="px-4 py-2 font-semibold">First steps</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 dark:divide-slate-700">
                {visibleCategories.map((category) => {
                  const guidance = guidanceById[category.id] || null;
                  const steps = Array.isArray(guidance?.prepSteps) ? guidance.prepSteps.slice(0, 2) : [];
                  return (
                    <tr key={category.id} className="align-top">
                      <th scope="row" className="px-4 py-3 font-medium text-slate-900 dark:text-slate-50">
                        <span className="inline-flex items-center gap-2">
                          <span
                            aria-hidden="true"
                            className="grid h-7 w-7 shrink-0 place-items-center rounded-md text-sm"
                            style={{ backgroundColor: category.colorHex, color: category.textColorHex || '#ffffff' }}
                          >
                            {category.icon || category.label?.slice(0, 1)}
                          </span>
                          <span>
                            {category.label}
                            {category.description ? (
                              <span className="block text-xs font-normal text-slate-500 dark:text-slate-400">
                                {category.description}
                              </span>
                            ) : null}
                          </span>
                        </span>
                      </th>
                      <td className="px-4 py-3">
                        {guidance ? (
                          <span className="flex flex-col gap-0.5">
                            <Swatch
                              colorHex={guidance.colorHex}
                              textColorHex={guidance.textColorHex}
                              colorName={guidance.colorName}
                              size="sm"
                            />
                            <span className="text-xs text-slate-500 dark:text-slate-400">{guidance.binName}</span>
                          </span>
                        ) : (
                          <span className="text-slate-400 dark:text-slate-500">No rule</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={[
                            'inline-flex rounded-full px-2 py-0.5 text-xs font-semibold',
                            guidance?.recyclable
                              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-200'
                              : 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-100',
                          ].join(' ')}
                        >
                          {guidance?.recyclable ? 'Yes' : 'No'}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-slate-700 dark:text-slate-200">
                        {steps.length > 0 ? (
                          <ol className="list-decimal space-y-1 pl-4">
                            {steps.map((step, i) => (
                              <li key={`${category.id}-step-${i}`}>{step}</li>
                            ))}
                          </ol>
                        ) : (
                          <span className="text-slate-400 dark:text-slate-500">
                            {guidance?.disposal || 'No preparation needed.'}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      ) : null}

      <div className="card space-y-2 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">Legend</h3>
        <ul className="flex flex-wrap gap-3">
          {bins.map((bin) => (
            <li key={`legend-${bin.id}`}>
              <Swatch
                colorHex={bin.colorHex}
                textColorHex={bin.textColorHex}
                colorName={`${bin.colorName} — ${bin.name}`}
                size="sm"
              />
            </li>
          ))}
        </ul>
        <p className="text-xs leading-5 text-slate-500 dark:text-slate-400">
          Colour is never the only signal here: every swatch is labelled with its colour name in
          text, and the tables repeat the bin name, so the guide stays readable without colour
          vision. Bin colours differ between councils — always check the wording, not the hue,
          against the bins you actually have.
        </p>
      </div>
    </section>
  );
}

export default BinColorGuide;
