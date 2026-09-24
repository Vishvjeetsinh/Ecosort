import React from 'react';
import PrepStepList from './PrepStepList.jsx';
import { withAlpha } from './ConfidenceBar.jsx';

/** Simple wheelie-bin silhouette; drawn inline so the app ships no icon font or CDN asset. */
function BinGlyph({ className = 'h-14 w-14' }) {
  return (
    <svg viewBox="0 0 64 64" aria-hidden="true" className={className} fill="none" stroke="currentColor" strokeWidth="3" strokeLinejoin="round" strokeLinecap="round">
      <path d="M18 12h28a4 4 0 0 1 4 4v2H14v-2a4 4 0 0 1 4-4Z" fill="currentColor" fillOpacity="0.25" />
      <path d="M26 12V9a3 3 0 0 1 3-3h6a3 3 0 0 1 3 3v3" />
      <path d="M16 18h32l-2.6 29.4A5 5 0 0 1 40.4 52H23.6a5 5 0 0 1-5-4.6L16 18Z" fill="currentColor" fillOpacity="0.15" />
      <path d="M27 27v16M37 27v16" />
      <circle cx="24" cy="57" r="3.5" fill="currentColor" fillOpacity="0.35" />
      <circle cx="40" cy="57" r="3.5" fill="currentColor" fillOpacity="0.35" />
    </svg>
  );
}

function CheckGlyph() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" className="mt-1 h-4 w-4 shrink-0 text-emerald-600 dark:text-emerald-400" fill="currentColor">
      <path d="M8.2 14.3 4 10.1l1.5-1.5 2.7 2.7 6.3-6.3L16 6.5l-7.8 7.8Z" />
    </svg>
  );
}

function CrossGlyph() {
  return (
    <svg viewBox="0 0 20 20" aria-hidden="true" className="mt-1 h-4 w-4 shrink-0 text-rose-600 dark:text-rose-400" fill="currentColor">
      <path d="M14.7 6.7 13.3 5.3 10 8.6 6.7 5.3 5.3 6.7 8.6 10l-3.3 3.3 1.4 1.4L10 11.4l3.3 3.3 1.4-1.4L11.4 10l3.3-3.3Z" />
    </svg>
  );
}

function ExampleColumn({ title, items, tone }) {
  const list = Array.isArray(items) ? items.filter(Boolean) : [];
  return (
    <div>
      <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {title}
      </h4>
      {list.length > 0 ? (
        <ul className="space-y-1.5">
          {list.map((item) => (
            <li key={item} className="flex gap-2 text-sm leading-6 text-slate-700 dark:text-slate-200">
              {tone === 'accept' ? <CheckGlyph /> : <CrossGlyph />}
              <span>{item}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-slate-500 dark:text-slate-400">No examples listed for this region.</p>
      )}
    </div>
  );
}

/**
 * The guidance centrepiece: which bin this item goes in, how to prepare it, and
 * whose rules are being quoted.
 */
export function BinGuideCard({ guidance, category, bin, region }) {
  if (!guidance) {
    return (
      <div className="card flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <div className="text-slate-400 dark:text-slate-500">
          <BinGlyph className="h-16 w-16" />
        </div>
        <p className="text-sm font-semibold text-slate-700 dark:text-slate-200">No bin guidance yet</p>
        <p className="max-w-xs text-sm text-slate-500 dark:text-slate-400">
          Pick a prediction on the left — or wait for the local recycling rules for your region to
          finish loading — and the bin, prep steps and accepted items appear here.
        </p>
      </div>
    );
  }

  const colorHex = guidance.colorHex || category?.colorHex || '#64748b';
  const textColorHex = guidance.textColorHex || '#ffffff';
  const categoryLabelText = category?.label || guidance.categoryId || 'Item';
  const recyclable = Boolean(guidance.recyclable);

  return (
    <article className="card flex h-full flex-col gap-5 p-4 sm:p-5" aria-label={`Disposal guidance for ${categoryLabelText}`}>
      <div
        className="flex items-center gap-4 rounded-2xl p-4 sm:p-5"
        style={{ backgroundColor: colorHex, color: textColorHex }}
      >
        <BinGlyph className="h-14 w-14 shrink-0 sm:h-16 sm:w-16" />
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-wide opacity-80">
            {categoryLabelText}
          </p>
          <p className="truncate text-2xl font-bold leading-tight sm:text-3xl">
            {guidance.colorName || 'Unlabelled'} bin
          </p>
          <p className="truncate text-sm opacity-90">{guidance.binName || bin?.name || 'Kerbside collection'}</p>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <span
          className={[
            'inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold',
            recyclable
              ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/15 dark:text-emerald-200'
              : 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-100',
          ].join(' ')}
        >
          <span aria-hidden="true">{recyclable ? '♻' : '✕'}</span>
          {recyclable ? 'Recyclable in this stream' : 'Not recyclable — disposal only'}
        </span>
        {bin?.id ? (
          <span
            className="inline-flex items-center rounded-full px-3 py-1 text-xs font-medium text-slate-700 dark:text-slate-100"
            style={{ backgroundColor: withAlpha(colorHex, 0.18) }}
          >
            Bin id: {bin.id}
          </span>
        ) : null}
      </div>

      {guidance.disposal ? (
        <p className="text-base font-semibold leading-7 text-slate-900 dark:text-slate-50 sm:text-lg">
          {guidance.disposal}
        </p>
      ) : null}

      {Array.isArray(guidance.prepSteps) && guidance.prepSteps.length > 0 ? (
        <div>
          <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
            Prepare it like this
          </h4>
          <PrepStepList steps={guidance.prepSteps} />
        </div>
      ) : null}

      <div className="grid gap-5 sm:grid-cols-2">
        <ExampleColumn title="Accepted here" items={guidance.acceptedExamples} tone="accept" />
        <ExampleColumn title="Not in this bin" items={guidance.rejectedExamples} tone="reject" />
      </div>

      {bin?.description ? (
        <p className="text-sm leading-6 text-slate-600 dark:text-slate-300">{bin.description}</p>
      ) : null}

      {guidance.notes ? (
        <p className="rounded-xl bg-slate-100 p-3 text-sm leading-6 text-slate-600 dark:bg-slate-800 dark:text-slate-300">
          {guidance.notes}
        </p>
      ) : null}

      {guidance.dropOff ? (
        <div className="rounded-xl border border-indigo-300 bg-indigo-50 p-3 dark:border-indigo-500/60 dark:bg-indigo-500/10">
          <p className="text-xs font-semibold uppercase tracking-wide text-indigo-700 dark:text-indigo-300">
            Drop-off required
          </p>
          <p className="mt-1 text-sm leading-6 text-indigo-900 dark:text-indigo-100">{guidance.dropOff}</p>
        </div>
      ) : null}

      <footer className="mt-auto border-t border-slate-200 pt-3 text-xs leading-5 text-slate-500 dark:border-slate-700 dark:text-slate-400">
        Rules for <span className="font-medium text-slate-700 dark:text-slate-200">{region?.name || 'your region'}</span>
        {region?.authority ? <> · {region.authority}</> : null}
        {region?.updated ? <> · updated {region.updated}</> : null}
        {region?.notes ? <div className="mt-1">{region.notes}</div> : null}
      </footer>
    </article>
  );
}

export default BinGuideCard;
