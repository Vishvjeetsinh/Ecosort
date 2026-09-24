import React, { useEffect, useMemo, useState } from 'react';
import { formatDuration } from '../lib/format.js';
import { categoriesOrFallback, categoryById } from '../lib/categories.js';
import EmptyState from './EmptyState.jsx';
import ErrorBanner from './ErrorBanner.jsx';
import PredictionList from './PredictionList.jsx';
import BinGuideCard from './BinGuideCard.jsx';
import RawLabelsDisclosure from './RawLabelsDisclosure.jsx';
import { withAlpha } from './ConfidenceBar.jsx';

const ENGINE_LABEL = {
  custom: 'Custom trained model',
  fallback: 'MobileNetV2 fallback',
};

function CameraIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className="h-10 w-10" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.8l1.2-2h6.9l1.2 2h1.9A2.5 2.5 0 0 1 21 8.5v9A2.5 2.5 0 0 1 18.5 20h-13A2.5 2.5 0 0 1 3 17.5v-9Z" />
      <circle cx="12" cy="13" r="3.6" />
    </svg>
  );
}

function Skeleton() {
  return (
    <div className="grid gap-4 lg:grid-cols-5" aria-hidden="true">
      <div className="space-y-2 lg:col-span-2">
        {[0, 1, 2].map((i) => (
          <div key={i} className="flex items-center gap-3 rounded-xl border border-slate-200 p-3 dark:border-slate-700">
            <div className="h-10 w-10 shrink-0 animate-pulse rounded-lg bg-slate-200 dark:bg-slate-700" />
            <div className="flex-1 space-y-2">
              <div className="h-3 w-2/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
              <div className="h-2 w-full animate-pulse rounded-full bg-slate-200 dark:bg-slate-700" />
            </div>
          </div>
        ))}
      </div>
      <div className="card space-y-4 p-4 lg:col-span-3">
        <div className="h-24 animate-pulse rounded-2xl bg-slate-200 dark:bg-slate-700" />
        <div className="h-4 w-1/3 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
        <div className="h-4 w-4/5 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
        <div className="grid gap-3 sm:grid-cols-2">
          <div className="h-20 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
          <div className="h-20 animate-pulse rounded bg-slate-200 dark:bg-slate-700" />
        </div>
      </div>
    </div>
  );
}

/**
 * Results + guidance for one classification.
 *
 * Takes either the resolved `guidance`/`bin`/`region` trio or the whole `rules`
 * payload for the active region, so App can wire it either way; `onSave`/`onSaved`
 * are accepted as synonyms for the same reason.
 */
export function ResultsPanel({
  result,
  guidance,
  bin,
  region,
  rules,
  categories,
  busy = false,
  error = null,
  engineKind,
  imageDataUrl = null,
  onSave,
  onSaved,
  onCorrect,
  saving = false,
  saved = false,
  selectedCategoryId,
  onSelectCategory,
}) {
  const categoryList = useMemo(() => categoriesOrFallback(categories), [categories]);

  const [internalSelected, setInternalSelected] = useState(null);
  const [pickerOpen, setPickerOpen] = useState(false);

  // A new classification invalidates both the manual selection and an open picker.
  useEffect(() => {
    setInternalSelected(null);
    setPickerOpen(false);
  }, [result]);

  const predictions = Array.isArray(result?.predictions) ? result.predictions : [];
  const topCategoryId = predictions[0]?.category ?? null;
  const activeId = selectedCategoryId ?? internalSelected ?? topCategoryId;

  const handleSelect = (categoryId) => {
    setInternalSelected(categoryId);
    onSelectCategory?.(categoryId);
  };

  const activeGuidance = guidance ?? (activeId ? rules?.categories?.[activeId] ?? null : null);
  const activeRegion = region ?? rules?.region ?? null;
  const activeBin =
    bin ??
    (activeGuidance && Array.isArray(rules?.bins)
      ? rules.bins.find((b) => b.id === activeGuidance.binId) ?? null
      : null);

  const kind = engineKind ?? result?.modelKind ?? 'fallback';
  const save = onSave ?? onSaved;

  if (error) {
    // ErrorBanner reads `error.message`, so a plain string has to be wrapped.
    const errorObject = typeof error === 'string' ? { message: error } : error;
    return <ErrorBanner title="Classification failed" error={errorObject} />;
  }

  if (busy) {
    return (
      <div aria-busy="true" aria-live="polite">
        <p className="sr-only">Classifying the image.</p>
        <Skeleton />
      </div>
    );
  }

  if (!result || predictions.length === 0) {
    return (
      <EmptyState
        icon={<CameraIcon />}
        title="Nothing classified yet"
        message="Take a photo with your webcam or upload an image, and EcoSort will tell you which bin it belongs in — entirely on this machine, with no image ever leaving it."
      />
    );
  }

  const activeCategory = categoryById(categoryList, activeId);

  return (
    <section className="space-y-4" aria-label="Classification result">
      <div className="grid gap-4 lg:grid-cols-5">
        <div className="lg:col-span-2">
          <PredictionList
            predictions={predictions}
            categories={categoryList}
            selectedId={activeId}
            onSelect={handleSelect}
            lowConfidence={Boolean(result.lowConfidence)}
            unmatchedMass={Number(result.unmatchedMass) || 0}
            engineKind={kind}
          />
        </div>
        <div className="lg:col-span-3">
          <BinGuideCard
            guidance={activeGuidance}
            category={activeCategory}
            bin={activeBin}
            region={activeRegion}
          />
        </div>
      </div>

      <div className="card flex flex-col gap-3 p-3 sm:p-4">
        <div className="flex flex-wrap items-center gap-3">
          {imageDataUrl ? (
            <img
              src={imageDataUrl}
              alt="The picture that was classified"
              className="h-12 w-12 shrink-0 rounded-lg object-cover ring-1 ring-slate-200 dark:ring-slate-700"
            />
          ) : null}

          <dl className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-600 dark:text-slate-300">
            <div className="flex items-center gap-1.5">
              <dt className="font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Engine</dt>
              <dd>{ENGINE_LABEL[kind] || kind}</dd>
            </div>
            <div className="flex items-center gap-1.5">
              <dt className="font-semibold uppercase tracking-wide text-slate-400 dark:text-slate-500">Inference</dt>
              <dd className="tabular-nums">
                {Number.isFinite(Number(result.durationMs)) ? formatDuration(result.durationMs) : 'not measured'}
              </dd>
            </div>
          </dl>

          <div className="ml-auto flex flex-wrap items-center gap-2">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => save?.()}
              disabled={saved || saving || !save}
            >
              {saved ? 'Saved ✓' : saving ? 'Saving…' : 'Save to history'}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={() => setPickerOpen((open) => !open)}
              aria-expanded={pickerOpen}
              aria-controls="results-correction-picker"
              disabled={!onCorrect}
            >
              Wrong? Pick the right category
            </button>
          </div>
        </div>

        {pickerOpen ? (
          <div id="results-correction-picker" className="border-t border-slate-200 pt-3 dark:border-slate-700">
            <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">
              Your correction is stored with this classification, so history and statistics count
              the right category.
            </p>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-5">
              {categoryList.map((category) => (
                <button
                  key={category.id}
                  type="button"
                  className="flex items-center gap-2 rounded-lg border border-slate-200 p-2 text-left text-xs font-medium text-slate-700 hover:border-slate-300 dark:border-slate-700 dark:text-slate-100"
                  onClick={() => {
                    onCorrect?.(category.id);
                    setPickerOpen(false);
                  }}
                  style={{ backgroundColor: withAlpha(category.colorHex, 0.1) }}
                >
                  <span
                    aria-hidden="true"
                    className="grid h-6 w-6 shrink-0 place-items-center rounded"
                    style={{ backgroundColor: category.colorHex, color: category.textColorHex || '#ffffff' }}
                  >
                    {category.icon || category.label?.slice(0, 1)}
                  </span>
                  <span className="truncate">{category.shortLabel || category.label}</span>
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      <RawLabelsDisclosure rawLabels={result.rawLabels} engineKind={kind} />
    </section>
  );
}

export default ResultsPanel;
