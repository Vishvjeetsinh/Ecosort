import React, { useEffect, useRef, useState } from 'react';
import Spinner from './Spinner.jsx';
import {
  formatConfidence,
  formatDateTime,
  formatDuration,
  formatRelativeTime,
  titleCase,
} from '../lib/format.js';
import { indexById, FALLBACK_CATEGORIES } from '../lib/categories.js';

const SOURCE_META = {
  webcam: { icon: '📷', label: 'Webcam' },
  upload: { icon: '📁', label: 'Upload' },
};

const MODEL_META = {
  custom: { label: 'Custom model' },
  fallback: { label: 'Fallback' },
};

const CONFIRM_TIMEOUT_MS = 5000;

function Badge({ children, title }) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1 rounded-full border border-slate-200 px-2 py-0.5 text-[11px] text-slate-600 dark:border-slate-700 dark:text-slate-300"
    >
      {children}
    </span>
  );
}

/**
 * One saved classification: what it was, how sure the model was, and the two
 * corrections a user can make (re-categorise, delete).
 */
export function HistoryItemCard({ item, category, guidance, onDelete, onCorrect, categories, busy = false }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [picking, setPicking] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const rootRef = useRef(null);
  const summaryRef = useRef(null);

  // A destructive confirmation that lingers is a trap: forget it after a few seconds.
  useEffect(() => {
    if (!confirmDelete) return undefined;
    const timer = window.setTimeout(() => setConfirmDelete(false), CONFIRM_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [confirmDelete]);

  useEffect(() => {
    if (!menuOpen) {
      setPicking(false);
      setConfirmDelete(false);
      return undefined;
    }
    const onPointerDown = (event) => {
      if (rootRef.current && !rootRef.current.contains(event.target)) setMenuOpen(false);
    };
    document.addEventListener('pointerdown', onPointerDown);
    return () => document.removeEventListener('pointerdown', onPointerDown);
  }, [menuOpen]);

  if (!item) return null;

  const catalogue = Array.isArray(categories) && categories.length ? categories : FALLBACK_CATEGORIES;
  const byId = indexById(catalogue);

  const effectiveId = item.effectiveCategory ?? item.correctedCategory ?? item.topCategory;
  const effective = category ?? byId[effectiveId] ?? null;
  const colorHex = effective?.colorHex || '#64748b';
  const textColorHex = effective?.textColorHex || '#ffffff';
  const label = effective?.label || titleCase(String(effectiveId ?? 'unknown'));
  const icon = effective?.icon || '♻️';

  const original = byId[item.topCategory] ?? null;
  const originalLabel = original?.label || titleCase(String(item.topCategory ?? 'unknown'));

  const sourceMeta = SOURCE_META[item.source] ?? { icon: '•', label: titleCase(String(item.source ?? 'unknown')) };
  const modelMeta = MODEL_META[item.modelKind] ?? { label: titleCase(String(item.modelKind ?? 'unknown')) };
  const absoluteTime = formatDateTime(item.createdAt);
  const predictions = Array.isArray(item.predictions) ? item.predictions : [];

  const closeMenu = () => {
    setMenuOpen(false);
    summaryRef.current?.focus();
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Escape' && menuOpen) {
      event.stopPropagation();
      closeMenu();
    }
  };

  const handleCorrect = (categoryId) => {
    onCorrect?.(item.id, categoryId);
    setPicking(false);
    closeMenu();
  };

  const handleDelete = () => {
    if (!confirmDelete) {
      setConfirmDelete(true);
      return;
    }
    setConfirmDelete(false);
    setMenuOpen(false);
    onDelete?.(item.id);
  };

  return (
    <article
      ref={rootRef}
      onKeyDown={handleKeyDown}
      className={`card relative flex gap-3 ${busy ? 'opacity-60' : ''}`}
      aria-busy={busy || undefined}
    >
      {busy ? (
        <span className="absolute bottom-3 right-3 z-10">
          <Spinner size={14} label="Saving change" />
        </span>
      ) : null}

      {item.imageDataUrl ? (
        <img
          src={item.imageDataUrl}
          alt={`Item classified as ${label}`}
          loading="lazy"
          decoding="async"
          className="h-20 w-20 shrink-0 rounded-lg border border-slate-200 object-cover dark:border-slate-700"
        />
      ) : (
        <span
          aria-hidden="true"
          className="grid h-20 w-20 shrink-0 place-items-center rounded-lg text-2xl"
          style={{ backgroundColor: colorHex, color: textColorHex }}
        >
          {icon}
        </span>
      )}

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span
                className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold"
                style={{ backgroundColor: colorHex, color: textColorHex }}
              >
                <span aria-hidden="true">{icon}</span>
                {label}
              </span>
              <span className="text-sm font-semibold tabular-nums text-slate-700 dark:text-slate-200">
                {formatConfidence(item.topConfidence)}
              </span>
              {item.correctedCategory ? (
                <span className="inline-flex items-center rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-medium text-amber-900 dark:bg-amber-500/20 dark:text-amber-200">
                  Corrected · was {originalLabel}
                </span>
              ) : null}
            </div>

            <p className="mt-1 truncate text-xs text-slate-500 dark:text-slate-400" title={item.topLabel}>
              {item.topLabel}
            </p>
          </div>

          <details
            open={menuOpen}
            onToggle={(event) => setMenuOpen(event.currentTarget.open)}
            className="relative shrink-0"
          >
            <summary
              ref={summaryRef}
              className="btn btn-ghost cursor-pointer list-none px-2 py-1 text-sm [&::-webkit-details-marker]:hidden"
              aria-label={`Actions for classification ${item.id}`}
            >
              <span aria-hidden="true">⋯</span>
            </summary>
            <div className="absolute right-0 z-20 mt-1 w-56 rounded-xl border border-slate-200 bg-white p-2 shadow-lg dark:border-slate-700 dark:bg-slate-900">
              {picking ? (
                <div>
                  <p className="px-1 pb-2 text-xs font-medium text-slate-500 dark:text-slate-400">
                    Correct category
                  </p>
                  <div className="grid grid-cols-2 gap-1">
                    {catalogue.map((option) => {
                      const current = option.id === effectiveId;
                      return (
                        <button
                          key={option.id}
                          type="button"
                          aria-pressed={current}
                          disabled={busy}
                          onClick={() => handleCorrect(option.id)}
                          className={`flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-left text-xs hover:bg-slate-100 disabled:opacity-50 dark:hover:bg-slate-800 ${
                            current ? 'ring-2 ring-emerald-500' : ''
                          }`}
                        >
                          <span
                            aria-hidden="true"
                            className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                            style={{ backgroundColor: option.colorHex }}
                          />
                          <span className="truncate">{option.shortLabel || option.label || option.id}</span>
                        </button>
                      );
                    })}
                  </div>
                  <button
                    type="button"
                    className="btn btn-ghost mt-2 w-full text-xs"
                    onClick={() => setPicking(false)}
                  >
                    Back
                  </button>
                </div>
              ) : (
                <div className="flex flex-col">
                  <button
                    type="button"
                    className="rounded-lg px-2 py-1.5 text-left text-sm hover:bg-slate-100 disabled:opacity-50 dark:hover:bg-slate-800"
                    onClick={() => setPicking(true)}
                    disabled={busy}
                  >
                    Correct category
                  </button>
                  <button
                    type="button"
                    className={`rounded-lg px-2 py-1.5 text-left text-sm disabled:opacity-50 ${
                      confirmDelete
                        ? 'bg-red-600 text-white hover:bg-red-700'
                        : 'text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-500/10'
                    }`}
                    onClick={handleDelete}
                    disabled={busy}
                  >
                    {confirmDelete ? 'Click again to delete' : 'Delete'}
                  </button>
                </div>
              )}
            </div>
          </details>
        </div>

        <div className="mt-2 flex flex-wrap items-center gap-1.5">
          <Badge title={absoluteTime}>{formatRelativeTime(item.createdAt)}</Badge>
          <Badge>
            <span aria-hidden="true">{sourceMeta.icon}</span>
            {sourceMeta.label}
          </Badge>
          <Badge>{modelMeta.label}</Badge>
          {item.durationMs !== null && item.durationMs !== undefined && Number.isFinite(Number(item.durationMs)) ? (
            <Badge>{formatDuration(item.durationMs)}</Badge>
          ) : null}
          {guidance ? (
            <Badge title={guidance.disposal || undefined}>
              <span
                aria-hidden="true"
                className="inline-block h-2 w-2 rounded-full"
                style={{ backgroundColor: guidance.colorHex || colorHex }}
              />
              {guidance.binName || guidance.colorName || 'Bin'}
              {guidance.recyclable ? ' · recyclable' : ''}
            </Badge>
          ) : null}
        </div>

        {predictions.length > 0 ? (
          <ul className="mt-2 space-y-1">
            {predictions.slice(0, 3).map((prediction, index) => {
              const meta = byId[prediction.category] ?? null;
              const ratio = Math.max(0, Math.min(1, Number(prediction.confidence) || 0));
              return (
                <li key={`${prediction.category}-${index}`} className="flex items-center gap-2 text-[11px]">
                  <span className="w-24 shrink-0 truncate text-slate-500 dark:text-slate-400" title={prediction.label}>
                    {prediction.label || meta?.label || prediction.category}
                  </span>
                  <span className="h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                    <span
                      className="block h-full rounded-full"
                      style={{
                        width: `${ratio * 100}%`,
                        backgroundColor: meta?.colorHex || colorHex,
                      }}
                    />
                  </span>
                  <span className="w-10 shrink-0 text-right tabular-nums text-slate-500 dark:text-slate-400">
                    {formatConfidence(prediction.confidence)}
                  </span>
                </li>
              );
            })}
          </ul>
        ) : null}

        {item.notes ? (
          <p className="mt-2 rounded-lg bg-slate-100 px-2 py-1 text-xs text-slate-600 dark:bg-slate-800 dark:text-slate-300">
            {item.notes}
          </p>
        ) : null}
      </div>
    </article>
  );
}

export default HistoryItemCard;
