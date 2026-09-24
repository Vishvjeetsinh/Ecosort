import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import HistoryFilters from './HistoryFilters.jsx';
import HistoryItemCard from './HistoryItemCard.jsx';
import EmptyState from './EmptyState.jsx';
import ErrorBanner from './ErrorBanner.jsx';
import { indexById, FALLBACK_CATEGORIES } from '../lib/categories.js';

const DEFAULT_LIMIT = 25;
const CONFIRM_TIMEOUT_MS = 6000;

const GROUP_FORMAT = new Intl.DateTimeFormat(undefined, {
  weekday: 'long',
  day: 'numeric',
  month: 'long',
  year: 'numeric',
});

/** Local calendar day — "Today" must mean the viewer's today, not UTC's. */
function dayKey(date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
}

function groupHeading(date, key, todayKey, yesterdayKey) {
  if (key === todayKey) return 'Today';
  if (key === yesterdayKey) return 'Yesterday';
  return GROUP_FORMAT.format(date);
}

function SkeletonCard() {
  return (
    <div className="card flex animate-pulse gap-3" aria-hidden="true">
      <div className="h-20 w-20 shrink-0 rounded-lg bg-slate-200 dark:bg-slate-800" />
      <div className="flex-1 space-y-2 py-1">
        <div className="h-4 w-2/5 rounded bg-slate-200 dark:bg-slate-800" />
        <div className="h-3 w-1/4 rounded bg-slate-200 dark:bg-slate-800" />
        <div className="h-2 w-full rounded bg-slate-200 dark:bg-slate-800" />
        <div className="h-2 w-5/6 rounded bg-slate-200 dark:bg-slate-800" />
      </div>
    </div>
  );
}

export function HistoryPanel({
  items,
  total,
  loading = false,
  error = null,
  filters,
  onFiltersChange,
  onDelete,
  onCorrect,
  onClearAll,
  categories,
  regions,
  rules,
}) {
  const [pendingIds, setPendingIds] = useState(() => new Set());
  const [clearStage, setClearStage] = useState(0);
  const mountedRef = useRef(true);

  useEffect(() => () => {
    mountedRef.current = false;
  }, []);

  useEffect(() => {
    if (clearStage === 0) return undefined;
    const timer = window.setTimeout(() => setClearStage(0), CONFIRM_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [clearStage]);

  const list = Array.isArray(items) ? items : [];
  const catalogue = Array.isArray(categories) && categories.length ? categories : FALLBACK_CATEGORIES;
  const byId = useMemo(() => indexById(catalogue), [catalogue]);

  const totalCount = Number.isFinite(Number(total)) ? Math.max(0, Number(total)) : list.length;
  const limit = Number(filters?.limit) > 0 ? Number(filters.limit) : DEFAULT_LIMIT;
  const offset = Number(filters?.offset) > 0 ? Number(filters.offset) : 0;
  const page = Math.floor(offset / limit) + 1;
  const pageCount = Math.max(1, Math.ceil(totalCount / limit));

  /**
   * The handlers App passes may be async; track which rows are mid-flight so the card
   * can disable its own controls instead of letting a user double-delete a row.
   */
  const runMutation = useCallback((id, action) => {
    let result;
    try {
      result = action();
    } catch (err) {
      console.error('EcoSort: history action failed', err);
      return;
    }
    if (!result || typeof result.then !== 'function') return;

    setPendingIds((prev) => {
      const next = new Set(prev);
      next.add(id);
      return next;
    });

    // Promise.resolve() normalises any thenable the caller hands back.
    Promise.resolve(result)
      .catch((err) => {
        console.error('EcoSort: history action failed', err);
      })
      .finally(() => {
        if (!mountedRef.current) return;
        setPendingIds((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      });
  }, []);

  const handleDelete = useCallback(
    (id) => runMutation(id, () => onDelete?.(id)),
    [onDelete, runMutation],
  );

  const handleCorrect = useCallback(
    (id, categoryId) => runMutation(id, () => onCorrect?.(id, categoryId)),
    [onCorrect, runMutation],
  );

  const handleClearAll = () => {
    if (clearStage === 0) {
      setClearStage(1);
      return;
    }
    setClearStage(0);
    try {
      const result = onClearAll?.();
      if (result && typeof result.catch === 'function') {
        result.catch((err) => console.error('EcoSort: clearing history failed', err));
      }
    } catch (err) {
      console.error('EcoSort: clearing history failed', err);
    }
  };

  const goToOffset = (nextOffset) => {
    onFiltersChange?.({ ...(filters ?? {}), offset: Math.max(0, nextOffset) });
  };

  const groups = useMemo(() => {
    const now = new Date();
    const todayKey = dayKey(now);
    const yesterday = new Date(now.getTime());
    yesterday.setDate(yesterday.getDate() - 1);
    const yesterdayKey = dayKey(yesterday);

    const ordered = [];
    const index = new Map();

    for (const item of list) {
      const date = new Date(item?.createdAt);
      const valid = !Number.isNaN(date.getTime());
      const key = valid ? dayKey(date) : 'unknown';
      if (!index.has(key)) {
        const group = {
          key,
          heading: valid ? groupHeading(date, key, todayKey, yesterdayKey) : 'Unknown date',
          items: [],
        };
        index.set(key, group);
        ordered.push(group);
      }
      index.get(key).items.push(item);
    }
    return ordered;
  }, [list]);

  const guidanceFor = (item) => {
    const categoryId = item?.effectiveCategory ?? item?.correctedCategory ?? item?.topCategory;
    return rules?.categories?.[categoryId] ?? null;
  };

  return (
    <section aria-label="Classification history" className="space-y-4">
      <HistoryFilters
        filters={filters}
        onChange={onFiltersChange}
        categories={catalogue}
        regions={regions}
        total={totalCount}
      />

      {error ? <ErrorBanner error={error} title="Could not load history" /> : null}

      {loading ? (
        <div className="space-y-3">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
          <p className="sr-only" role="status">
            Loading history
          </p>
        </div>
      ) : null}

      {!loading && list.length === 0 && !error ? (
        <EmptyState
          icon="🗂️"
          title="No classifications yet"
          message="Classify an item on the Classify tab and save it — every saved result appears here with its confidence, bin guidance and the model that produced it."
        />
      ) : null}

      {!loading && list.length > 0 ? (
        <div className="space-y-6">
          {groups.map((group) => (
            <div key={group.key} className="space-y-3">
              <h3 className="sticky top-0 z-10 bg-slate-50/90 py-1 text-xs font-semibold uppercase tracking-wide text-slate-500 backdrop-blur dark:bg-slate-950/90 dark:text-slate-400">
                {group.heading}
              </h3>
              <ul className="space-y-3">
                {group.items.map((item) => (
                  <li key={item.id}>
                    <HistoryItemCard
                      item={item}
                      category={byId[item.effectiveCategory ?? item.correctedCategory ?? item.topCategory] ?? null}
                      guidance={guidanceFor(item)}
                      onDelete={handleDelete}
                      onCorrect={handleCorrect}
                      categories={catalogue}
                      busy={pendingIds.has(item.id)}
                    />
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      ) : null}

      {!loading && totalCount > 0 ? (
        <nav aria-label="History pages" className="flex items-center justify-between gap-2">
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => goToOffset(offset - limit)}
            disabled={offset <= 0}
          >
            Previous
          </button>
          <p className="text-xs text-slate-500 dark:text-slate-400" aria-live="polite">
            Page <span className="tabular-nums">{Math.min(page, pageCount)}</span> of{' '}
            <span className="tabular-nums">{pageCount}</span>
          </p>
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => goToOffset(offset + limit)}
            disabled={offset + limit >= totalCount}
          >
            Next
          </button>
        </nav>
      ) : null}

      <div className="flex flex-wrap items-center justify-end gap-2 border-t border-slate-200 pt-3 dark:border-slate-800">
        {clearStage === 1 ? (
          <p className="mr-auto text-xs text-slate-500 dark:text-slate-400">
            This deletes every saved classification, including rows hidden by the current filters.
          </p>
        ) : null}
        {clearStage === 1 ? (
          <button type="button" className="btn btn-ghost text-xs" onClick={() => setClearStage(0)}>
            Cancel
          </button>
        ) : null}
        <button
          type="button"
          onClick={handleClearAll}
          disabled={totalCount === 0 || loading}
          className={
            clearStage === 1
              ? 'btn bg-red-600 text-xs text-white hover:bg-red-700'
              : 'btn btn-ghost text-xs text-red-600 dark:text-red-400'
          }
        >
          {clearStage === 1
            ? `Really clear ${totalCount} ${totalCount === 1 ? 'item' : 'items'}?`
            : 'Clear all history'}
        </button>
      </div>
    </section>
  );
}

export default HistoryPanel;
