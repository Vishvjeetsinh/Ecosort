import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from '../lib/api.js';

export const DEFAULT_HISTORY_FILTERS = Object.freeze({
  limit: 25,
  offset: 0,
  category: '',
  source: '',
  modelKind: '',
  regionId: '',
  days: 0, // 0 = no time window
});

function isAbort(err) {
  return Boolean(err) && err.name === 'AbortError';
}

/** Translate the UI's "last N days" control into the API's inclusive `from` date. */
function toQuery(filters) {
  const query = {
    limit: filters.limit,
    offset: filters.offset,
    category: filters.category,
    source: filters.source,
    modelKind: filters.modelKind,
    regionId: filters.regionId,
    includeImage: true,
  };
  const days = Number(filters.days);
  if (Number.isFinite(days) && days > 0) {
    query.from = new Date(Date.now() - days * 86_400_000).toISOString();
  }
  return query;
}

/**
 * Owns the history list: filters, paging, and the mutations that act on it.
 *
 * `remove` is optimistic — the row disappears immediately and is restored if the
 * DELETE fails. Mutations reject on failure so the caller can surface a toast.
 */
export default function useHistory(initialFilters) {
  const [filters, setFiltersState] = useState(() => ({
    ...DEFAULT_HISTORY_FILTERS,
    ...(initialFilters || {}),
  }));
  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [reloadToken, setReloadToken] = useState(0);

  // Snapshot source for optimistic rollback, kept in refs so the mutation callbacks
  // stay referentially stable. Synced in an effect (never during render) — a click
  // can only happen after the commit, so the refs are always current when read.
  const itemsRef = useRef(items);
  const totalRef = useRef(total);
  useEffect(() => {
    itemsRef.current = items;
    totalRef.current = total;
  }, [items, total]);

  const setFilters = useCallback((next) => {
    setFiltersState((previous) => {
      const patch = typeof next === 'function' ? next(previous) : next;
      if (!patch || typeof patch !== 'object') return previous;
      // Callers pass a full merged object as often as a patch, so compare values
      // rather than keys: only a genuine offset-only change may keep the offset.
      const changedKeys = Object.keys(patch).filter(
        (key) => !Object.is(previous[key], patch[key]),
      );
      if (changedKeys.length === 0) return previous; // identity kept => no refetch
      const merged = { ...previous, ...patch };
      const onlyOffsetChanged = changedKeys.every((key) => key === 'offset');
      return onlyOffsetChanged ? merged : { ...merged, offset: 0 };
    });
  }, []);

  const refresh = useCallback(() => {
    setReloadToken((n) => n + 1);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    setLoading(true);
    api
      .listClassifications(toQuery(filters), controller.signal)
      .then((response) => {
        if (!active) return;
        setItems(Array.isArray(response?.items) ? response.items : []);
        setTotal(Number.isFinite(response?.total) ? response.total : 0);
        setError(null);
      })
      .catch((err) => {
        if (!active || isAbort(err)) return;
        setItems([]);
        setTotal(0);
        setError(err);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [filters, reloadToken]);

  const remove = useCallback(
    async (id) => {
      const previousItems = itemsRef.current;
      const previousTotal = totalRef.current;
      setItems(previousItems.filter((item) => item.id !== id));
      setTotal(Math.max(0, previousTotal - 1));

      try {
        await api.deleteClassification(id);
      } catch (err) {
        setItems(previousItems);
        setTotal(previousTotal);
        setError(err);
        throw err;
      }
      // A deletion frees a slot on the current page — pull the next row in.
      refresh();
    },
    [refresh],
  );

  const correct = useCallback(async (id, category) => {
    const response = await api.updateClassification(id, {
      correctedCategory: category || null,
    });
    const updated = response?.item;
    if (!updated) {
      const err = new Error('The backend did not return the updated classification.');
      setError(err);
      throw err;
    }
    setItems((current) => current.map((item) => (item.id === updated.id ? updated : item)));
    return updated;
  }, []);

  const clearAll = useCallback(async () => {
    const previousItems = itemsRef.current;
    const previousTotal = totalRef.current;
    setItems([]);
    setTotal(0);

    try {
      await api.clearClassifications();
    } catch (err) {
      setItems(previousItems);
      setTotal(previousTotal);
      setError(err);
      throw err;
    }
    setFiltersState((previous) => ({ ...previous, offset: 0 }));
    refresh();
  }, [refresh]);

  return {
    items,
    total,
    loading,
    error,
    filters,
    setFilters,
    refresh,
    remove,
    correct,
    clearAll,
  };
}

export { useHistory };
