import { useEffect, useState } from 'react';

const PREFIX = 'ecosort:';

/** Warn once per key: a blocked storage is a permanent condition, not a per-write event. */
const warned = new Set();

function warnOnce(key, err) {
  if (warned.has(key)) return;
  warned.add(key);
  console.warn(
    `[ecosort] localStorage is unavailable for "${key}" — this setting will not persist across reloads.`,
    err,
  );
}

function readStored(storageKey) {
  try {
    const raw = window.localStorage.getItem(storageKey);
    if (raw === null) return undefined;
    return JSON.parse(raw);
  } catch (err) {
    // Either storage is blocked (private mode) or the value predates JSON storage.
    warnOnce(storageKey, err);
    return undefined;
  }
}

/**
 * JSON-backed state persisted under a namespaced key.
 *
 * Degrades to plain in-memory state when `localStorage` throws or is absent, so the
 * app still works in private browsing and in a jsdom test environment.
 *
 * @param {string} key unprefixed key, e.g. "theme" -> "ecosort:theme"
 * @param {*} initial value, or a factory called only on first mount
 * @returns {[*, Function]} the usual `[value, setValue]`; setValue accepts an updater
 */
export default function useLocalStorage(key, initial) {
  const storageKey = PREFIX + key;

  const [value, setValue] = useState(() => {
    const stored = readStored(storageKey);
    if (stored !== undefined) return stored;
    return typeof initial === 'function' ? initial() : initial;
  });

  // Writing in an effect (not inside the setState updater) keeps the updater pure,
  // which matters under StrictMode's double-invocation in development.
  useEffect(() => {
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(value));
    } catch (err) {
      warnOnce(storageKey, err);
    }
  }, [storageKey, value]);

  return [value, setValue];
}

export { useLocalStorage };
