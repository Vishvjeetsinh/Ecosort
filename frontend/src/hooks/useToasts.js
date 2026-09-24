import { useCallback, useEffect, useRef, useState } from 'react';

const DEFAULT_DURATION_MS = 5000;

// Monotonic and module-scoped: stable ids without Math.random, and unique even if
// two hook instances ever coexist.
let nextToastId = 0;

/**
 * Transient notifications.
 * @returns {{toasts: Array, push: Function, dismiss: Function, clear: Function}}
 */
export default function useToasts() {
  const [toasts, setToasts] = useState([]);
  const timers = useRef(new Map());

  const dismiss = useCallback((id) => {
    const handle = timers.current.get(id);
    if (handle !== undefined) {
      clearTimeout(handle);
      timers.current.delete(id);
    }
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  const push = useCallback(
    ({ title, message, tone = 'info', duration = DEFAULT_DURATION_MS } = {}) => {
      const id = ++nextToastId;
      const toast = {
        id,
        title: title || '',
        message: message || '',
        tone: tone === 'success' || tone === 'error' ? tone : 'info',
      };
      setToasts((current) => [...current, toast]);

      if (duration > 0) {
        timers.current.set(
          id,
          setTimeout(() => dismiss(id), duration),
        );
      }
      return id;
    },
    [dismiss],
  );

  const clear = useCallback(() => {
    for (const handle of timers.current.values()) clearTimeout(handle);
    timers.current.clear();
    setToasts([]);
  }, []);

  useEffect(() => {
    const pending = timers.current;
    return () => {
      for (const handle of pending.values()) clearTimeout(handle);
      pending.clear();
    };
  }, []);

  return { toasts, push, dismiss, clear };
}

export { useToasts };
