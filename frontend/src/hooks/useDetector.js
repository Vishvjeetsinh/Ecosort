/**
 * React binding for `lib/detector.js`, shaped like useClassifier.
 *
 * The detector is ~18 MB and only the Live scan tab needs it, so it is loaded the first
 * time `enabled` turns true - never on app start - and then kept for the life of the app,
 * so leaving the tab and coming back does not pay for the download and warm-up again.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { loadDetector } from '../lib/detector.js';

const IDLE_STATE = { status: 'idle', detector: null, progress: null, error: null };

export function useDetector({ enabled = false } = {}) {
  const [state, setState] = useState(IDLE_STATE);

  const detectorRef = useRef(null);
  const mountedRef = useRef(false);
  // Bumped on every load attempt; a load that finishes after being superseded disposes
  // what it loaded instead of publishing it (see useClassifier for the full story).
  const requestRef = useRef(0);

  const load = useCallback(async () => {
    requestRef.current += 1;
    const token = requestRef.current;
    const isCurrent = () => mountedRef.current && requestRef.current === token;

    const previous = detectorRef.current;
    detectorRef.current = null;
    setState({
      ...IDLE_STATE,
      status: 'loading',
      progress: { stage: 'probing', message: 'Looking for the object detector…', fraction: 0 },
    });
    if (previous) previous.dispose();

    try {
      const detector = await loadDetector({
        onProgress: (progress) => {
          if (isCurrent()) setState((prev) => ({ ...prev, progress }));
        },
      });
      if (!isCurrent()) {
        detector.dispose();
        return null;
      }
      detectorRef.current = detector;
      setState({
        status: 'ready',
        detector,
        progress: { stage: 'ready', message: `Object detector ready on ${detector.backend}.`, fraction: 1 },
        error: null,
      });
      return detector;
    } catch (err) {
      if (!isCurrent()) return null;
      setState({ ...IDLE_STATE, status: 'error', error: err instanceof Error ? err : new Error(String(err)) });
      return null;
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestRef.current += 1;
      if (detectorRef.current) {
        detectorRef.current.dispose();
        detectorRef.current = null;
      }
    };
  }, []);

  const wanted = enabled && state.status === 'idle';
  useEffect(() => {
    // load() turns every failure into state, so there is nothing left to reject.
    if (wanted) void load();
  }, [wanted, load]);

  return {
    detector: state.detector,
    status: state.status,
    progress: state.progress,
    error: state.error,
    ready: state.status === 'ready',
    reload: load,
  };
}

export default useDetector;
