/**
 * React binding for `lib/classifier.js`.
 *
 * Loading a model is expensive — 14 MB of weights for MobileNetV2, ~107 MB for
 * InceptionResNetV2, plus a warm-up inference — so exactly one engine exists at
 * a time. It is created on mount, re-created when the selected `modelId`
 * changes (the model picker) or when `reload()` is called (what SettingsPanel
 * does once the user has trained a custom model), and disposed on unmount.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { loadEngine } from '../lib/classifier.js';

const IDLE_STATE = {
  status: 'idle',
  engine: null,
  kind: null,
  backend: null,
  modelId: null,
  displayName: null,
  downloadBytes: null,
  requestedModelId: null,
  progress: null,
  error: null,
};

/** '' is the stored "auto" selection (ARCHITECTURE 2.4); so is a missing id. */
function normaliseId(modelId) {
  return typeof modelId === 'string' && modelId.trim() !== '' ? modelId.trim() : null;
}

/**
 * @param {{modelId?: string|null, autoLoad?: boolean, topK?: number}} [options]
 *   `modelId` is the user's choice; '' or null means "let the backend decide".
 */
export function useClassifier({ modelId = null, autoLoad = true, topK = 3 } = {}) {
  const [state, setState] = useState(IDLE_STATE);

  const engineRef = useRef(null);
  const mountedRef = useRef(false);
  // Bumped on every load attempt. tfjs gives no handle to abort a download in
  // flight, so "cancelling" a superseded load means refusing its result and
  // disposing the engine it eventually hands back — otherwise a slow switch
  // back and forth could leave two models holding GPU memory, or let an older
  // request overwrite the state of a newer one.
  const requestRef = useRef(0);

  const load = useCallback(async (requestedId) => {
    requestRef.current += 1;
    const token = requestRef.current;
    const requestedModelId = normaliseId(requestedId);
    const isCurrent = () => mountedRef.current && requestRef.current === token;

    // Detach before disposing: a disposed engine must never be reachable from
    // state, not even for a single render.
    const previous = engineRef.current;
    engineRef.current = null;
    setState({
      ...IDLE_STATE,
      status: 'loading',
      requestedModelId,
      progress: { stage: 'probing', message: 'Looking for a local model…', fraction: 0 },
    });
    if (previous) previous.dispose();

    try {
      const engine = await loadEngine({
        modelId: requestedModelId,
        onProgress: (progress) => {
          if (isCurrent()) setState((prev) => ({ ...prev, progress }));
        },
      });

      // A newer selection — or an unmount — won the race while this loaded.
      if (!isCurrent()) {
        engine.dispose();
        return null;
      }

      engineRef.current = engine;
      setState({
        status: 'ready',
        engine,
        kind: engine.kind,
        backend: engine.backend,
        modelId: engine.modelId,
        displayName: engine.displayName,
        downloadBytes: engine.downloadBytes,
        // Differs from modelId when the stored choice no longer exists, so the
        // UI can say which model it actually got.
        requestedModelId: engine.requestedModelId,
        progress: {
          stage: 'ready',
          message: `Ready — ${engine.displayName || engine.kind} on ${engine.backend}.`,
          fraction: 1,
        },
        error: null,
      });
      return engine;
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      if (!isCurrent()) return null;
      setState({ ...IDLE_STATE, status: 'error', requestedModelId, error });
      return null;
    }
  }, []);

  // A new identity on every selection change, which is what drives both the
  // reload effect below and reload() re-resolving the *current* id.
  const loadSelected = useCallback(() => load(modelId), [load, modelId]);

  // Declared first so it runs before the loading effect on mount, and so its
  // cleanup marks the hook unmounted before that effect starts disposing.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (autoLoad) {
      // The promise is intentionally not awaited here; all failures are turned
      // into state by load() itself, so there is nothing left to reject.
      void loadSelected();
    }

    return () => {
      // Supersede anything in flight and free the weights immediately: the next
      // model must not have to share GPU memory with the one it replaces.
      requestRef.current += 1;
      if (engineRef.current) {
        engineRef.current.dispose();
        engineRef.current = null;
      }
    };
  }, [autoLoad, loadSelected]);

  const classify = useCallback(
    async (source, options = {}) => {
      const engine = engineRef.current;
      if (!engine) {
        throw new Error('The classifier is not ready yet — wait for the model to finish loading.');
      }
      return engine.classify(source, { topK, ...options });
    },
    [topK],
  );

  const reload = useCallback(() => loadSelected(), [loadSelected]);

  return {
    engine: state.engine,
    kind: state.kind,
    backend: state.backend,
    modelId: state.modelId,
    displayName: state.displayName,
    downloadBytes: state.downloadBytes,
    requestedModelId: state.requestedModelId,
    // Drives the capture resolution: 224 for the MobileNets, 299 for the Inception
    // family. Read off the live engine so it is always the size actually in use.
    inputSize: state.engine ? state.engine.inputSize : null,
    status: state.status,
    progress: state.progress,
    error: state.error,
    ready: state.status === 'ready',
    classify,
    reload,
  };
}

export default useClassifier;
