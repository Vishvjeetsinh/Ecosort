import { useCallback, useEffect, useMemo, useState } from 'react';

import * as api from './lib/api.js';
import { categoriesOrFallback, categoryLabel } from './lib/categories.js';
import { formatCount } from './lib/format.js';

import useLocalStorage from './hooks/useLocalStorage.js';
import useToasts from './hooks/useToasts.js';
import useRules from './hooks/useRules.js';
import useHistory from './hooks/useHistory.js';
import useDetector from './hooks/useDetector.js';

// Written by the ML agent: resolve either export style so a mismatch degrades to the
// actionable "no model" screen instead of throwing during the first render.
import * as classifierModule from './hooks/useClassifier.js';
import { downscaleToDataUrl } from './lib/imageUtils.js';

import AppHeader from './components/AppHeader.jsx';
import Tabs, { panelId, tabId } from './components/Tabs.jsx';
import ErrorBanner from './components/ErrorBanner.jsx';
import Toaster from './components/Toaster.jsx';
import Spinner from './components/Spinner.jsx';

import CapturePanel from './components/CapturePanel.jsx';
import LiveScanPanel from './components/LiveScanPanel.jsx';
import ResultsPanel from './components/ResultsPanel.jsx';
import BinColorGuide from './components/BinColorGuide.jsx';
import HistoryPanel from './components/HistoryPanel.jsx';
import StatsPanel from './components/StatsPanel.jsx';
import SettingsPanel from './components/SettingsPanel.jsx';

const UNAVAILABLE_ENGINE = Object.freeze({
  engine: null,
  kind: 'none',
  status: 'error',
  progress: null,
  error: new Error('The classifier module did not load.'),
  ready: false,
  classify: () => Promise.reject(new Error('The classifier is unavailable.')),
  reload: () => {},
});

function useUnavailableClassifier() {
  return UNAVAILABLE_ENGINE;
}

const useClassifier =
  classifierModule.useClassifier || classifierModule.default || useUnavailableClassifier;

/** Matches the backend's MAX_IMAGE_DATA_URL default (characters, not bytes). */
const MAX_IMAGE_DATA_URL = 400_000;
const THUMBNAIL_MAX_SIDE = 320;

/**
 * Past this, switching models is a real wait worth announcing first — the
 * InceptionResNetV2 export alone is ~107 MB of float16 shards.
 */
const LARGE_DOWNLOAD_BYTES = 50 * 1024 * 1024;

const DEFAULT_SETTINGS = Object.freeze({
  autoSave: true,
  saveThumbnails: true,
  topK: 3,
  mirrorWebcam: true,
});

function icon(path) {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" aria-hidden="true">
      <path
        d={path}
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

const TAB_ICONS = {
  classify: icon('M4 7.5A2.5 2.5 0 0 1 6.5 5h1.2l1-1.6h6.6l1 1.6h1.2A2.5 2.5 0 0 1 20 7.5v9A2.5 2.5 0 0 1 17.5 19h-11A2.5 2.5 0 0 1 4 16.5v-9Zm8 9a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z'),
  scan: icon('M4 8V5.5A1.5 1.5 0 0 1 5.5 4H8m8 0h2.5A1.5 1.5 0 0 1 20 5.5V8m0 8v2.5a1.5 1.5 0 0 1-1.5 1.5H16m-8 0H5.5A1.5 1.5 0 0 1 4 18.5V16M7.5 8.5h4v4h-4zm5 3h4v4.5h-4z'),
  guide: icon('M4 5.5A1.5 1.5 0 0 1 5.5 4H11v16H5.5A1.5 1.5 0 0 1 4 18.5v-13Zm9-1.5h5.5A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5H13V4Z'),
  history: icon('M3.5 12a8.5 8.5 0 1 0 2.6-6.1M3.5 5v4h4M12 7.5V12l3 2'),
  stats: icon('M4 20V10m5 10V4m5 16v-7m5 7V8'),
};

const TAB_IDS = ['classify', 'scan', 'guide', 'history', 'stats'];

function prefersDark() {
  try {
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  } catch {
    // matchMedia is absent in some embedded webviews; light is a safe default.
    return false;
  }
}

function isAbort(err) {
  return Boolean(err) && err.name === 'AbortError';
}

function toError(value, fallbackMessage) {
  if (value instanceof Error) return value;
  if (typeof value === 'string' && value.length > 0) return new Error(value);
  return new Error(fallbackMessage);
}

/**
 * Download sizes are quoted 1024-based, the way the README and `make convert-model`
 * quote them, so the 112,262,070-byte InceptionResNetV2 reads as the promised 107 MB.
 */
function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  const mb = n / (1024 * 1024);
  return `${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB`;
}

export default function App() {
  const { toasts, push, dismiss } = useToasts();

  /* ------------------------------------------------------------ shell state */

  const [theme, setTheme] = useLocalStorage('theme', () => (prefersDark() ? 'dark' : 'light'));
  const [activeTab, setActiveTab] = useState('classify');
  const [regionId, setRegionId] = useLocalStorage('regionId', '');
  const [storedSettings, setStoredSettings] = useLocalStorage('settings', DEFAULT_SETTINGS);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [statsDays, setStatsDays] = useLocalStorage('statsDays', 30);
  const [captureMode, setCaptureMode] = useLocalStorage('captureMode', 'webcam');

  // Merged so a stored object written by an older build never drops a new setting.
  const settings = useMemo(
    () => ({ ...DEFAULT_SETTINGS, ...(storedSettings && typeof storedSettings === 'object' ? storedSettings : {}) }),
    [storedSettings],
  );

  // SettingsPanel offers three values - 'light', 'dark' and 'system'. 'system' has to
  // keep tracking the OS preference while it is selected, not just read it once, so the
  // media query stays subscribed for as long as that option is active.
  useEffect(() => {
    const root = document.documentElement;
    const meta = document.getElementById('meta-theme-color');

    const apply = (dark) => {
      root.classList.toggle('dark', dark);
      if (meta) meta.setAttribute('content', dark ? '#020617' : '#f8fafc');
    };

    if (theme !== 'system') {
      apply(theme === 'dark');
      return undefined;
    }

    let query;
    try {
      query = window.matchMedia('(prefers-color-scheme: dark)');
    } catch {
      apply(false);
      return undefined;
    }

    apply(query.matches);
    const onChange = (event) => apply(event.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, [theme]);

  // The header button is a quick light/dark flip. Starting from 'system' it commits to
  // the opposite of whatever the OS is currently showing, which is what a user pressing
  // it expects; 'system' itself stays reachable from the settings panel.
  const toggleTheme = useCallback(() => {
    setTheme((current) => {
      if (current === 'system') return prefersDark() ? 'light' : 'dark';
      return current === 'dark' ? 'light' : 'dark';
    });
  }, [setTheme]);

  const handleSettingsChange = useCallback(
    (next) => {
      setStoredSettings((previous) => {
        const base = { ...DEFAULT_SETTINGS, ...(previous || {}) };
        const patch = typeof next === 'function' ? next(base) : next;
        if (!patch || typeof patch !== 'object') return base;
        return { ...base, ...patch };
      });
    },
    [setStoredSettings],
  );

  /* ------------------------------------------------------ rules & taxonomy */

  const {
    categories: apiCategories,
    regions,
    defaultRegion,
    rules,
    loading: rulesLoading,
    error: rulesError,
    reload: reloadRules,
  } = useRules(regionId);

  const categories = useMemo(() => categoriesOrFallback(apiCategories), [apiCategories]);

  // Adopt the server's default region, and heal a stored id that no longer exists.
  useEffect(() => {
    if (regions.length === 0) return;
    const known = regions.some((region) => region.id === regionId);
    if (known) return;
    const next = regions.some((region) => region.id === defaultRegion)
      ? defaultRegion
      : regions[0].id;
    if (next && next !== regionId) setRegionId(next);
  }, [regions, defaultRegion, regionId, setRegionId]);

  /* ------------------------------------------------------------- inference */

  // '' means "auto" — let the classifier take whatever the backend recommends. A stored
  // id whose model has since been deleted is healed inside loadEngine, not here, so the
  // user is never stranded on a model that is no longer on disk.
  const [modelId, setModelId] = useLocalStorage('modelId', '');

  const {
    engine: classifierEngine,
    kind: engineKind,
    backend: engineBackend,
    status: engineStatus,
    progress: engineProgress,
    error: engineError,
    ready: engineReady,
    classify,
    reload: reloadEngine,
    modelId: engineModelId,
    displayName: engineDisplayName,
    downloadBytes: engineDownloadBytes,
    requestedModelId,
    inputSize: engineInputSize,
  } = useClassifier({ modelId });

  // Grab the frame at the resolution the active model actually consumes, so a 299-input
  // model (the Inception family) sees real pixels instead of a 224 capture upsampled by
  // the classifier. The floor keeps a capture taken before the engine is ready usable by
  // any supported model.
  const captureSize = Math.max(320, engineInputSize || 0);

  const engineBlocked = engineStatus === 'error' || engineKind === 'none';

  // Only Live scan needs the ~18 MB detector: it loads the first time that tab opens and
  // then stays loaded, so switching tabs back and forth costs nothing.
  const scanDetector = useDetector({ enabled: activeTab === 'scan' });

  const [modelStatus, setModelStatus] = useState(null);
  const [modelStatusToken, setModelStatusToken] = useState(0);
  const [models, setModels] = useState([]);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    api
      .getModelStatus(controller.signal)
      .then((status) => {
        if (active) setModelStatus(status);
      })
      .catch((err) => {
        if (!active || isAbort(err)) return;
        // Non-fatal: the badge falls back to whatever the engine itself reports.
        console.warn('[ecosort] Could not read /api/model/status:', err.message);
        setModelStatus(null);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [modelStatusToken]);

  // The registry is re-read once the engine reports ready, because a model converted
  // while the app is open (`make convert-model`) should appear in the picker without a
  // page reload — the backend discovers model directories per request.
  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    api
      .listModels(controller.signal)
      .then((response) => {
        if (!active) return;
        setModels(Array.isArray(response?.models) ? response.models : []);
      })
      .catch((err) => {
        if (!active || isAbort(err)) return;
        // A backend from before the registry has no such route; an empty list simply
        // hides the picker and the classifier keeps its own custom → fallback order.
        if (err.status !== 404) {
          console.warn('[ecosort] Could not list the available models:', err.message);
        }
        setModels([]);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [engineReady, modelStatusToken]);

  const recommendedModel = useMemo(
    () => models.find((model) => model.recommended) || null,
    [models],
  );

  const handleModelChange = useCallback(
    (nextId) => {
      const id = typeof nextId === 'string' ? nextId : '';
      setModelId(id);

      // '' resolves to whatever the backend recommends, so the toast names that model
      // rather than the word "auto" — the download that follows is the recommended one's.
      const chosen = models.find((model) => model.id === id) || (id ? null : recommendedModel);
      const name = chosen ? chosen.displayName || chosen.id : id || 'the recommended model';
      const bytes = chosen ? Number(chosen.downloadBytes) : NaN;
      const size = formatBytes(bytes);

      push({
        tone: 'info',
        title: `Loading ${name}`,
        message:
          size && bytes > LARGE_DOWNLOAD_BYTES
            ? `This one is ${size} — the first load downloads it into this browser, then it is cached.`
            : 'Loading the weights into this tab. Nothing is uploaded.',
      });
    },
    [models, push, recommendedModel, setModelId],
  );

  const handleReloadEngine = useCallback(() => {
    setModelStatusToken((n) => n + 1);
    reloadEngine();
    push({
      tone: 'info',
      title: 'Reloading the classifier',
      message: 'Re-probing the backend for a custom model, then loading weights.',
    });
  }, [push, reloadEngine]);

  /* --------------------------------------------------------------- history */

  const history = useHistory();

  /* ----------------------------------------------------------------- stats */

  const [stats, setStats] = useState(null);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState(null);
  const [statsToken, setStatsToken] = useState(0);

  const refreshStats = useCallback(() => setStatsToken((n) => n + 1), []);

  useEffect(() => {
    if (activeTab !== 'stats' || !regionId) return undefined;

    const controller = new AbortController();
    let active = true;

    setStatsLoading(true);
    api
      .getStats({ regionId, days: statsDays }, controller.signal)
      .then((response) => {
        if (!active) return;
        setStats(response || null);
        setStatsError(null);
      })
      .catch((err) => {
        if (!active || isAbort(err)) return;
        setStats(null);
        setStatsError(err);
      })
      .finally(() => {
        if (active) setStatsLoading(false);
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [activeTab, regionId, statsDays, statsToken]);

  /* ------------------------------------------------------ capture & result */

  const [capture, setCapture] = useState(null);
  const [captureError, setCaptureError] = useState(null);
  const [result, setResult] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState(null);
  const [classifying, setClassifying] = useState(false);
  const [classifyError, setClassifyError] = useState(null);
  const [savedItem, setSavedItem] = useState(null);
  const [saving, setSaving] = useState(false);

  const persistResult = useCallback(
    async (classifyResult, shot) => {
      if (!classifyResult || !shot) return null;
      if (!regionId) {
        push({
          tone: 'error',
          title: 'Cannot save yet',
          message: 'Pick a recycling region first — history rows are stored per region.',
        });
        return null;
      }

      setSaving(true);
      try {
        let imageDataUrl = null;
        if (settings.saveThumbnails) {
          try {
            let thumb = shot.canvas
              ? downscaleToDataUrl(shot.canvas, { maxSide: THUMBNAIL_MAX_SIDE })
              : null;
            if (typeof thumb !== 'string' || thumb.length === 0) thumb = shot.dataUrl;
            if (typeof thumb === 'string' && thumb.length > 0) {
              if (thumb.length <= MAX_IMAGE_DATA_URL) {
                imageDataUrl = thumb;
              } else {
                console.warn(
                  `[ecosort] Thumbnail is ${thumb.length} chars (limit ${MAX_IMAGE_DATA_URL}); saving without an image.`,
                );
              }
            }
          } catch (err) {
            // A missing thumbnail must never block saving the classification itself.
            console.warn('[ecosort] Could not build a thumbnail:', err);
          }
        }

        const body = {
          predictions: classifyResult.predictions.slice(0, 10).map((prediction) => ({
            category: prediction.category,
            label: prediction.label,
            confidence: prediction.confidence,
          })),
          source: shot.source === 'webcam' ? 'webcam' : 'upload',
          modelKind: classifyResult.modelKind === 'custom' ? 'custom' : 'fallback',
          regionId,
          rawLabels: Array.isArray(classifyResult.rawLabels) ? classifyResult.rawLabels : null,
          durationMs: Number.isFinite(classifyResult.durationMs)
            ? Math.round(classifyResult.durationMs)
            : null,
        };
        if (imageDataUrl) body.imageDataUrl = imageDataUrl;

        const response = await api.createClassification(body);
        const item = response?.item || null;
        setSavedItem(item);
        history.refresh();
        refreshStats();
        push({
          tone: 'success',
          title: 'Saved to history',
          message: `${categoryLabel(categories, classifyResult.predictions[0].category)} recorded for this region.`,
        });
        return item;
      } catch (err) {
        push({ tone: 'error', title: 'Could not save to history', message: err.message });
        return null;
      } finally {
        setSaving(false);
      }
    },
    [categories, history.refresh, push, refreshStats, regionId, settings.saveThumbnails],
  );

  const runClassification = useCallback(
    async (shot) => {
      if (!shot || !shot.canvas) {
        setClassifyError(new Error('There is no image to classify yet.'));
        return;
      }

      setClassifying(true);
      setClassifyError(null);
      setResult(null);
      setSavedItem(null);

      try {
        const classifyResult = await classify(shot.canvas, { topK: settings.topK });
        if (
          !classifyResult ||
          !Array.isArray(classifyResult.predictions) ||
          classifyResult.predictions.length === 0
        ) {
          throw new Error('The classifier returned no predictions for this image.');
        }

        setResult(classifyResult);
        setSelectedCategory(classifyResult.predictions[0].category);

        if (classifyResult.lowConfidence) {
          push({
            tone: 'info',
            title: 'Low confidence',
            message:
              'The model is unsure about this item. Try better lighting, a plain background, or a closer shot.',
          });
        }

        if (settings.autoSave) {
          await persistResult(classifyResult, shot);
        }
      } catch (err) {
        const error = toError(err, 'The classifier failed for an unknown reason.');
        setClassifyError(error);
        push({ tone: 'error', title: 'Classification failed', message: error.message });
      } finally {
        setClassifying(false);
      }
    },
    [classify, persistResult, push, settings.autoSave, settings.topK],
  );

  const handleImageReady = useCallback(
    (shot) => {
      if (!shot) return;
      setCapture(shot);
      setCaptureError(null);
      // runClassification owns its failures, so there is no rejection to handle here.
      void runClassification(shot);
    },
    [runClassification],
  );

  const handleCaptureError = useCallback(
    (err) => {
      if (!err) {
        setCaptureError(null);
        return;
      }
      const error = toError(err, 'The camera or file could not be read.');
      setCaptureError(error);
      push({ tone: 'error', title: 'Capture problem', message: error.message });
    },
    [push],
  );

  const handleSave = useCallback(() => {
    if (!result || !capture) return;
    void persistResult(result, capture);
  }, [capture, persistResult, result]);

  /* ------------------------------------------------------------ live scan */

  const [scanSaving, setScanSaving] = useState(false);

  // One history row per item found in a frozen frame or photo, each with its own crop as
  // the thumbnail - the history and stats then count items, exactly as for single shots.
  const handleSaveScan = useCallback(
    async (items, { source, durationMs } = {}) => {
      if (!regionId) {
        push({
          tone: 'error',
          title: 'Cannot save yet',
          message: 'Pick a recycling region first — history rows are stored per region.',
        });
        return;
      }
      const list = (Array.isArray(items) ? items : []).filter(
        (item) => Array.isArray(item?.predictions) && item.predictions.length > 0,
      );
      if (list.length === 0) return;

      setScanSaving(true);
      let saved = 0;
      let firstError = null;
      try {
        for (const item of list) {
          const body = {
            predictions: item.predictions.slice(0, 10).map((prediction) => ({
              category: prediction.category,
              label: prediction.label,
              confidence: Math.min(1, Math.max(0, prediction.confidence)),
            })),
            source: source === 'webcam' ? 'webcam' : 'upload',
            modelKind: engineKind === 'custom' ? 'custom' : 'fallback',
            regionId,
            rawLabels: null,
            durationMs: Number.isFinite(durationMs) ? Math.round(durationMs) : null,
          };
          const thumb = item.imageDataUrl;
          if (settings.saveThumbnails && typeof thumb === 'string' && thumb.length <= MAX_IMAGE_DATA_URL) {
            body.imageDataUrl = thumb;
          }
          try {
            await api.createClassification(body);
            saved += 1;
          } catch (err) {
            firstError = firstError || err;
          }
        }
      } finally {
        setScanSaving(false);
      }

      if (saved > 0) {
        history.refresh();
        refreshStats();
      }
      if (firstError) {
        push({
          tone: 'error',
          title: saved > 0 ? `Saved ${saved} of ${list.length} items` : 'Could not save to history',
          message: firstError.message,
        });
      } else {
        push({
          tone: 'success',
          title: 'Saved to history',
          message: `${saved === 1 ? '1 item' : `${saved} items`} recorded for this region.`,
        });
      }
    },
    [engineKind, history.refresh, push, refreshStats, regionId, settings.saveThumbnails],
  );

  const handleCorrectCurrent = useCallback(
    async (categoryId) => {
      if (!categoryId) return;
      setSelectedCategory(categoryId);

      if (!savedItem) {
        push({
          tone: 'info',
          title: 'Not saved yet',
          message: 'Save this classification first, then its category can be corrected.',
        });
        return;
      }

      try {
        const response = await api.updateClassification(savedItem.id, {
          correctedCategory: categoryId,
        });
        if (response?.item) setSavedItem(response.item);
        history.refresh();
        refreshStats();
        push({
          tone: 'success',
          title: 'Correction saved',
          message: `Recorded as ${categoryLabel(categories, categoryId)}.`,
        });
      } catch (err) {
        push({ tone: 'error', title: 'Could not save the correction', message: err.message });
      }
    },
    [categories, history.refresh, push, refreshStats, savedItem],
  );

  const handleHistoryDelete = useCallback(
    async (id) => {
      try {
        await history.remove(id);
        refreshStats();
        if (savedItem && savedItem.id === id) setSavedItem(null);
        push({ tone: 'success', title: 'Entry deleted', message: 'The history row is gone.' });
      } catch (err) {
        push({ tone: 'error', title: 'Could not delete', message: err.message });
      }
    },
    [history.remove, push, refreshStats, savedItem],
  );

  const handleHistoryCorrect = useCallback(
    async (id, categoryId) => {
      try {
        const updated = await history.correct(id, categoryId);
        refreshStats();
        if (savedItem && savedItem.id === id) setSavedItem(updated);
        push({
          tone: 'success',
          title: 'Correction saved',
          message: `Recorded as ${categoryLabel(categories, categoryId)}.`,
        });
      } catch (err) {
        push({ tone: 'error', title: 'Could not save the correction', message: err.message });
      }
    },
    [categories, history.correct, push, refreshStats, savedItem],
  );

  const handleClearAll = useCallback(async () => {
    try {
      await history.clearAll();
      refreshStats();
      setSavedItem(null);
      push({
        tone: 'success',
        title: 'History cleared',
        message: 'Every stored classification has been deleted.',
      });
    } catch (err) {
      push({ tone: 'error', title: 'Could not clear history', message: err.message });
    }
  }, [history.clearAll, push, refreshStats]);

  /* ------------------------------------------------- derived region guidance */

  const guidance = useMemo(() => {
    if (!rules || !selectedCategory) return null;
    const byCategory = rules.categories;
    if (!byCategory || typeof byCategory !== 'object') return null;
    return byCategory[selectedCategory] || null;
  }, [rules, selectedCategory]);

  const bin = useMemo(() => {
    if (!rules || !guidance) return null;
    const bins = Array.isArray(rules.bins) ? rules.bins : [];
    return bins.find((candidate) => candidate.id === guidance.binId) || null;
  }, [guidance, rules]);

  const [guideQuery, setGuideQuery] = useState('');

  const tabItems = useMemo(
    () => [
      { id: 'classify', label: 'Classify', icon: TAB_ICONS.classify },
      { id: 'scan', label: 'Live scan', icon: TAB_ICONS.scan },
      { id: 'guide', label: 'Bin guide', icon: TAB_ICONS.guide },
      {
        id: 'history',
        label: 'History',
        icon: TAB_ICONS.history,
        badge: history.total > 0 ? formatCount(history.total) : null,
      },
      { id: 'stats', label: 'Stats', icon: TAB_ICONS.stats },
    ],
    [history.total],
  );

  const handleTabChange = useCallback((next) => {
    if (TAB_IDS.includes(next)) setActiveTab(next);
  }, []);

  /* ----------------------------------------------------------------- render */

  return (
    <div className="flex min-h-full flex-col">
      <AppHeader
        modelStatus={modelStatus}
        engineKind={engineKind}
        engineStatus={engineStatus}
        engineProgress={engineProgress}
        engineDisplayName={engineDisplayName}
        models={models}
        modelId={modelId}
        onModelChange={handleModelChange}
        requestedModelId={requestedModelId}
        activeModelId={engineModelId}
        regions={regions}
        regionId={regionId}
        onRegionChange={setRegionId}
        onOpenSettings={() => setSettingsOpen(true)}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="mx-auto w-full max-w-shell flex-1 px-3 py-4 sm:px-5">
        <Tabs value={activeTab} onChange={handleTabChange} items={tabItems} label="EcoSort sections" />

        {rulesError ? (
          <div className="mt-4">
            <ErrorBanner
              title="Could not load the recycling rules"
              error={rulesError}
              onRetry={reloadRules}
            />
          </div>
        ) : null}

        {activeTab === 'classify' ? (
          <section
            role="tabpanel"
            id={panelId('classify')}
            aria-labelledby={tabId('classify')}
            tabIndex={-1}
            className="mt-4 grid gap-4 lg:grid-cols-2"
          >
            <div className="flex flex-col gap-4">
              {engineBlocked ? <EngineUnavailable error={engineError} onRetry={handleReloadEngine} /> : null}

              {!engineBlocked && !engineReady ? (
                <EngineLoading progress={engineProgress} />
              ) : null}

              <CapturePanel
                mode={captureMode}
                onModeChange={setCaptureMode}
                onImageReady={handleImageReady}
                busy={classifying}
                disabled={engineBlocked || !engineReady}
                error={captureError}
                onError={handleCaptureError}
                mirror={settings.mirrorWebcam}
                captureSize={captureSize}
              />
            </div>

            <div className="flex flex-col gap-4">
              {/* Owned here rather than passed to ResultsPanel: this banner carries the
                  retry/dismiss actions, and ResultsPanel keeps showing the last result. */}
              {classifyError ? (
                <ErrorBanner
                  title="Classification failed"
                  error={classifyError}
                  onRetry={capture ? () => void runClassification(capture) : undefined}
                  onDismiss={() => setClassifyError(null)}
                />
              ) : null}

              <ResultsPanel
                result={result}
                guidance={guidance}
                bin={bin}
                rules={rules}
                region={rules ? rules.region : null}
                categories={categories}
                busy={classifying || rulesLoading}
                engineKind={engineKind}
                imageDataUrl={capture ? capture.dataUrl : null}
                selectedCategoryId={selectedCategory}
                onSelectCategory={setSelectedCategory}
                onSave={handleSave}
                onSaved={handleSave}
                onCorrect={handleCorrectCurrent}
                saving={saving}
                saved={Boolean(savedItem)}
              />
            </div>
          </section>
        ) : null}

        {activeTab === 'scan' ? (
          <section
            role="tabpanel"
            id={panelId('scan')}
            aria-labelledby={tabId('scan')}
            tabIndex={-1}
            className="mt-4"
          >
            <LiveScanPanel
              detector={scanDetector}
              engine={classifierEngine}
              engineReady={engineReady}
              engineProgress={engineProgress}
              engineBlocked={engineBlocked}
              rules={rules}
              categories={categories}
              mirror={settings.mirrorWebcam}
              topK={settings.topK}
              onSaveItems={handleSaveScan}
              saving={scanSaving}
            />
          </section>
        ) : null}

        {activeTab === 'guide' ? (
          <section
            role="tabpanel"
            id={panelId('guide')}
            aria-labelledby={tabId('guide')}
            tabIndex={-1}
            className="mt-4"
          >
            {rulesLoading && !rules ? (
              <LoadingBlock label="Loading the bin guide" />
            ) : (
              <BinColorGuide
                rules={rules}
                categories={categories}
                query={guideQuery}
                onQueryChange={setGuideQuery}
              />
            )}
          </section>
        ) : null}

        {activeTab === 'history' ? (
          <section
            role="tabpanel"
            id={panelId('history')}
            aria-labelledby={tabId('history')}
            tabIndex={-1}
            className="mt-4 flex flex-col gap-4"
          >
            <HistoryPanel
              items={history.items}
              total={history.total}
              loading={history.loading}
              error={history.error}
              filters={history.filters}
              onFiltersChange={history.setFilters}
              onDelete={handleHistoryDelete}
              onCorrect={handleHistoryCorrect}
              onClearAll={handleClearAll}
              categories={categories}
              regions={regions}
              rules={rules}
            />
          </section>
        ) : null}

        {activeTab === 'stats' ? (
          <section
            role="tabpanel"
            id={panelId('stats')}
            aria-labelledby={tabId('stats')}
            tabIndex={-1}
            className="mt-4 flex flex-col gap-4"
          >
            <StatsPanel
              stats={stats}
              loading={statsLoading}
              error={statsError}
              categories={categories}
              days={statsDays}
              onDaysChange={setStatsDays}
              regionId={regionId}
              regions={regions}
            />
          </section>
        ) : null}
      </main>

      <footer className="border-t border-slate-200 px-3 py-4 text-center text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400 sm:px-5">
        EcoSort runs entirely on this machine — images are classified in your browser and
        never leave it.
      </footer>

      <SettingsPanel
        open={settingsOpen}
        onClose={() => setSettingsOpen(false)}
        settings={settings}
        onChange={handleSettingsChange}
        modelStatus={modelStatus}
        onReloadEngine={handleReloadEngine}
        theme={theme}
        onThemeChange={setTheme}
        engineKind={engineKind}
        engineBackend={engineBackend}
        models={models}
        modelId={modelId}
        onModelChange={handleModelChange}
        engineModelId={engineModelId}
        engineDownloadBytes={engineDownloadBytes}
        requestedModelId={requestedModelId}
      />

      <Toaster toasts={toasts} onDismiss={dismiss} />
    </div>
  );
}

/**
 * The first load pulls ~14 MB of weights, so the wait needs a reason on screen —
 * a disabled capture panel with no explanation reads as a broken app.
 */
function EngineLoading({ progress }) {
  const fraction =
    progress && Number.isFinite(progress.fraction)
      ? Math.round(Math.min(1, Math.max(0, progress.fraction)) * 100)
      : null;

  return (
    <div className="card flex items-center gap-3 p-3 text-sm text-slate-700 dark:text-slate-200">
      <Spinner size={18} label="Loading the classifier" />
      <div className="min-w-0 flex-1">
        <p className="font-medium">Preparing the classifier…</p>
        <p className="truncate text-xs text-slate-500 dark:text-slate-400">
          {progress?.message || 'Loading the model into your browser. Nothing is uploaded.'}
        </p>
      </div>
      {fraction !== null ? (
        <span className="shrink-0 font-mono text-xs tabular-nums text-slate-500 dark:text-slate-400">
          {fraction}%
        </span>
      ) : null}
    </div>
  );
}

function LoadingBlock({ label }) {
  return (
    <div className="card flex items-center justify-center gap-3 p-10 text-sm text-slate-600 dark:text-slate-300">
      <Spinner size={18} label={label} />
      <span>{label}…</span>
    </div>
  );
}

/**
 * Shown when neither a custom model nor the MobileNetV2 fallback could be loaded.
 * The exact commands matter here — this is the one screen a first-time user hits
 * when the build-time weight download was skipped or failed.
 */
function EngineUnavailable({ error, onRetry }) {
  return (
    <div className="card border-amber-300 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40">
      <h2 className="text-base font-semibold text-amber-900 dark:text-amber-100">
        No classifier model is available
      </h2>
      <p className="mt-1 text-sm text-amber-900/90 dark:text-amber-100/90">
        EcoSort needs at least one model in <code className="font-mono">models/</code> before
        it can classify anything. Nothing is downloaded at runtime, so one of these has to be
        put in place first — any single one is enough, and installing several lets you switch
        between them in Settings.
      </p>

      <ol className="mt-3 space-y-3 text-sm text-amber-900 dark:text-amber-100">
        <li>
          <p className="font-medium">1. Fetch the pretrained fallback (fastest)</p>
          <pre className="mt-1 overflow-x-auto rounded-lg bg-amber-100 p-2 font-mono text-xs text-amber-950 dark:bg-amber-900/50 dark:text-amber-50">
{`make fetch-models
docker compose restart backend`}
          </pre>
          <p className="mt-1 text-xs opacity-80">
            Without Make:{' '}
            <code className="font-mono">node scripts/fetch-mobilenet.mjs</code>
          </p>
        </li>
        <li>
          <p className="font-medium">2. Or train your own model on waste photos</p>
          <pre className="mt-1 overflow-x-auto rounded-lg bg-amber-100 p-2 font-mono text-xs text-amber-950 dark:bg-amber-900/50 dark:text-amber-50">
{`make train
docker compose restart backend`}
          </pre>
          <p className="mt-1 text-xs opacity-80">
            See <code className="font-mono">ml/README.md</code> for the expected dataset
            layout; the export lands in <code className="font-mono">models/custom/</code>.
          </p>
        </li>
        <li>
          <p className="font-medium">3. Or convert a larger pretrained model</p>
          <pre className="mt-1 overflow-x-auto rounded-lg bg-amber-100 p-2 font-mono text-xs text-amber-950 dark:bg-amber-900/50 dark:text-amber-50">
{`make convert-model ARCH=InceptionResNetV2`}
          </pre>
          <p className="mt-1 text-xs opacity-80">
            More accurate than MobileNetV2, but ~107 MB for the browser to download once.
            It lands in <code className="font-mono">models/inceptionresnetv2/</code> and is
            picked up without a restart — press <strong>Check again</strong> below.
          </p>
        </li>
      </ol>

      {error && error.message ? (
        <p className="mt-3 font-mono text-[11px] text-amber-900/80 dark:text-amber-100/70">
          {error.message}
        </p>
      ) : null}

      <button type="button" className="btn btn-ghost mt-3" onClick={onRetry}>
        Check again
      </button>
    </div>
  );
}
