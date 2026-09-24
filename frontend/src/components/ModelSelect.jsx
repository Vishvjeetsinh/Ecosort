import { useId } from 'react';

import { formatPercent } from '../lib/format.js';
import Spinner from './Spinner.jsx';

const MB = 1024 * 1024;

// Past this the download deserves a warning *before* the tap rather than a surprise
// progress bar after it — the largest registry entry (InceptionResNetV2, float16) is
// ~107 MB, which is a real cost on a phone.
const LARGE_DOWNLOAD_BYTES = 50 * MB;

// Used only when the caller cannot tell us the real model name (an older caller, or the
// registry has not arrived yet). Same wording as ModelStatusBadge so the two never
// disagree about what is running.
const KIND_NAMES = {
  custom: 'Custom model',
  fallback: 'MobileNet fallback',
  none: 'No model',
};

const STATES = {
  ready_custom: {
    dot: 'bg-emerald-500',
    text: 'text-emerald-700 dark:text-emerald-300',
    title:
      'Loaded: your own model from models/custom — it was trained on waste photos, so it predicts waste categories directly.',
  },
  ready_fallback: {
    dot: 'bg-amber-500',
    text: 'text-amber-700 dark:text-amber-300',
    title:
      'Loaded: a pretrained ImageNet model. Its 1000 object classes are mapped onto waste categories — accurate for common objects, approximate for the rest.',
  },
  error: {
    dot: 'bg-red-500',
    text: 'text-red-700 dark:text-red-300',
    title:
      'No model is loaded. Pick another one below, or run `make fetch-models` to download the pretrained weights.',
  },
  idle: {
    dot: 'bg-slate-400 dark:bg-slate-500',
    text: 'text-slate-600 dark:text-slate-300',
    title: 'The classifier has not been loaded into this tab yet.',
  },
  loading: {
    dot: 'bg-slate-400 dark:bg-slate-500',
    text: 'text-slate-600 dark:text-slate-300',
    title: 'Loading the classifier into your browser. Nothing is sent to a server.',
  },
};

function describeState(status, kind) {
  if (status === 'error') return STATES.error;
  if (status === 'loading') return STATES.loading;
  if (status === 'ready') {
    if (kind === 'custom') return STATES.ready_custom;
    if (kind === 'fallback') return STATES.ready_fallback;
  }
  return STATES.idle;
}

/**
 * Download sizes come from the weight manifest, so they are exact byte counts. One
 * decimal is all it takes to tell 13.3 MB from 107.1 MB without pretending to a
 * precision nobody reads.
 *
 * @param {number|string|null|undefined} bytes
 * @returns {string|null} null when the size is unknown — better to say nothing than "0.0 MB"
 */
function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return null;
  const mb = n / MB;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`;
  return `${mb.toFixed(1)} MB`;
}

function downloadBytesOf(model) {
  if (!model) return 0;
  // The backend computes `downloadBytes` from the manifest; metadata.json may also carry
  // it when the converter wrote one. Either is authoritative, neither is guaranteed.
  const bytes = model.downloadBytes ?? model.metadata?.downloadBytes;
  const n = Number(bytes);
  return Number.isFinite(n) && n > 0 ? n : 0;
}

function nameOf(model) {
  return model?.displayName || model?.id || '';
}

function optionLabel(model) {
  const size = formatBytes(downloadBytesOf(model));
  const base = size ? `${nameOf(model)} — ${size}` : nameOf(model);
  // Marking the auto-pick inline keeps "Auto (recommended)" honest: the user can see
  // which entry it resolves to without opening settings.
  return model?.recommended ? `${base} · recommended` : base;
}

/**
 * The model picker. A native <select> on purpose: it is keyboard- and screen-reader
 * accessible for free and renders as the platform wheel/menu on mobile, which no custom
 * popover matches.
 *
 * @param {{models?:Array<object>, value?:string, onChange?:Function,
 *          status?:'idle'|'loading'|'ready'|'error', kind?:'custom'|'fallback'|'none',
 *          displayName?:string, progress?:{stage?:string,message?:string,fraction?:number}|null,
 *          requestedModelId?:string, activeModelId?:string, disabled?:boolean,
 *          label?:string, className?:string}} props
 */
export function ModelSelect({
  models = [],
  value = '',
  onChange,
  status = 'idle',
  kind = 'none',
  displayName = '',
  progress = null,
  requestedModelId = '',
  activeModelId = '',
  disabled = false,
  label = 'Classification model',
  className = '',
}) {
  const selectId = useId();
  const hintId = `${selectId}-hint`;

  const catalogue = Array.isArray(models) ? models : [];
  const available = catalogue.filter((model) => model && model.id && model.available !== false);
  const byId = (id) => (id ? available.find((model) => model.id === id) || null : null);

  const recommended = available.find((model) => model.recommended) || null;
  // An empty value means "auto", which resolves to the recommended entry — that is the
  // download the browser will actually make, so the size warning has to follow it too.
  const selected = value ? byId(value) : recommended;
  // A stored id whose directory has since been deleted must not silently read as "Auto":
  // the select would then show a choice the user never made.
  const stale = Boolean(value) && !byId(value);

  const busy = status === 'loading';
  const percent =
    busy && progress && Number.isFinite(progress.fraction)
      ? formatPercent(progress.fraction, 0)
      : null;

  const state = describeState(status, kind);
  const activeName = displayName || KIND_NAMES[kind] || KIND_NAMES.none;

  const requested = requestedModelId
    ? catalogue.find((model) => model && model.id === requestedModelId) || null
    : null;
  const requestedName = nameOf(requested) || requestedModelId;
  // Only meaningful once an engine is actually up: mid-load `displayName` still names the
  // previous model, and on error there is nothing running to be "used instead".
  // Compare the id the engine was ASKED for against the id it actually loaded. Deriving
  // this from the catalogue instead would report a phantom substitution whenever the
  // separate /api/models request fails, since an empty catalogue makes every id look gone.
  const substituted =
    status === 'ready' && Boolean(requestedModelId) && Boolean(activeModelId)
      ? requestedModelId !== activeModelId
      : false;

  const largeSize =
    downloadBytesOf(selected) > LARGE_DOWNLOAD_BYTES
      ? formatBytes(downloadBytesOf(selected))
      : null;

  const autoTitle = recommended
    ? `Auto — currently ${optionLabel(recommended)}`
    : 'Auto — use whichever model the backend recommends';
  // Native selects clip their own text, so the untruncated label lives in the tooltip.
  const selectTitle = value
    ? (selected ? optionLabel(selected) : `${value} — unavailable`)
    : autoTitle;

  return (
    <div className={`flex min-w-0 flex-col gap-1 ${className}`.trim()}>
      <label htmlFor={selectId} className="sr-only">
        {label}
      </label>

      {/* The status line sits ABOVE the control rather than beside it. Inline, it and the
          select shared one narrow column and the model name in the select was truncated to
          a stub — the one piece of text the picker exists to show. */}
      <div className="flex min-w-0 flex-col gap-1">
        {/* Colour is never the only signal: the dot is paired with the model's name. */}
        <span
          className={`flex min-w-0 items-center gap-1.5 text-xs font-medium ${state.text}`}
          title={state.title}
          aria-label={`${label} status: ${activeName}${percent ? `, loading ${percent}` : ''}. ${state.title}`}
        >
          {busy ? (
            <Spinner size={12} label="Loading model" />
          ) : (
            <span className={`h-2 w-2 shrink-0 rounded-full ${state.dot}`} aria-hidden="true" />
          )}
          <span className="min-w-0 truncate" title={activeName}>
            {activeName}
          </span>
          {percent ? <span className="tabular-nums">{percent}</span> : null}
        </span>

        <select
          id={selectId}
          className="input w-full min-w-0 py-1.5"
          value={value || ''}
          title={
            available.length === 0
              ? 'The backend did not report any selectable models.'
              : selectTitle
          }
          // Swapping weights mid-load would race the engine that is already loading.
          disabled={disabled || busy || available.length === 0}
          aria-label={label}
          aria-describedby={largeSize ? hintId : undefined}
          onChange={(event) => onChange && onChange(event.target.value)}
        >
          <option value="" title={autoTitle}>
            Auto (recommended)
          </option>
          {stale ? (
            <option value={value} disabled>
              {`${value} — unavailable`}
            </option>
          ) : null}
          {available.map((model) => (
            <option key={model.id} value={model.id} title={model.description || undefined}>
              {optionLabel(model)}
            </option>
          ))}
        </select>
      </div>

      {largeSize ? (
        <p id={hintId} className="text-xs text-slate-500 dark:text-slate-400">
          First load downloads {largeSize}; it is cached by the browser afterwards.
        </p>
      ) : null}

      {substituted ? (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-lg border border-amber-300 bg-amber-50 px-2 py-1 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200">
          <span className="min-w-0 flex-1 break-words">
            <strong className="font-semibold">{requestedName}</strong> is not available —
            using {activeName} instead.
          </span>
          <button
            type="button"
            className="btn btn-ghost shrink-0 px-2 py-0.5 text-xs"
            onClick={() => onChange && onChange('')}
          >
            Use Auto
          </button>
        </div>
      ) : null}
    </div>
  );
}

export default ModelSelect;
