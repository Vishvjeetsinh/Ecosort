import React, { useEffect, useId, useRef } from 'react';
import { formatDateTime, formatPercent, titleCase } from '../lib/format.js';

const FOCUSABLE_SELECTOR = [
  'a[href]',
  'button:not([disabled])',
  'input:not([disabled])',
  'select:not([disabled])',
  'textarea:not([disabled])',
  '[tabindex]:not([tabindex="-1"])',
].join(', ');

const THEMES = [
  { value: 'system', label: 'System', hint: 'Follow your operating system.' },
  { value: 'light', label: 'Light', hint: 'Always the light palette.' },
  { value: 'dark', label: 'Dark', hint: 'Always the dark palette.' },
];

function focusableIn(root) {
  if (!root) return [];
  return Array.from(root.querySelectorAll(FOCUSABLE_SELECTOR)).filter(
    (element) => element.getClientRects().length > 0,
  );
}

function bool(value, fallback) {
  return typeof value === 'boolean' ? value : fallback;
}

/**
 * Same 1024-based formatting App.jsx uses for its toast, so the picker, the toast and
 * the README all quote the InceptionResNetV2 download as 107 MB.
 */
function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return null;
  if (n < 1024) return `${Math.round(n)} B`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)} KB`;
  const mb = n / (1024 * 1024);
  return `${mb >= 100 ? Math.round(mb) : mb.toFixed(1)} MB`;
}

/** Above this the first load is a noticeable wait, so the row says so before it starts. */
const LARGE_DOWNLOAD_BYTES = 50 * 1024 * 1024;

/**
 * The one-line spec strip under a model name. Everything comes from the merged
 * descriptor in docs/ARCHITECTURE.md §2.3/§2.4, and anything a hand-written
 * metadata.json omits is simply left out rather than printed as "unknown".
 */
function modelFacts(entry) {
  const metadata = (entry && entry.metadata) || {};
  const facts = [];

  const size = formatBytes(entry?.downloadBytes ?? metadata.downloadBytes);
  if (size) facts.push(size);

  const inputSize = Number(metadata.inputSize);
  if (Number.isFinite(inputSize) && inputSize > 0) facts.push(`${inputSize}×${inputSize}`);

  // classes.length first: it is the array actually used to name outputs, so if a
  // hand-edited metadata.json carries a stale classCount the chip list and this number
  // would otherwise disagree with no error anywhere.
  const classCount = Number(
    Array.isArray(metadata.classes) ? metadata.classes.length : (metadata.classCount ?? NaN),
  );
  if (Number.isFinite(classCount) && classCount > 0) facts.push(`${classCount} classes`);

  const quantization = metadata.quantization;
  if (typeof quantization === 'string' && quantization && quantization !== 'none') {
    facts.push(quantization);
  }

  return facts;
}

function Section({ title, description, children }) {
  return (
    <section className="border-t border-slate-200 px-4 py-4 first:border-t-0 dark:border-slate-800">
      <h3 className="text-sm font-semibold text-slate-900 dark:text-slate-50">{title}</h3>
      {description ? (
        <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{description}</p>
      ) : null}
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Toggle({ id, label, hint, checked, onChange }) {
  return (
    <div className="flex items-start justify-between gap-3 py-2">
      <span className="min-w-0">
        <span id={`${id}-label`} className="block text-sm text-slate-800 dark:text-slate-100">
          {label}
        </span>
        <span className="block text-xs text-slate-500 dark:text-slate-400">{hint}</span>
      </span>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={`${id}-label`}
        onClick={() => onChange(!checked)}
        className={`mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors ${
          checked
            ? 'border-emerald-600 bg-emerald-600'
            : 'border-slate-300 bg-slate-200 dark:border-slate-600 dark:bg-slate-700'
        }`}
      >
        <span
          aria-hidden="true"
          className={`ml-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform ${
            checked ? 'translate-x-5' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  );
}

function Pill({ tone = 'slate', children }) {
  const tones = {
    emerald: 'bg-emerald-100 text-emerald-800 dark:bg-emerald-500/20 dark:text-emerald-200',
    slate: 'bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-300',
    amber: 'bg-amber-100 text-amber-900 dark:bg-amber-500/20 dark:text-amber-200',
  };
  return (
    <span
      className={`rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
        tones[tone] || tones.slate
      }`}
    >
      {children}
    </span>
  );
}

/**
 * One row of the model chooser.
 *
 * Native radios sharing a `name` are used on purpose: they give arrow-key navigation
 * within the group and a single tab stop for free, which a div-based implementation
 * would have to rebuild by hand — and get wrong inside the panel's focus trap.
 */
function ModelOption({
  name,
  value,
  checked,
  disabled = false,
  onSelect,
  title,
  description,
  facts = [],
  recommended = false,
  loaded = false,
  loading = false,
  warning = null,
}) {
  return (
    <label
      className={`flex items-start gap-3 rounded-lg border px-3 py-2 transition-colors ${
        checked
          ? 'border-emerald-500 bg-emerald-50/70 dark:border-emerald-600 dark:bg-emerald-950/30'
          : 'border-slate-200 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-900'
      } ${disabled ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'}`}
    >
      <input
        type="radio"
        name={name}
        value={value}
        checked={checked}
        disabled={disabled}
        onChange={() => onSelect?.(value)}
        className="mt-1 h-4 w-4 shrink-0 accent-emerald-600"
      />
      <span className="min-w-0 flex-1">
        <span className="flex flex-wrap items-center gap-1.5">
          <span className="text-sm font-medium text-slate-800 dark:text-slate-100">{title}</span>
          {recommended ? <Pill tone="emerald">Recommended</Pill> : null}
          {loaded ? <Pill tone="slate">Loaded</Pill> : null}
          {loading ? <Pill tone="amber">Loading</Pill> : null}
        </span>
        {description ? (
          <span className="mt-0.5 block text-xs text-slate-500 dark:text-slate-400">
            {description}
          </span>
        ) : null}
        {facts.length > 0 ? (
          <span className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 font-mono text-[11px] text-slate-500 dark:text-slate-400">
            {facts.map((fact) => (
              <span key={fact}>{fact}</span>
            ))}
          </span>
        ) : null}
        {warning ? (
          <span className="mt-1 block text-[11px] text-amber-700 dark:text-amber-300">{warning}</span>
        ) : null}
      </span>
    </label>
  );
}

function MetaRow({ label, value }) {
  if (value === null || value === undefined || value === '') return null;
  return (
    <div className="flex items-baseline justify-between gap-3 py-0.5">
      <dt className="text-xs text-slate-500 dark:text-slate-400">{label}</dt>
      <dd className="text-right text-xs font-medium text-slate-700 dark:text-slate-200">{value}</dd>
    </div>
  );
}

/**
 * Right-hand slide-over for appearance, classification and model settings.
 *
 * It is a real modal dialog: focus is trapped, Escape and the backdrop close it, the
 * page behind it cannot scroll, and focus goes back to whatever opened it.
 *
 * `modelId` is the stored *selection* ('' = auto), `engineModelId` is what is actually
 * loaded in this tab and `requestedModelId` is what the classifier was last asked for —
 * the three differ while a switch is in flight, and the rows say so.
 */
export function SettingsPanel({
  open,
  onClose,
  settings,
  onChange,
  modelStatus,
  onReloadEngine,
  theme,
  onThemeChange,
  engineKind,
  engineBackend,
  models,
  modelId,
  onModelChange,
  engineModelId,
  engineDownloadBytes,
  requestedModelId,
}) {
  const ids = useId();
  const panelRef = useRef(null);
  const closeRef = useRef(null);

  useEffect(() => {
    if (!open) return undefined;

    const opener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';

    // The panel mounts in this same commit, so the focus call waits a tick for layout.
    const timer = window.setTimeout(() => {
      const target = closeRef.current ?? panelRef.current;
      target?.focus();
    }, 0);

    return () => {
      window.clearTimeout(timer);
      document.body.style.overflow = previousOverflow;
      if (opener && document.contains(opener)) opener.focus();
    };
  }, [open]);

  if (!open) return null;

  const currentSettings = settings ?? {};
  const topK = Math.min(5, Math.max(1, Number(currentSettings.topK) || 3));
  const autoSave = bool(currentSettings.autoSave, true);
  const saveThumbnails = bool(currentSettings.saveThumbnails, true);
  const mirrorWebcam = bool(currentSettings.mirrorWebcam, true);

  const set = (key, value) => onChange?.({ ...currentSettings, [key]: value });

  const handleKeyDown = (event) => {
    if (event.key === 'Escape') {
      event.stopPropagation();
      onClose?.();
      return;
    }
    if (event.key !== 'Tab') return;

    const nodes = focusableIn(panelRef.current);
    if (nodes.length === 0) {
      event.preventDefault();
      panelRef.current?.focus();
      return;
    }

    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    const active = document.activeElement;
    const inside = panelRef.current?.contains(active);

    if (event.shiftKey && (active === first || !inside)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && (active === last || !inside)) {
      event.preventDefault();
      first.focus();
    }
  };

  const registry = Array.isArray(models) ? models : [];
  const selectedModelId = typeof modelId === 'string' ? modelId : '';
  const recommendedEntry = registry.find((entry) => entry.recommended) || null;
  const loadedEntry = engineModelId
    ? registry.find((entry) => entry.id === engineModelId) || null
    : null;
  // A switch in flight has a requested id that is not the loaded one yet; that row is
  // the one the user is waiting on, so it reads "Loading" rather than "Loaded".
  const pendingModelId =
    requestedModelId && requestedModelId !== engineModelId ? requestedModelId : null;

  // modelStatus is the original two-slot answer. It still drives this block so the panel
  // keeps working against a backend that has no registry; the registry, when present,
  // is more precise and wins.
  const custom = modelStatus?.custom ?? null;
  const fallback = modelStatus?.fallback ?? null;
  const active = modelStatus?.active ?? 'none';

  let activeMetadata = null;
  if (active === 'custom') activeMetadata = custom?.metadata ?? null;
  else if (active === 'fallback') activeMetadata = fallback?.metadata ?? null;
  if (loadedEntry?.metadata) activeMetadata = loadedEntry.metadata;

  let activeLabel = 'None — no model installed';
  if (active === 'custom') activeLabel = 'Custom model (your training run)';
  else if (active === 'fallback') activeLabel = 'Fallback — MobileNetV2 / ImageNet';
  if (loadedEntry) activeLabel = loadedEntry.displayName || loadedEntry.id;

  const classes = Array.isArray(activeMetadata?.classes) ? activeMetadata.classes : null;
  const trainedAt = activeMetadata?.trainedAt ?? activeMetadata?.createdAt ?? null;
  const valAccuracy = Number(activeMetadata?.metrics?.valAccuracy);
  const quantization =
    typeof activeMetadata?.quantization === 'string' ? activeMetadata.quantization : 'none';
  const loadedSize = formatBytes(engineDownloadBytes ?? loadedEntry?.downloadBytes);

  return (
    <div className="fixed inset-0 z-50 flex" onKeyDown={handleKeyDown}>
      <div
        role="presentation"
        aria-hidden="true"
        onClick={() => onClose?.()}
        className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm"
      />

      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={`${ids}-title`}
        tabIndex={-1}
        className="relative ml-auto flex h-full w-full max-w-md flex-col overflow-y-auto bg-white shadow-2xl outline-none dark:bg-slate-950"
      >
        <header className="sticky top-0 z-10 flex items-center justify-between gap-3 border-b border-slate-200 bg-white/95 px-4 py-3 backdrop-blur dark:border-slate-800 dark:bg-slate-950/95">
          <h2 id={`${ids}-title`} className="text-base font-semibold text-slate-900 dark:text-slate-50">
            Settings
          </h2>
          <button ref={closeRef} type="button" className="btn btn-ghost px-2 py-1" onClick={() => onClose?.()}>
            Close
          </button>
        </header>

        <Section title="Appearance" description="EcoSort remembers this choice on this device.">
          <fieldset>
            <legend className="sr-only">Theme</legend>
            <div className="space-y-1">
              {THEMES.map((option) => (
                <label
                  key={option.value}
                  className="flex cursor-pointer items-start gap-3 rounded-lg px-2 py-2 hover:bg-slate-50 dark:hover:bg-slate-900"
                >
                  <input
                    type="radio"
                    name={`${ids}-theme`}
                    value={option.value}
                    checked={(theme ?? 'system') === option.value}
                    onChange={() => onThemeChange?.(option.value)}
                    className="mt-0.5 h-4 w-4 accent-emerald-600"
                  />
                  <span>
                    <span className="block text-sm text-slate-800 dark:text-slate-100">{option.label}</span>
                    <span className="block text-xs text-slate-500 dark:text-slate-400">{option.hint}</span>
                  </span>
                </label>
              ))}
            </div>
          </fieldset>
        </Section>

        <Section title="Classification" description="How captures are run and stored.">
          <div className="py-2">
            <label
              htmlFor={`${ids}-topk`}
              className="flex items-baseline justify-between gap-3 text-sm text-slate-800 dark:text-slate-100"
            >
              Predictions to show
              <span className="tabular-nums text-slate-500 dark:text-slate-400">Top {topK}</span>
            </label>
            <input
              id={`${ids}-topk`}
              type="range"
              min="1"
              max="5"
              step="1"
              value={topK}
              onChange={(event) => set('topK', Number(event.target.value))}
              className="mt-2 w-full accent-emerald-600"
            />
            <p className="text-xs text-slate-500 dark:text-slate-400">
              The results panel lists this many candidate categories.
            </p>
          </div>

          <Toggle
            id={`${ids}-autosave`}
            label="Save automatically"
            hint="Store every classification in history without asking."
            checked={autoSave}
            onChange={(value) => set('autoSave', value)}
          />
          <Toggle
            id={`${ids}-thumbs`}
            label="Keep thumbnails"
            hint="Store a small image with each history entry. Turn off to keep the database tiny."
            checked={saveThumbnails}
            onChange={(value) => set('saveThumbnails', value)}
          />
          <Toggle
            id={`${ids}-mirror`}
            label="Mirror webcam"
            hint="Show the webcam preview flipped, like a mirror."
            checked={mirrorWebcam}
            onChange={(value) => set('mirrorWebcam', value)}
          />
        </Section>

        <Section title="Model" description="Inference runs in this browser tab — nothing is uploaded.">
          {registry.length > 0 ? (
            <div role="radiogroup" aria-label="Classifier model" className="space-y-1.5">
              {/*
                A stored id can outlive its directory (the user deleted a converted model, or
                is on another machine). loadEngine silently heals that, but the radio group
                would then show nothing checked at all, which reads as a broken control --
                so say what happened and offer the one-click way out.
              */}
              {selectedModelId !== '' && !registry.some((entry) => entry.id === selectedModelId) ? (
                <p className="rounded-lg border border-amber-300 bg-amber-50 px-2.5 py-2 text-xs text-amber-900 dark:border-amber-700/60 dark:bg-amber-950/40 dark:text-amber-100">
                  The model you picked (<code className="font-mono">{selectedModelId}</code>) is no
                  longer installed, so EcoSort chose one for you.{' '}
                  <button
                    type="button"
                    className="font-semibold underline underline-offset-2"
                    onClick={() => onModelChange?.('')}
                  >
                    Switch to Auto
                  </button>
                  .
                </p>
              ) : null}

              <ModelOption
                name={`${ids}-model`}
                value=""
                checked={selectedModelId === ''}
                onSelect={() => onModelChange?.('')}
                title="Auto (recommended)"
                description={
                  recommendedEntry
                    ? `Whichever model the backend recommends — currently ${
                        recommendedEntry.displayName || recommendedEntry.id
                      }.`
                    : 'Whichever model the backend recommends: your own trained model when one is installed, otherwise the smallest pretrained one.'
                }
              />

              {registry.map((entry) => {
                const bytes = Number(entry.downloadBytes);
                const heavy = Number.isFinite(bytes) && bytes > LARGE_DOWNLOAD_BYTES;
                return (
                  <ModelOption
                    key={entry.id}
                    name={`${ids}-model`}
                    value={entry.id}
                    checked={selectedModelId === entry.id}
                    disabled={entry.available === false}
                    onSelect={() => onModelChange?.(entry.id)}
                    title={entry.displayName || entry.id}
                    description={entry.description || entry.metadata?.description || null}
                    facts={modelFacts(entry)}
                    recommended={Boolean(entry.recommended)}
                    loaded={Boolean(engineModelId) && entry.id === engineModelId}
                    loading={entry.id === pendingModelId}
                    warning={
                      heavy
                        ? `First load downloads ${formatBytes(bytes)} into this browser; after that it is served from the browser cache.`
                        : null
                    }
                  />
                );
              })}
            </div>
          ) : (
            <p className="rounded-lg border border-slate-200 p-3 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
              This backend does not report a model registry, so EcoSort is using its built-in
              order: your custom model when one is installed, otherwise the MobileNetV2
              fallback.
            </p>
          )}

          <dl className="mt-3 space-y-1">
            <MetaRow label="Loaded model" value={activeLabel} />
            <MetaRow
              label="Engine kind"
              value={engineKind ? titleCase(String(engineKind)) : 'Not loaded'}
            />
            <MetaRow label="TensorFlow.js backend" value={engineBackend || 'Unknown'} />
            <MetaRow label="Download size" value={loadedSize} />
          </dl>

          {activeMetadata ? (
            <dl className="mt-3 space-y-1 border-t border-slate-200 pt-3 dark:border-slate-800">
              <MetaRow label="Base model" value={activeMetadata.baseModel} />
              <MetaRow
                label="Input size"
                value={activeMetadata.inputSize ? `${activeMetadata.inputSize} × ${activeMetadata.inputSize}` : null}
              />
              <MetaRow label="Classes" value={classes ? classes.length : (activeMetadata.classCount ?? null)} />
              <MetaRow label="Quantization" value={quantization !== 'none' ? quantization : null} />
              <MetaRow label="Trained" value={trainedAt ? formatDateTime(trainedAt) : null} />
              <MetaRow
                label="Validation accuracy"
                value={Number.isFinite(valAccuracy) && valAccuracy > 0 ? formatPercent(valAccuracy) : null}
              />
            </dl>
          ) : null}

          {classes && classes.length > 0 ? (
            <ul className="mt-2 flex flex-wrap gap-1">
              {classes.map((name) => (
                <li
                  key={name}
                  className="rounded-full bg-slate-100 px-2 py-0.5 text-[11px] text-slate-600 dark:bg-slate-800 dark:text-slate-300"
                >
                  {name}
                </li>
              ))}
            </ul>
          ) : null}

          <button type="button" className="btn btn-primary mt-3 w-full" onClick={() => onReloadEngine?.()}>
            Reload model
          </button>

          <div className="mt-3 rounded-lg bg-slate-50 p-3 text-xs text-slate-600 dark:bg-slate-900 dark:text-slate-300">
            <p className="font-medium text-slate-800 dark:text-slate-100">Add another model</p>
            <p className="mt-1">Run any of these from the project root:</p>
            <ul className="mt-2 space-y-2">
              <li>
                <code className="block overflow-x-auto rounded bg-slate-200 px-1.5 py-1 font-mono dark:bg-slate-800">
                  make fetch-models
                </code>
                <span className="mt-0.5 block">Pretrained MobileNetV2 — the small, quick default.</span>
              </li>
              <li>
                <code className="block overflow-x-auto rounded bg-slate-200 px-1.5 py-1 font-mono dark:bg-slate-800">
                  make convert-model ARCH=InceptionResNetV2
                </code>
                <span className="mt-0.5 block">Larger and more accurate, at a ~107 MB download.</span>
              </li>
              <li>
                <code className="block overflow-x-auto rounded bg-slate-200 px-1.5 py-1 font-mono dark:bg-slate-800">
                  make convert-model ARCH=InceptionV3 QUANTIZE=uint8
                </code>
                <span className="mt-0.5 block">Any Keras architecture, quantised down to 8 bits.</span>
              </li>
              <li>
                <code className="block overflow-x-auto rounded bg-slate-200 px-1.5 py-1 font-mono dark:bg-slate-800">
                  make train
                </code>
                <span className="mt-0.5 block">
                  Your own waste-trained model, exported to <code>models/custom/</code>.
                </span>
              </li>
            </ul>
            <p className="mt-2">
              Converted models appear in the picker above without a restart — the backend
              re-reads <code>models/</code> per request. Press <strong>Reload model</strong> if a
              brand new one has not shown up yet.
            </p>
          </div>
        </Section>

        <Section title="About">
          <p className="text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            EcoSort classifies a photo of an item into one of ten waste categories and shows the bin
            and preparation steps for your region. The model runs entirely in this browser and your
            history stays in a local SQLite file — no image ever leaves this machine, and the app works
            with the network unplugged.
          </p>
          <p className="mt-2 text-xs leading-relaxed text-slate-600 dark:text-slate-300">
            Recycling rules vary enormously between localities, and the bundled rules are a general
            guide only. Always confirm with your local waste authority before putting something in a
            kerbside bin — especially batteries, electronics and anything hazardous.
          </p>
        </Section>
      </div>
    </div>
  );
}

export default SettingsPanel;
