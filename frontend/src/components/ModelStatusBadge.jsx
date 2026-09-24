import Spinner from './Spinner.jsx';

const VARIANTS = {
  custom: {
    text: 'Custom model',
    className:
      'border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/60 dark:text-emerald-200',
    dot: 'bg-emerald-500',
    tooltip:
      'Running your own model from models/custom — trained on waste photos by ml/train.py, so it predicts waste categories directly.',
  },
  fallback: {
    text: 'MobileNet fallback',
    className:
      'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/60 dark:text-amber-200',
    dot: 'bg-amber-500',
    tooltip:
      'No custom model found, so EcoSort is using a pretrained ImageNet model and mapping its 1000 object classes onto waste categories. Accurate for common objects, approximate for the rest.',
  },
  none: {
    text: 'No model',
    className:
      'border-red-300 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/60 dark:text-red-200',
    dot: 'bg-red-500',
    tooltip:
      'Neither a custom model nor the MobileNetV2 fallback is available. Run `make fetch-models` to download the pretrained weights.',
  },
};

/**
 * @param {{kind:'custom'|'fallback'|'none', status:'idle'|'loading'|'ready'|'error',
 *          progress?:{stage?:string, message?:string, fraction?:number}|null,
 *          displayName?:string}} props
 */
export function ModelStatusBadge({ kind, status, progress, displayName }) {
  const loading = status === 'loading' || status === 'idle';
  const name = typeof displayName === 'string' ? displayName.trim() : '';

  if (loading) {
    const fraction =
      progress && Number.isFinite(progress.fraction)
        ? Math.round(Math.min(1, Math.max(0, progress.fraction)) * 100)
        : null;
    const detail = progress?.message || 'fetching weights into your browser';
    const tooltip = name
      ? `Loading ${name}: ${detail}. Nothing is sent to a server.`
      : progress?.message
        ? `Loading the classifier: ${progress.message}`
        : 'Loading the classifier into your browser. Nothing is sent to a server.';

    return (
      <span
        className="chip border border-slate-300 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
        title={tooltip}
        aria-label={tooltip}
      >
        <Spinner size={12} label="Loading model" />
        <span>Loading{fraction !== null ? ` ${fraction}%` : '…'}</span>
      </span>
    );
  }

  const variant = VARIANTS[status === 'error' ? 'none' : kind] || VARIANTS.none;
  // The registry can serve any number of models, so the real name beats the slot name
  // ("MobileNet fallback") whenever the caller knows it — the colour still carries the
  // kind. On an error there is nothing loaded, so naming a model would be a lie.
  const text = name && status !== 'error' ? name : variant.text;
  const tooltip = name && text !== variant.text ? `${text} — ${variant.tooltip}` : variant.tooltip;

  return (
    <span
      className={`chip border ${variant.className}`}
      title={tooltip}
      aria-label={`Model status: ${text}. ${variant.tooltip}`}
    >
      <span className={`h-2 w-2 shrink-0 rounded-full ${variant.dot}`} aria-hidden="true" />
      {/* A converted model name can be long; the chip must not stretch the header. */}
      <span className="max-w-[10rem] truncate">{text}</span>
    </span>
  );
}

export default ModelStatusBadge;
