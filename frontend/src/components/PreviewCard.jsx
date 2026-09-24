import Spinner from './Spinner.jsx';

const SOURCE_LABELS = { webcam: 'Webcam', upload: 'Upload' };

function metaChips(meta) {
  if (!meta || typeof meta !== 'object') return [];
  const chips = [];
  if (Number.isFinite(meta.durationMs)) chips.push(`${Math.round(meta.durationMs)} ms`);
  if (meta.modelKind === 'custom') chips.push('Custom model');
  else if (meta.modelKind === 'fallback') chips.push('MobileNetV2 fallback');
  if (Number.isFinite(meta.width) && Number.isFinite(meta.height)) chips.push(`${meta.width}×${meta.height}`);
  if (meta.topLabel) chips.push(String(meta.topLabel));
  if (meta.capturedAt) {
    const at = new Date(meta.capturedAt);
    if (!Number.isNaN(at.getTime())) chips.push(at.toLocaleTimeString());
  }
  return chips;
}

export default function PreviewCard({ dataUrl, source, onClear, onReclassify, busy = false, meta }) {
  if (!dataUrl) return null;

  const sourceLabel = SOURCE_LABELS[source] ?? 'Image';
  const chips = metaChips(meta);

  return (
    <figure className="card flex flex-col gap-3 p-4">
      <figcaption className="flex flex-wrap items-center justify-between gap-2">
        <span className="inline-flex items-center gap-2 rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-700 dark:bg-slate-700 dark:text-slate-200">
          {sourceLabel}
        </span>
        {chips.length > 0 ? (
          <span className="flex flex-wrap items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            {chips.map((chip) => (
              <span key={chip}>{chip}</span>
            ))}
          </span>
        ) : null}
      </figcaption>

      {/* Fixed-aspect box + object-contain so a tall photo is letterboxed, never cropped. */}
      <div className="relative flex aspect-video w-full items-center justify-center overflow-hidden rounded-xl bg-slate-100 dark:bg-slate-900">
        <img src={dataUrl} alt={`Captured item from the ${sourceLabel.toLowerCase()}`} className="max-h-full max-w-full object-contain" />
        {busy ? (
          <div className="absolute inset-0 grid place-items-center bg-slate-900/50">
            <Spinner label="Classifying" />
          </div>
        ) : null}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className="btn btn-primary" onClick={onReclassify} disabled={busy || typeof onReclassify !== 'function'}>
          Classify again
        </button>
        <button type="button" className="btn btn-ghost" onClick={onClear} disabled={busy || typeof onClear !== 'function'}>
          Clear
        </button>
      </div>
    </figure>
  );
}
