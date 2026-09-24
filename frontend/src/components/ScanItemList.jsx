import { categoryById } from '../lib/categories.js';
import { LOW_CONFIDENCE_THRESHOLD } from '../lib/classifier.js';
import { ConfidenceBar } from './ConfidenceBar.jsx';

/**
 * Where one category goes in the selected region: the colours come from the region's bin
 * (the thing the user acts on), falling back to the category's own colour while the rules
 * load. Shared with the overlay so a box and its row can never disagree.
 */
export function describeBin(rules, categories, categoryId) {
  const guidance = (categoryId && rules?.categories?.[categoryId]) || null;
  const category = categoryById(categories, categoryId);
  return {
    guidance,
    category,
    colorHex: guidance?.colorHex || category?.colorHex || '#64748b',
    textColorHex: guidance?.textColorHex || category?.textColorHex || '#ffffff',
    binLabel: guidance?.colorName ? `${guidance.colorName} bin` : null,
    binName: guidance?.binName || null,
    binId: guidance?.binId || null,
  };
}

function percent(value) {
  return `${Math.round(Math.min(1, Math.max(0, Number(value) || 0)) * 100)}%`;
}

/**
 * The scene as a sorting plan: how many items go to each bin. Grouped by bin, not by
 * category, because two categories often share a bin (single-stream recycling) and the
 * bin is what the user walks to.
 */
function BinSummary({ tracks, rules, categories }) {
  const groups = new Map();
  for (const track of tracks) {
    if (!track.top) continue;
    const bin = describeBin(rules, categories, track.top.category);
    const key = bin.binId || `category:${track.top.category}`;
    const group = groups.get(key) || { bin, count: 0 };
    group.count += 1;
    groups.set(key, group);
  }
  if (groups.size === 0) return null;

  return (
    <ul className="flex flex-wrap gap-2" aria-label="Items per bin">
      {[...groups.entries()].map(([key, { bin, count }]) => (
        <li
          key={key}
          className="chip"
          style={{ backgroundColor: bin.colorHex, color: bin.textColorHex }}
          title={bin.binName || undefined}
        >
          <span className="font-semibold tabular-nums">{count}×</span>
          {bin.binLabel || bin.category?.label || 'Unsorted'}
        </li>
      ))}
    </ul>
  );
}

/**
 * The right-hand column of Live scan: one row per tracked item, in the order the items
 * appeared, with the region's bin for each.
 */
export default function ScanItemList({
  tracks,
  rules,
  categories,
  selectedId,
  onSelect,
  mode,
  debug = false,
}) {
  const heading =
    mode === 'photo' ? 'Items in this photo' : mode === 'frozen' ? 'Items in this frame' : 'Items in view';

  return (
    <section className="card flex flex-col gap-3 p-4" aria-label={heading}>
      <div className="flex items-baseline justify-between gap-2">
        <h2 className="text-base font-semibold text-slate-900 dark:text-slate-50">{heading}</h2>
        <p className="text-sm tabular-nums text-slate-500 dark:text-slate-400" aria-live="polite">
          {tracks.length === 1 ? '1 item' : `${tracks.length} items`}
        </p>
      </div>

      <BinSummary tracks={tracks} rules={rules} categories={categories} />

      {tracks.length === 0 ? (
        <p className="rounded-lg bg-slate-50 p-4 text-sm text-slate-500 dark:bg-slate-800/60 dark:text-slate-400">
          {mode === 'live'
            ? 'Point the camera at a few items spread out on a table. Each one gets its own box, coloured by the bin it belongs in.'
            : 'No items were found. Try the High sensitivity setting, or a photo with the items further apart.'}
        </p>
      ) : (
        <ul className="flex flex-col gap-2">
          {tracks.map((track) => {
            const top = track.top;
            const bin = describeBin(rules, categories, top?.category);
            const selected = track.id === selectedId;
            const unsure = !top || top.confidence < LOW_CONFIDENCE_THRESHOLD;
            return (
              <li key={track.id}>
                <button
                  type="button"
                  onClick={() => onSelect?.(track.id)}
                  aria-pressed={selected}
                  className={`flex w-full items-center gap-3 rounded-lg border p-2.5 text-left transition-colors ${
                    selected
                      ? 'border-brand-500 bg-brand-50 dark:border-brand-400 dark:bg-brand-500/10'
                      : 'border-slate-200 hover:bg-slate-50 dark:border-slate-700 dark:hover:bg-slate-800'
                  } ${track.stale ? 'opacity-60' : ''}`}
                >
                  <span
                    className="grid h-10 w-10 shrink-0 place-items-center rounded-lg text-lg"
                    style={{ backgroundColor: bin.colorHex, color: bin.textColorHex }}
                    aria-hidden="true"
                  >
                    {bin.category?.icon || '•'}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="truncate text-sm font-semibold text-slate-900 dark:text-slate-50">
                        {top ? bin.category?.label || top.category : 'Unrecognised item'}
                      </span>
                      <span className="shrink-0 font-mono text-xs tabular-nums text-slate-500 dark:text-slate-400">
                        {top ? percent(top.confidence) : '—'}
                      </span>
                    </span>
                    <span className="block truncate text-xs text-slate-500 dark:text-slate-400">
                      {bin.binLabel ? `${bin.binLabel} · ${bin.binName}` : 'Bin guidance loading…'}
                      {unsure ? ' · unsure' : ''}
                      {track.stale ? ' · out of view' : ''}
                    </span>
                    <span className="mt-1.5 block">
                      <ConfidenceBar
                        value={top?.confidence ?? 0}
                        colorHex={bin.colorHex}
                        height={5}
                        label={`Confidence for item ${track.id}`}
                      />
                    </span>
                    {debug ? (
                      <span className="mt-1 block truncate font-mono text-[11px] text-slate-400 dark:text-slate-500">
                        #{track.id} · detector “{track.cocoLabel}” {percent(track.detectorScore)} ·{' '}
                        {track.predictions.map((p) => `${p.category} ${percent(p.confidence)}`).join(', ')}
                      </span>
                    ) : null}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
