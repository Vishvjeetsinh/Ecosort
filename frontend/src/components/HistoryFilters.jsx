import React, { useId } from 'react';
import { formatCount } from '../lib/format.js';
import { FALLBACK_CATEGORIES } from '../lib/categories.js';

const SOURCE_OPTIONS = [
  { value: '', label: 'All sources' },
  { value: 'webcam', label: 'Webcam' },
  { value: 'upload', label: 'Upload' },
];

const MODEL_OPTIONS = [
  { value: '', label: 'All models' },
  { value: 'custom', label: 'Custom model' },
  { value: 'fallback', label: 'Fallback (MobileNetV2)' },
];

const DAY_OPTIONS = [
  { value: '', label: 'Any time' },
  { value: '1', label: 'Today' },
  { value: '7', label: 'Last 7 days' },
  { value: '30', label: 'Last 30 days' },
  { value: '90', label: 'Last 90 days' },
  { value: '365', label: 'Last year' },
];

const DEFAULT_LIMIT = 25;

function labelClass() {
  return 'block text-xs font-medium text-slate-500 dark:text-slate-400';
}

export function HistoryFilters({ filters, onChange, categories, regions, total }) {
  const ids = useId();
  const active = filters ?? {};
  const catalogue = Array.isArray(categories) && categories.length ? categories : FALLBACK_CATEGORIES;
  const regionList = Array.isArray(regions) ? regions : [];

  const category = active.category ?? '';
  const source = active.source ?? '';
  const modelKind = active.modelKind ?? '';
  const regionId = active.regionId ?? '';
  const days = active.days === null || active.days === undefined ? '' : String(active.days);

  const limit = Number(active.limit) > 0 ? Number(active.limit) : DEFAULT_LIMIT;
  const offset = Number(active.offset) > 0 ? Number(active.offset) : 0;
  const totalCount = Number.isFinite(Number(total)) ? Math.max(0, Number(total)) : 0;
  const showing = Math.max(0, Math.min(limit, totalCount - offset));

  const hasFilters = Boolean(category || source || modelKind || regionId || days);

  // Any filter change invalidates the current page, so the offset always resets.
  const apply = (patch) => {
    onChange?.({ ...active, ...patch, offset: 0 });
  };

  const clearAll = () => {
    apply({ category: '', source: '', modelKind: '', regionId: '', days: null });
  };

  return (
    <div className="card space-y-3">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <div>
          <label className={labelClass()} htmlFor={`${ids}-category`}>
            Category
          </label>
          <select
            id={`${ids}-category`}
            className="input mt-1 w-full"
            value={category}
            onChange={(event) => apply({ category: event.target.value })}
          >
            <option value="">All categories</option>
            {catalogue.map((item) => (
              // <option> cannot carry a styled colour swatch, so the category icon
              // stands in for the colour dot used elsewhere in the UI.
              <option key={item.id} value={item.id}>
                {`${item.icon ? `${item.icon} ` : ''}${item.label ?? item.id}`}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass()} htmlFor={`${ids}-source`}>
            Source
          </label>
          <select
            id={`${ids}-source`}
            className="input mt-1 w-full"
            value={source}
            onChange={(event) => apply({ source: event.target.value })}
          >
            {SOURCE_OPTIONS.map((option) => (
              <option key={option.value || 'any'} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass()} htmlFor={`${ids}-model`}>
            Model
          </label>
          <select
            id={`${ids}-model`}
            className="input mt-1 w-full"
            value={modelKind}
            onChange={(event) => apply({ modelKind: event.target.value })}
          >
            {MODEL_OPTIONS.map((option) => (
              <option key={option.value || 'any'} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass()} htmlFor={`${ids}-region`}>
            Region
          </label>
          <select
            id={`${ids}-region`}
            className="input mt-1 w-full"
            value={regionId}
            onChange={(event) => apply({ regionId: event.target.value })}
          >
            <option value="">All regions</option>
            {regionList.map((region) => (
              <option key={region.id} value={region.id}>
                {region.name ?? region.id}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className={labelClass()} htmlFor={`${ids}-days`}>
            Time window
          </label>
          <select
            id={`${ids}-days`}
            className="input mt-1 w-full"
            value={days}
            onChange={(event) => {
              const raw = event.target.value;
              apply({ days: raw === '' ? null : Number(raw) });
            }}
          >
            {DAY_OPTIONS.map((option) => (
              <option key={option.value || 'any'} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-slate-500 dark:text-slate-400" aria-live="polite">
          Showing <span className="font-semibold tabular-nums text-slate-700 dark:text-slate-200">{showing}</span>{' '}
          of <span className="font-semibold tabular-nums text-slate-700 dark:text-slate-200">{formatCount(totalCount)}</span>
        </p>
        <button
          type="button"
          className="btn btn-ghost text-xs"
          onClick={clearAll}
          disabled={!hasFilters}
        >
          Clear filters
        </button>
      </div>
    </div>
  );
}

export default HistoryFilters;
