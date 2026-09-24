import React, { useId } from 'react';
import CategoryBreakdown from './CategoryBreakdown.jsx';
import DayBarChart from './DayBarChart.jsx';
import EmptyState from './EmptyState.jsx';
import ErrorBanner from './ErrorBanner.jsx';
import {
  formatConfidence,
  formatCount,
  formatDateTime,
  formatPercent,
  formatRelativeTime,
  titleCase,
} from '../lib/format.js';
import { FALLBACK_CATEGORIES } from '../lib/categories.js';

const DAY_WINDOWS = [7, 30, 90, 365];

const SOURCE_LABELS = { webcam: 'Webcam', upload: 'Upload' };
const MODEL_LABELS = { custom: 'Custom model', fallback: 'Fallback (MobileNetV2)' };

function StatTile({ label, value, hint, valueTitle }) {
  return (
    <div className="card">
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500 dark:text-slate-400">
        {label}
      </p>
      <p
        className="mt-1 text-2xl font-semibold tabular-nums text-slate-900 dark:text-slate-50"
        title={valueTitle || undefined}
      >
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{hint}</p>
    </div>
  );
}

function CountTable({ caption, rows, emptyText }) {
  const total = rows.reduce((sum, row) => sum + row.count, 0);
  return (
    <div className="card">
      <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">{caption}</h4>
      {rows.length === 0 ? (
        <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">{emptyText}</p>
      ) : (
        <table className="mt-2 w-full text-sm">
          <thead>
            <tr className="text-left text-[11px] uppercase tracking-wide text-slate-400 dark:text-slate-500">
              <th scope="col" className="font-medium">
                Name
              </th>
              <th scope="col" className="w-16 text-right font-medium">
                Count
              </th>
              <th scope="col" className="w-16 text-right font-medium">
                Share
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key} className="border-t border-slate-100 dark:border-slate-800">
                <td className="py-1.5 text-slate-700 dark:text-slate-200">{row.label}</td>
                <td className="py-1.5 text-right tabular-nums text-slate-700 dark:text-slate-200">
                  {formatCount(row.count)}
                </td>
                <td className="py-1.5 text-right tabular-nums text-slate-500 dark:text-slate-400">
                  {formatPercent(total > 0 ? row.count / total : 0)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function SkeletonBlock({ className }) {
  return <div className={`animate-pulse rounded bg-slate-200 dark:bg-slate-800 ${className}`} aria-hidden="true" />;
}

export function StatsPanel({
  stats,
  loading = false,
  error = null,
  categories,
  days,
  onDaysChange,
  regionId,
  regions,
}) {
  const ids = useId();
  const catalogue = Array.isArray(categories) && categories.length ? categories : FALLBACK_CATEGORIES;
  const regionList = Array.isArray(regions) ? regions : [];
  const regionName = regionList.find((region) => region.id === regionId)?.name || regionId || 'your region';

  const requestedDays = Number(days);
  const reportedDays = Number(stats?.windowDays);
  let windowDays = 30;
  if (requestedDays > 0) windowDays = requestedDays;
  else if (reportedDays > 0) windowDays = reportedDays;

  const sourceRows = (Array.isArray(stats?.bySource) ? stats.bySource : []).map((row) => ({
    key: String(row.source),
    label: SOURCE_LABELS[row.source] || titleCase(String(row.source ?? 'unknown')),
    count: Number(row.count) || 0,
  }));

  const modelRows = (Array.isArray(stats?.byModelKind) ? stats.byModelKind : []).map((row) => ({
    key: String(row.modelKind),
    label: MODEL_LABELS[row.modelKind] || titleCase(String(row.modelKind ?? 'unknown')),
    count: Number(row.count) || 0,
  }));

  const topLabels = (Array.isArray(stats?.topLabels) ? stats.topLabels : []).filter(
    (row) => row && row.label,
  );
  const topLabelMax = topLabels.reduce((max, row) => Math.max(max, Number(row.count) || 0), 0);

  const windowSelector = (
    <div className="flex items-center gap-2">
      <span id={`${ids}-window`} className="text-xs text-slate-500 dark:text-slate-400">
        Window
      </span>
      <div
        role="group"
        aria-labelledby={`${ids}-window`}
        className="inline-flex overflow-hidden rounded-lg border border-slate-200 dark:border-slate-700"
      >
        {DAY_WINDOWS.map((value) => {
          const selected = value === windowDays;
          return (
            <button
              key={value}
              type="button"
              aria-pressed={selected}
              onClick={() => onDaysChange?.(value)}
              className={`px-2.5 py-1 text-xs tabular-nums ${
                selected
                  ? 'bg-emerald-600 text-white'
                  : 'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800'
              }`}
            >
              {value}d
            </button>
          );
        })}
      </div>
    </div>
  );

  const header = (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-50">Statistics</h2>
      {windowSelector}
    </div>
  );

  if (loading && !stats) {
    return (
      <section aria-label="Statistics" className="space-y-4">
        {header}
        <p className="sr-only" role="status">
          Loading statistics
        </p>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
          {[0, 1, 2, 3, 4].map((index) => (
            <div key={index} className="card space-y-2">
              <SkeletonBlock className="h-3 w-2/3" />
              <SkeletonBlock className="h-7 w-1/2" />
              <SkeletonBlock className="h-3 w-full" />
            </div>
          ))}
        </div>
        <div className="card space-y-2">
          <SkeletonBlock className="h-4 w-1/4" />
          <SkeletonBlock className="h-[140px] w-full" />
        </div>
      </section>
    );
  }

  if (error && !stats) {
    return (
      <section aria-label="Statistics" className="space-y-4">
        {header}
        <ErrorBanner error={error} title="Could not load statistics" />
      </section>
    );
  }

  const total = Number(stats?.total) || 0;

  if (!stats || total === 0) {
    return (
      <section aria-label="Statistics" className="space-y-4">
        {header}
        {error ? <ErrorBanner error={error} title="Could not refresh statistics" /> : null}
        <EmptyState
          icon="📊"
          title="Nothing to chart yet"
          message="Statistics appear once you have saved at least one classification. Head to the Classify tab, identify an item and save the result."
        />
      </section>
    );
  }

  return (
    <section aria-label="Statistics" className="space-y-4">
      {header}
      {error ? <ErrorBanner error={error} title="Could not refresh statistics" /> : null}

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        <StatTile
          label="Total"
          value={formatCount(total)}
          hint="Every classification you have saved."
        />
        <StatTile
          label={`Last ${windowDays} days`}
          value={formatCount(Number(stats.totalInWindow) || 0)}
          hint="Saved inside the selected window."
        />
        <StatTile
          label="Avg confidence"
          value={formatConfidence(Number(stats.avgConfidence) || 0)}
          hint="Mean top-1 confidence across saved items."
        />
        <StatTile
          label="Recyclable rate"
          value={formatPercent(Number(stats.recyclableRate) || 0)}
          hint={`Share recyclable under ${regionName} rules — this rate moves with the region.`}
        />
        <StatTile
          label="Last activity"
          value={stats.lastClassifiedAt ? formatRelativeTime(stats.lastClassifiedAt) : 'Never'}
          valueTitle={stats.lastClassifiedAt ? formatDateTime(stats.lastClassifiedAt) : undefined}
          hint="When you last saved a classification."
        />
      </div>

      <div className="card">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
          Daily activity · last {windowDays} days
        </h3>
        <div className="mt-3">
          <DayBarChart byDay={stats.byDay} height={150} colorHex="#059669" />
        </div>
      </div>

      <div className="card">
        <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">By category</h3>
        <CategoryBreakdown
          byCategory={stats.byCategory}
          total={Number(stats.totalInWindow) || total}
          categories={catalogue}
        />
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <CountTable caption="By source" rows={sourceRows} emptyText="No captures recorded yet." />
        <CountTable caption="By model" rows={modelRows} emptyText="No model runs recorded yet." />
      </div>

      <div className="card">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Most common items</h3>
        {topLabels.length === 0 ? (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
            No labels recorded yet.
          </p>
        ) : (
          <ol className="mt-2 space-y-2">
            {topLabels.map((row, index) => {
              const count = Number(row.count) || 0;
              return (
                <li key={`${row.label}-${index}`} className="flex items-center gap-3 text-sm">
                  <span className="w-4 shrink-0 text-right text-xs tabular-nums text-slate-400 dark:text-slate-500">
                    {index + 1}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-baseline justify-between gap-2">
                      <span className="truncate text-slate-700 dark:text-slate-200" title={row.label}>
                        {row.label}
                      </span>
                      <span className="shrink-0 text-xs tabular-nums text-slate-500 dark:text-slate-400">
                        {formatCount(count)}
                      </span>
                    </span>
                    <span className="mt-1 block h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                      <span
                        className="block h-full rounded-full bg-emerald-600 dark:bg-emerald-500"
                        style={{ width: `${topLabelMax > 0 ? (count / topLabelMax) * 100 : 0}%` }}
                      />
                    </span>
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </section>
  );
}

export default StatsPanel;
