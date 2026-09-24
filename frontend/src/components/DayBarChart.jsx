import React, { useEffect, useRef, useState } from 'react';

const FULL_DAY = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  timeZone: 'UTC',
});
const SHORT_DAY = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'short',
  timeZone: 'UTC',
});

/**
 * `byDay` keys are calendar dates ("YYYY-MM-DD"), so they are parsed and rendered in
 * UTC. Parsing them locally would shift every label back a day for any viewer west
 * of Greenwich.
 */
function parseDay(day) {
  if (typeof day !== 'string') return null;
  const parsed = new Date(`${day}T00:00:00Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatDay(day, formatter) {
  const parsed = parseDay(day);
  return parsed ? formatter.format(parsed) : String(day ?? '');
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? '' : 's'}`;
}

/** Round an axis maximum up to a value a human would have chosen. */
function niceCeil(value) {
  if (!Number.isFinite(value) || value <= 0) return 1;
  if (value <= 5) return Math.ceil(value);
  const magnitude = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 1.5, 2, 2.5, 5, 7.5, 10]) {
    const candidate = step * magnitude;
    if (value <= candidate) return Math.round(candidate);
  }
  return Math.round(10 * magnitude);
}

/** A rect with only its top corners rounded. */
function barPath(x, y, width, height, radius) {
  const r = Math.max(0, Math.min(radius, width / 2, height));
  return [
    `M ${x} ${y + height}`,
    `V ${y + r}`,
    `A ${r} ${r} 0 0 1 ${x + r} ${y}`,
    `H ${x + width - r}`,
    `A ${r} ${r} 0 0 1 ${x + width} ${y + r}`,
    `V ${y + height}`,
    'Z',
  ].join(' ');
}

/**
 * Measure the container instead of relying on a fixed viewBox: a fixed viewBox scaled
 * down to a 360px phone would shrink the axis labels below legibility.
 */
function useMeasuredWidth(initial) {
  const ref = useRef(null);
  const [width, setWidth] = useState(initial);

  useEffect(() => {
    const element = ref.current;
    if (!element) return undefined;

    const apply = () => {
      const next = element.clientWidth;
      if (next > 0) setWidth(next);
    };
    apply();

    if (typeof ResizeObserver === 'function') {
      const observer = new ResizeObserver(apply);
      observer.observe(element);
      return () => observer.disconnect();
    }
    window.addEventListener('resize', apply);
    return () => window.removeEventListener('resize', apply);
  }, []);

  return { ref, width };
}

/**
 * Hand-rolled daily activity chart — no chart library anywhere in EcoSort.
 */
export function DayBarChart({ byDay, height = 140, colorHex = '#059669' }) {
  const series = (Array.isArray(byDay) ? byDay : [])
    .filter((entry) => entry && typeof entry === 'object')
    .map((entry) => ({
      day: String(entry.day ?? ''),
      count: Number.isFinite(Number(entry.count)) ? Math.max(0, Number(entry.count)) : 0,
    }));

  const { ref, width } = useMeasuredWidth(640);

  const padding = { top: 14, right: 8, bottom: 20, left: 38 };
  const chartHeight = Math.max(90, Number(height) || 140);
  const plotWidth = Math.max(40, width - padding.left - padding.right);
  const plotHeight = Math.max(30, chartHeight - padding.top - padding.bottom);
  const baselineY = padding.top + plotHeight;

  const counts = series.map((entry) => entry.count);
  const maxCount = counts.length ? Math.max(...counts) : 0;
  const axisMax = maxCount > 0 ? niceCeil(maxCount) : 0;
  const totalCount = counts.reduce((sum, value) => sum + value, 0);

  // A single-day window still has a slot, so this never divides by zero.
  const slot = counts.length > 0 ? plotWidth / counts.length : plotWidth;
  const barWidth = Math.max(2, Math.min(48, slot - Math.min(8, slot * 0.28)));

  const peak = series.reduce(
    (best, entry) => (entry.count > best.count ? entry : best),
    { day: '', count: 0 },
  );

  let ariaLabel;
  if (series.length === 0) {
    ariaLabel = 'Daily classification chart with no data.';
  } else if (totalCount === 0) {
    ariaLabel = `Daily classifications from ${formatDay(series[0].day, FULL_DAY)} to ${formatDay(
      series[series.length - 1].day,
      FULL_DAY,
    )}: no activity.`;
  } else {
    ariaLabel = `Daily classifications from ${formatDay(series[0].day, FULL_DAY)} to ${formatDay(
      series[series.length - 1].day,
      FULL_DAY,
    )}. ${plural(totalCount, 'classification')} in total, peaking at ${plural(
      peak.count,
      'classification',
    )} on ${formatDay(peak.day, FULL_DAY)}.`;
  }

  // Only three x labels: a 365-day window would otherwise render an unreadable smear.
  const labelIndices = [];
  if (series.length > 0) {
    labelIndices.push(0);
    if (series.length > 2) labelIndices.push(Math.floor((series.length - 1) / 2));
    if (series.length > 1) labelIndices.push(series.length - 1);
  }

  return (
    <div ref={ref} className="w-full text-slate-500 dark:text-slate-400">
      <svg
        role="img"
        aria-label={ariaLabel}
        width={width}
        height={chartHeight}
        viewBox={`0 0 ${width} ${chartHeight}`}
        className="block w-full overflow-visible"
      >
        {axisMax > 0 ? (
          <>
            <line
              x1={padding.left}
              y1={padding.top}
              x2={width - padding.right}
              y2={padding.top}
              stroke="currentColor"
              strokeWidth="1"
              strokeDasharray="3 4"
              opacity="0.35"
            />
            <text
              x={padding.left - 6}
              y={padding.top + 4}
              textAnchor="end"
              fill="currentColor"
              fontSize="10"
            >
              {axisMax}
            </text>
          </>
        ) : null}

        <text x={padding.left - 6} y={baselineY + 4} textAnchor="end" fill="currentColor" fontSize="10">
          0
        </text>

        <line
          x1={padding.left}
          y1={baselineY}
          x2={width - padding.right}
          y2={baselineY}
          stroke="currentColor"
          strokeWidth="1"
          opacity="0.6"
        />

        {axisMax > 0
          ? series.map((entry, index) => {
              const barHeight = (entry.count / axisMax) * plotHeight;
              if (barHeight <= 0) return null;
              const x = padding.left + slot * index + (slot - barWidth) / 2;
              const y = baselineY - barHeight;
              return (
                <path
                  key={`${entry.day}-${index}`}
                  d={barPath(x, y, barWidth, barHeight, 3)}
                  fill={colorHex}
                >
                  <title>{`${formatDay(entry.day, FULL_DAY)} — ${plural(
                    entry.count,
                    'classification',
                  )}`}</title>
                </path>
              );
            })
          : null}

        {labelIndices.map((index) => {
          const x = padding.left + slot * index + slot / 2;
          const clamped = Math.min(Math.max(x, padding.left + 12), width - padding.right - 12);
          return (
            <text
              key={`label-${index}`}
              x={clamped}
              y={chartHeight - 4}
              textAnchor="middle"
              fill="currentColor"
              fontSize="10"
            >
              {formatDay(series[index].day, SHORT_DAY)}
            </text>
          );
        })}

        {axisMax === 0 ? (
          <text
            x={padding.left + plotWidth / 2}
            y={padding.top + plotHeight / 2}
            textAnchor="middle"
            fill="currentColor"
            fontSize="12"
          >
            No activity in this window
          </text>
        ) : null}
      </svg>
    </div>
  );
}

export default DayBarChart;
