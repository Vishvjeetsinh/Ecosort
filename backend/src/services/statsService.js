/**
 * The GET /api/stats payload.
 *
 * Pure assembly: the SQL lives in historyService, the taxonomy in categoryService and the
 * recyclability verdict in rulesService. The day window is computed in UTC in JavaScript rather
 * than in SQL so the series is dense (zero-filled) and independent of SQLite's date functions.
 */
import { createLogger } from '../logger.js';
import { getCategory, listCategories } from './categoryService.js';
import { countClassifications, getLastClassifiedAt, getWindowAggregates } from './historyService.js';
import { defaultRegion, isRecyclable, isValidRegion } from './rulesService.js';

const logger = createLogger('stats');

const DAY_MS = 24 * 60 * 60 * 1000;
const DEFAULT_DAYS = 30;
const MAX_DAYS = 365;
const TOP_LABEL_LIMIT = 10;

function clampDays(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return DEFAULT_DAYS;
  return Math.min(MAX_DAYS, Math.max(1, Math.trunc(numeric)));
}

/** Shares and averages are rounded so float noise (0.30000000000000004) never reaches the UI. */
function round4(value) {
  return Math.round(value * 10000) / 10000;
}

/** The last `windowDays` UTC days, today included, as ascending YYYY-MM-DD strings. */
function buildWindow(windowDays, now) {
  const reference = new Date(now);
  const todayStart = Date.UTC(reference.getUTCFullYear(), reference.getUTCMonth(), reference.getUTCDate());
  const firstStart = todayStart - (windowDays - 1) * DAY_MS;

  const days = [];
  for (let offset = 0; offset < windowDays; offset += 1) {
    days.push(new Date(firstStart + offset * DAY_MS).toISOString().slice(0, 10));
  }

  return {
    days,
    from: new Date(firstStart).toISOString(),
    // Half-open upper bound: everything up to the end of today, nothing dated tomorrow.
    to: new Date(todayStart + DAY_MS).toISOString(),
  };
}

export function getStats({ regionId, days, now = Date.now() } = {}) {
  const windowDays = clampDays(days);

  let region = defaultRegion;
  if (typeof regionId === 'string' && regionId.trim() !== '') {
    if (isValidRegion(regionId)) {
      region = regionId;
    } else {
      // Routes reject unknown regions; reaching here means an internal caller, so degrade rather
      // than fail the whole stats page.
      logger.warn('Stats requested for an unknown region — using the default', { regionId, using: region });
    }
  }

  const window = buildWindow(windowDays, now);
  const aggregates = getWindowAggregates({
    from: window.from,
    to: window.to,
    topLabelLimit: TOP_LABEL_LIMIT,
  });

  const totalInWindow = aggregates.totalInWindow;
  const countsByCategory = new Map(aggregates.byCategory.map((row) => [row.category, row.count]));

  const byCategory = listCategories()
    .map((category) => {
      const count = countsByCategory.get(category.id) ?? 0;
      return {
        category: category.id,
        label: category.label,
        colorHex: category.colorHex,
        count,
        share: totalInWindow === 0 ? 0 : round4(count / totalInWindow),
      };
    })
    .sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));

  let recyclableCount = 0;
  for (const [category, count] of countsByCategory) {
    if (getCategory(category) === null) {
      logger.warn('Ignoring rows with a category outside the taxonomy', { category, count });
      continue;
    }
    if (isRecyclable(region, category)) recyclableCount += count;
  }

  const countsByDay = new Map(aggregates.byDay.map((row) => [row.day, row.count]));

  return {
    total: countClassifications(),
    totalInWindow,
    windowDays,
    avgConfidence: totalInWindow === 0 ? 0 : round4(aggregates.avgConfidence),
    recyclableRate: totalInWindow === 0 ? 0 : round4(recyclableCount / totalInWindow),
    byCategory,
    bySource: aggregates.bySource,
    byModelKind: aggregates.byModelKind,
    byDay: window.days.map((day) => ({ day, count: countsByDay.get(day) ?? 0 })),
    topLabels: aggregates.topLabels,
    lastClassifiedAt: getLastClassifiedAt(),
  };
}
