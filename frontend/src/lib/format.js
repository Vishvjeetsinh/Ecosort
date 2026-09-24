/**
 * Pure display formatters. No imports, no DOM, no side effects — every function
 * here is directly unit-testable and safe to call during render.
 */

const MINUTE_MS = 60_000;
const HOUR_MS = 3_600_000;
const DAY_MS = 86_400_000;

function toFiniteNumber(value) {
  // Number(null) and Number('') are both 0, which would silently turn "no value"
  // into a confident 0% — reject them before coercing.
  if (value === null || value === undefined || value === '') return null;
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

/**
 * Format a 0..1 ratio as a percentage.
 * @param {number} value 0..1 (values outside the range are clamped)
 * @param {number} [digits] decimal places; defaults to 1 below 10%, else 0
 */
export function formatPercent(value, digits) {
  const n = toFiniteNumber(value);
  if (n === null) return '—';
  const clamped = Math.min(1, Math.max(0, n));
  const places =
    typeof digits === 'number' ? digits : clamped > 0 && clamped < 0.1 ? 1 : 0;
  return `${(clamped * 100).toFixed(places)}%`;
}

/** Confidence always reads with one decimal so small differences stay visible. */
export function formatConfidence(value) {
  const n = toFiniteNumber(value);
  if (n === null) return '—';
  const clamped = Math.min(1, Math.max(0, n));
  return `${(clamped * 100).toFixed(1)}%`;
}

export function formatDateTime(iso, locale) {
  const date = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleString(locale, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatDate(iso, locale) {
  const date = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';
  return date.toLocaleDateString(locale, { year: 'numeric', month: 'short', day: 'numeric' });
}

/**
 * "just now" / "3 min ago" / "yesterday", falling back to a locale date beyond a week.
 * @param {string|Date} iso
 * @param {Date} [now] injectable for deterministic tests
 */
export function formatRelativeTime(iso, now = new Date(), locale) {
  const date = iso instanceof Date ? iso : new Date(iso);
  if (Number.isNaN(date.getTime())) return '—';

  const diff = now.getTime() - date.getTime();
  if (diff < 0) {
    // Clock skew between the browser and the container: do not print "in -2 min".
    return 'just now';
  }
  if (diff < 45 * 1000) return 'just now';
  if (diff < HOUR_MS) {
    const minutes = Math.max(1, Math.round(diff / MINUTE_MS));
    return `${minutes} min ago`;
  }
  if (diff < DAY_MS) {
    const hours = Math.max(1, Math.round(diff / HOUR_MS));
    return `${hours} ${hours === 1 ? 'hour' : 'hours'} ago`;
  }

  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfThatDay = new Date(
    date.getFullYear(),
    date.getMonth(),
    date.getDate(),
  ).getTime();
  const dayGap = Math.round((startOfToday - startOfThatDay) / DAY_MS);

  if (dayGap <= 1) return 'yesterday';
  if (dayGap < 7) return `${dayGap} days ago`;
  return formatDate(date, locale);
}

/** "820 ms" / "1.4 s" / "2 min 5 s". */
export function formatDuration(ms) {
  const n = toFiniteNumber(ms);
  if (n === null || n < 0) return '—';
  if (n < 1000) return `${Math.round(n)} ms`;
  if (n < 60_000) return `${(n / 1000).toFixed(n < 10_000 ? 2 : 1)} s`;
  const minutes = Math.floor(n / 60_000);
  const seconds = Math.round((n % 60_000) / 1000);
  return `${minutes} min ${seconds} s`;
}

/** Group digits so four- and five-figure history counts stay readable. */
export function formatCount(value, locale) {
  const n = toFiniteNumber(value);
  if (n === null) return '0';
  return Math.round(n).toLocaleString(locale);
}

/** "general waste" -> "General Waste". Leaves existing capitals alone. */
export function titleCase(value) {
  if (typeof value !== 'string' || value.length === 0) return '';
  return value
    .split(/([\s\-_/]+)/)
    .map((part) =>
      /^[\s\-_/]+$/.test(part) ? part : part.charAt(0).toUpperCase() + part.slice(1),
    )
    .join('')
    .replace(/[_]/g, ' ');
}

/** Cheap ellipsis for labels that must not wrap a card. */
export function truncate(value, maxLength = 64) {
  if (typeof value !== 'string') return '';
  if (value.length <= maxLength) return value;
  return `${value.slice(0, Math.max(0, maxLength - 1)).trimEnd()}…`;
}
