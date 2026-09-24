/**
 * Tests for `lib/format.js` (owned by the frontend-core area).
 *
 * The exact wording of the formatters is that module's business, so these
 * assertions pin the properties the rest of the app relies on — the magnitude
 * that ends up on screen, determinism, and ordering — rather than exact
 * strings. `formatRelativeTime` is tested against an injected "now" when the
 * function supports one; support is detected behaviourally, because a default
 * parameter (`now = Date.now()`) hides the second argument from `fn.length`.
 */

import { describe, expect, it } from 'vitest';

import * as format from '../format.js';

const ISO = '2020-01-01T00:00:00.000Z';
const NOW_DATE = new Date('2020-01-01T00:00:05.000Z');
const NOW_MS = NOW_DATE.getTime();

/** First number that appears in a formatted string, e.g. "12.3%" -> 12.3 */
function firstNumber(text) {
  const match = String(text).match(/-?\d+(?:[.,]\d+)?/);
  return match ? Number.parseFloat(match[0].replace(',', '.')) : Number.NaN;
}

function attempt(fn) {
  try {
    const value = fn();
    return typeof value === 'string' ? value : null;
  } catch {
    return null;
  }
}

/**
 * Returns the `now` value that formatRelativeTime actually honours, or null
 * when it always reads the wall clock.
 */
function detectInjectableNow() {
  const ambient = attempt(() => format.formatRelativeTime(ISO));
  for (const candidate of [NOW_DATE, NOW_MS]) {
    const injected = attempt(() => format.formatRelativeTime(ISO, candidate));
    if (injected !== null && injected !== ambient) return candidate;
  }
  return null;
}

describe('format module', () => {
  it('exports the three formatters', () => {
    expect(typeof format.formatPercent).toBe('function');
    expect(typeof format.formatRelativeTime).toBe('function');
    expect(typeof format.formatDuration).toBe('function');
  });
});

describe('formatPercent', () => {
  it('renders a 0..1 fraction as a percentage', () => {
    expect(format.formatPercent(0)).toContain('0');
    expect(format.formatPercent(0.5)).toContain('50');
    expect(format.formatPercent(1)).toContain('100');
  });

  it('keeps the magnitude of an awkward fraction', () => {
    expect(firstNumber(format.formatPercent(0.1234))).toBeGreaterThanOrEqual(12);
    expect(firstNumber(format.formatPercent(0.1234))).toBeLessThan(13);
  });

  it('always returns a non-empty string', () => {
    for (const value of [0, 0.01, 0.333, 0.999, 1]) {
      const text = format.formatPercent(value);
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    }
  });

  it('is monotonic', () => {
    const values = [0, 0.25, 0.5, 0.75, 1].map((v) => firstNumber(format.formatPercent(v)));
    for (let i = 1; i < values.length; i += 1) {
      expect(values[i]).toBeGreaterThan(values[i - 1]);
    }
  });

  it('is deterministic', () => {
    expect(format.formatPercent(0.42)).toBe(format.formatPercent(0.42));
  });
});

describe('formatDuration', () => {
  const samples = [0, 12, 850, 1500, 65_000, 3_600_000];

  it('returns a non-empty string containing a digit for every magnitude', () => {
    for (const ms of samples) {
      const text = format.formatDuration(ms);
      expect(typeof text, `formatDuration(${ms})`).toBe('string');
      expect(text.length, `formatDuration(${ms})`).toBeGreaterThan(0);
      expect(text, `formatDuration(${ms})`).toMatch(/\d/);
    }
  });

  it('distinguishes wildly different durations', () => {
    expect(format.formatDuration(850)).not.toBe(format.formatDuration(65_000));
    expect(format.formatDuration(12)).not.toBe(format.formatDuration(3_600_000));
  });

  it('is deterministic', () => {
    expect(format.formatDuration(1500)).toBe(format.formatDuration(1500));
  });
});

describe('formatRelativeTime', () => {
  const injectableNow = detectInjectableNow();

  it('returns a non-empty string for an ISO timestamp', () => {
    const text = format.formatRelativeTime(ISO);
    expect(typeof text).toBe('string');
    expect(text.length).toBeGreaterThan(0);
  });

  it('is deterministic for the same input', () => {
    expect(format.formatRelativeTime(ISO)).toBe(format.formatRelativeTime(ISO));
  });

  it.skipIf(injectableNow === null)('uses the injected "now" instead of the wall clock', () => {
    const justNow = format.formatRelativeTime(ISO, injectableNow);
    const sameAgain = format.formatRelativeTime(ISO, injectableNow);
    expect(sameAgain).toBe(justNow);
    // The wall clock is years past 2020, so the two must disagree.
    expect(justNow).not.toBe(format.formatRelativeTime(ISO));
  });

  it.skipIf(injectableNow === null)('separates seconds, days and years ago', () => {
    const fixedNow = new Date('2021-06-01T12:00:00.000Z');
    const seconds = format.formatRelativeTime('2021-06-01T11:59:55.000Z', fixedNow);
    const days = format.formatRelativeTime('2021-05-29T12:00:00.000Z', fixedNow);
    const years = format.formatRelativeTime('2019-06-01T12:00:00.000Z', fixedNow);

    expect(new Set([seconds, days, years]).size).toBe(3);
    for (const text of [seconds, days, years]) {
      expect(typeof text).toBe('string');
      expect(text.length).toBeGreaterThan(0);
    }
  });

  it.skipIf(injectableNow !== null)(
    'distinguishes a recent timestamp from an old one (ambient clock only)',
    () => {
      const recent = new Date(Date.now() - 5_000).toISOString();
      const old = new Date(Date.now() - 400 * 24 * 60 * 60 * 1000).toISOString();
      expect(format.formatRelativeTime(recent)).not.toBe(format.formatRelativeTime(old));
    },
  );
});
