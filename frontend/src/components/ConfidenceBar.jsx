import React from 'react';

/**
 * Turn a category `colorHex` into a translucent surface colour.
 *
 * Lives here (rather than in a lib/ module) because this area only owns the
 * results/guidance components — every one of them needs the same tint maths for
 * chips, rings and swatch backgrounds, and duplicating it would let the shades drift.
 */
export function withAlpha(hex, alpha = 0.12) {
  const raw = typeof hex === 'string' ? hex.trim().replace(/^#/, '') : '';
  const full = raw.length === 3 ? raw.split('').map((c) => c + c).join('') : raw;
  if (!/^[0-9a-fA-F]{6}$/.test(full)) {
    // slate-500: a neutral that stays legible in both themes when data is missing
    return `rgba(100, 116, 139, ${alpha})`;
  }
  const n = Number.parseInt(full, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  if (n <= 0) return 0;
  if (n >= 1) return 1;
  return n;
}

/**
 * An accessible confidence meter.
 *
 * `colorHex` comes from the API (category colour), so the fill must be an inline
 * style: a Tailwind class assembled from data would be purged from the build.
 */
export function ConfidenceBar({ value, colorHex = '#64748b', label = 'Confidence', height = 8 }) {
  const ratio = clamp01(value);
  const percent = ratio * 100;
  const rounded = Math.round(percent);
  const trackHeight = typeof height === 'number' ? `${height}px` : String(height);

  return (
    <div
      role="meter"
      aria-label={label}
      aria-valuenow={rounded}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuetext={`${rounded}%`}
      className="w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700"
      style={{ height: trackHeight }}
    >
      <div
        className="h-full rounded-full motion-safe:transition-[width] motion-safe:duration-500 motion-safe:ease-out motion-reduce:transition-none"
        style={{ width: `${percent}%`, backgroundColor: colorHex }}
      />
    </div>
  );
}

export default ConfidenceBar;
