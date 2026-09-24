import { describe, expect, it } from 'vitest';

import { boxArea, boxFromYxyx, iou, lerpBox, mirrorBox, toPixelRegion } from '../boxes.js';

/** Box maths is floating point; compare edges to ~1e-6. */
function expectBox(actual, expected) {
  for (const key of ['x', 'y', 'width', 'height']) expect(actual[key]).toBeCloseTo(expected[key], 6);
}

describe('boxFromYxyx', () => {
  it('converts the detector order into an x/y/width/height box', () => {
    expectBox(boxFromYxyx(0.1, 0.2, 0.5, 0.6), { x: 0.2, y: 0.1, width: 0.4, height: 0.4 });
  });

  it('clamps the overshoot SSD box decoding produces at the frame edge', () => {
    const box = boxFromYxyx(-0.05, 0.9, 0.5, 1.2);
    expect(box.y).toBe(0);
    expect(box.x).toBe(0.9);
    expect(box.x + box.width).toBeCloseTo(1);
  });

  it('tolerates swapped min/max', () => {
    expect(boxFromYxyx(0.5, 0.6, 0.1, 0.2)).toEqual(boxFromYxyx(0.1, 0.2, 0.5, 0.6));
  });

  it('returns null for a box with no area left, including one entirely off-frame', () => {
    expect(boxFromYxyx(0.3, 0.3, 0.3, 0.6)).toBeNull();
    expect(boxFromYxyx(1.1, 0.2, 1.4, 0.4)).toBeNull();
    expect(boxFromYxyx(NaN, 0, 1, 1)).not.toBeNull(); // NaN reads as 0, not as a crash
  });
});

describe('iou', () => {
  const a = { x: 0, y: 0, width: 0.5, height: 0.5 };

  it('is 1 for identical boxes and 0 for disjoint ones', () => {
    expect(iou(a, { ...a })).toBeCloseTo(1);
    expect(iou(a, { x: 0.6, y: 0.6, width: 0.2, height: 0.2 })).toBe(0);
  });

  it('matches the hand-computed overlap', () => {
    // Half of `a` overlaps a same-size box shifted right by 0.25: inter 0.125, union 0.375.
    expect(iou(a, { x: 0.25, y: 0, width: 0.5, height: 0.5 })).toBeCloseTo(1 / 3);
  });

  it('is 0 for missing or degenerate boxes', () => {
    expect(iou(a, null)).toBe(0);
    expect(iou(a, { x: 0.1, y: 0.1, width: 0, height: 0.2 })).toBe(0);
    expect(boxArea(null)).toBe(0);
  });
});

describe('lerpBox and mirrorBox', () => {
  const from = { x: 0, y: 0, width: 0.2, height: 0.2 };
  const to = { x: 0.4, y: 0.2, width: 0.4, height: 0.6 };

  it('interpolates every edge and lands exactly on the target at t=1', () => {
    expectBox(lerpBox(from, to, 0.5), { x: 0.2, y: 0.1, width: 0.3, height: 0.4 });
    expect(lerpBox(from, to, 1)).toEqual(to);
    expect(lerpBox(from, to, 7)).toEqual(to);
  });

  it('mirrors horizontally around the frame centre', () => {
    expectBox(mirrorBox({ x: 0.1, y: 0.3, width: 0.2, height: 0.1 }), {
      x: 0.7,
      y: 0.3,
      width: 0.2,
      height: 0.1,
    });
  });
});

describe('toPixelRegion', () => {
  it('makes a padded square around the box centre', () => {
    // A 100x200 px box centred at (500, 300) in a 1000x600 frame, 10% padding a side.
    const region = toPixelRegion({ x: 0.45, y: 1 / 3, width: 0.1, height: 1 / 3 }, 1000, 600);
    expect(region.width).toBe(240);
    expect(region.height).toBe(240);
    expect(region.x + region.width / 2).toBe(500);
    expect(region.y + region.height / 2).toBe(300);
  });

  it('shifts rather than shrinks a square that overhangs the frame', () => {
    const region = toPixelRegion({ x: 0.9, y: 0.9, width: 0.1, height: 0.1 }, 1000, 1000);
    expect(region).toEqual({ x: 880, y: 880, width: 120, height: 120 });
  });

  it('cuts a square larger than the frame down to the short edge', () => {
    const region = toPixelRegion({ x: 0, y: 0, width: 1, height: 1 }, 1280, 720);
    expect(region.width).toBe(720);
    expect(region.height).toBe(720);
    expect(region.y).toBe(0);
    expect(region.x).toBeGreaterThanOrEqual(0);
    expect(region.x + region.width).toBeLessThanOrEqual(1280);
  });

  it('can keep the box aspect ratio', () => {
    const region = toPixelRegion({ x: 0.25, y: 0.25, width: 0.5, height: 0.25 }, 400, 400, {
      square: false,
      padding: 0,
    });
    expect(region).toEqual({ x: 100, y: 100, width: 200, height: 100 });
  });

  it('never returns an empty or out-of-frame region', () => {
    for (const box of [
      { x: 0, y: 0, width: 0, height: 0 },
      { x: 0.999, y: 0.999, width: 0.001, height: 0.001 },
      { x: 0.5, y: 0.5, width: 0.9, height: 0.9 },
    ]) {
      const region = toPixelRegion(box, 64, 48);
      expect(region.width).toBeGreaterThanOrEqual(1);
      expect(region.height).toBeGreaterThanOrEqual(1);
      expect(region.x).toBeGreaterThanOrEqual(0);
      expect(region.y).toBeGreaterThanOrEqual(0);
      expect(region.x + region.width).toBeLessThanOrEqual(64);
      expect(region.y + region.height).toBeLessThanOrEqual(48);
    }
  });
});
