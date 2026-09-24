import { describe, expect, it } from 'vitest';

import { createTracker } from '../tracker.js';

const BOTTLE = { x: 0.1, y: 0.1, width: 0.2, height: 0.4 };
const CAN = { x: 0.6, y: 0.5, width: 0.15, height: 0.2 };

function observe(box, predictions, extra = {}) {
  return { box, score: 0.6, cocoId: 44, cocoLabel: 'bottle', predictions, ...extra };
}

const plastic = (confidence) => ({ category: 'plastic', label: 'Plastic', confidence });
const glass = (confidence) => ({ category: 'glass', label: 'Glass', confidence });
const metal = (confidence) => ({ category: 'metal', label: 'Metal', confidence });

describe('createTracker', () => {
  it('hides a new item until it has been seen twice', () => {
    const tracker = createTracker();
    expect(tracker.update([observe(BOTTLE, [plastic(0.8)])])).toEqual([]);
    const shown = tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    expect(shown).toHaveLength(1);
    expect(shown[0].top.category).toBe('plastic');
  });

  it('keeps one id for an item that moves a little between frames', () => {
    const tracker = createTracker();
    tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    const [first] = tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    const moved = { ...BOTTLE, x: BOTTLE.x + 0.03, y: BOTTLE.y + 0.02 };
    const [second] = tracker.update([observe(moved, [plastic(0.8)])]);
    expect(second.id).toBe(first.id);
    // Smoothed: part of the way to the new position, not all of it.
    expect(second.box.x).toBeGreaterThan(BOTTLE.x);
    expect(second.box.x).toBeLessThan(moved.x);
  });

  it('matches each detection to its own item when several are in view', () => {
    const tracker = createTracker();
    tracker.update([observe(BOTTLE, [plastic(0.8)]), observe(CAN, [metal(0.9)])]);
    const tracks = tracker.update([observe(CAN, [metal(0.9)]), observe(BOTTLE, [plastic(0.8)])]);
    expect(tracks.map((t) => t.top.category)).toEqual(['plastic', 'metal']);
    expect(new Set(tracks.map((t) => t.id)).size).toBe(2);
  });

  it('does not flip the label on one contrary frame', () => {
    const tracker = createTracker();
    for (let i = 0; i < 4; i += 1) tracker.update([observe(BOTTLE, [plastic(0.55), glass(0.4)])]);
    const [afterBlip] = tracker.update([observe(BOTTLE, [glass(0.6), plastic(0.35)])]);
    expect(afterBlip.top.category).toBe('plastic');

    // ...but does follow evidence that persists.
    let latest;
    for (let i = 0; i < 4; i += 1) latest = tracker.update([observe(BOTTLE, [glass(0.6), plastic(0.35)])]);
    expect(latest[0].top.category).toBe('glass');
  });

  it('lets a category that stops being suggested fade out', () => {
    const tracker = createTracker({ scoreAlpha: 0.5 });
    tracker.update([observe(BOTTLE, [plastic(0.8), glass(0.1)])]);
    let tracks;
    for (let i = 0; i < 12; i += 1) tracks = tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    expect(tracks[0].predictions.map((p) => p.category)).toEqual(['plastic']);
  });

  it('coasts through a short dropout, marked stale, then forgets the item', () => {
    const tracker = createTracker({ maxMisses: 2 });
    tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    const [seen] = tracker.update([observe(BOTTLE, [plastic(0.8)])]);

    const [coasting] = tracker.update([]);
    expect(coasting.id).toBe(seen.id);
    expect(coasting.stale).toBe(true);
    expect(tracker.update([])).toHaveLength(1);
    expect(tracker.update([])).toEqual([]);
  });

  it('recovers the same id when a coasting item is detected again', () => {
    const tracker = createTracker();
    tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    const [seen] = tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    tracker.update([]);
    const [back] = tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    expect(back.id).toBe(seen.id);
    expect(back.stale).toBe(false);
  });

  it('snap mode shows new items at once, exactly where they were found', () => {
    const tracker = createTracker();
    const tracks = tracker.update([observe(BOTTLE, [plastic(0.8)])], { snap: true });
    expect(tracks).toHaveLength(1);
    expect(tracks[0].box).toEqual(BOTTLE);
  });

  it('snap mode lands an existing track exactly and drops the ones not found', () => {
    const tracker = createTracker();
    tracker.update([observe(BOTTLE, [plastic(0.8)]), observe(CAN, [metal(0.9)])]);
    tracker.update([observe(BOTTLE, [plastic(0.8)]), observe(CAN, [metal(0.9)])]);
    const moved = { ...BOTTLE, x: BOTTLE.x + 0.05 };
    const tracks = tracker.update([observe(moved, [plastic(0.8)])], { snap: true });
    expect(tracks).toHaveLength(1);
    expect(tracks[0].box).toEqual(moved);
  });

  it('reset forgets everything', () => {
    const tracker = createTracker({ minHits: 1 });
    tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    tracker.reset();
    expect(tracker.snapshot()).toEqual([]);
  });

  it('returns frozen snapshots React can hold on to', () => {
    const tracker = createTracker({ minHits: 1 });
    const [track] = tracker.update([observe(BOTTLE, [plastic(0.8)])]);
    expect(Object.isFrozen(track)).toBe(true);
    expect(Object.isFrozen(track.box)).toBe(true);
    tracker.update([observe({ ...BOTTLE, x: 0.3 }, [plastic(0.8)])]);
    expect(track.box.x).toBe(BOTTLE.x);
  });

  it('ignores junk observations instead of throwing', () => {
    const tracker = createTracker({ minHits: 1 });
    expect(tracker.update(null)).toEqual([]);
    expect(tracker.update([null, { score: 1 }])).toEqual([]);
    const [track] = tracker.update([{ box: BOTTLE }]);
    expect(track.top).toBeNull();
  });
});
