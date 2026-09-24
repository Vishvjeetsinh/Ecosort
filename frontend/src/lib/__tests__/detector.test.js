import { describe, expect, it, vi } from 'vitest';

// Only the pure selection policy is under test; the model itself is exercised in the
// browser. Stubbing tfjs keeps the suite independent of WebGL in jsdom.
vi.mock('@tensorflow/tfjs', () => ({
  Tensor: class Tensor {},
  tidy: (fn) => fn(),
  ready: async () => undefined,
  getBackend: () => 'cpu',
  loadGraphModel: async () => {
    throw new Error('tfjs is stubbed in this suite');
  },
}));

const {
  COCO_LABELS,
  DEFAULT_MIN_SCORE,
  IGNORED_COCO_IDS,
  STILL_TILES,
  buildClassMask,
  cocoLabel,
  mergeTiledDetections,
  selectDetections,
} = await import('../detector.js');

/** Build the flat arrays selectDetections consumes from readable rows. */
function raw(rows) {
  return {
    scores: Float32Array.from(rows.map((r) => r.score)),
    classes: Int32Array.from(rows.map((r) => r.column ?? 43)),
    boxes: Float32Array.from(rows.flatMap((r) => r.yxyx)),
  };
}

describe('the COCO class table', () => {
  it('names the 80 categories in use and falls back for the gaps', () => {
    expect(Object.keys(COCO_LABELS)).toHaveLength(80);
    expect(cocoLabel(44)).toBe('bottle');
    expect(cocoLabel(12)).toBe('object');
  });

  it('masks out people and furniture, and nothing an item could be', () => {
    const mask = buildClassMask();
    expect(mask).toHaveLength(90);
    expect(mask[0]).toBe(0); // column 0 = id 1 = person
    expect(mask[66]).toBe(0); // id 67 = dining table
    expect(mask[43]).toBe(1); // id 44 = bottle
    expect(mask[76]).toBe(1); // id 77 = cell phone
    expect([...mask].filter((v) => v === 0)).toHaveLength(IGNORED_COCO_IDS.size);
  });
});

describe('selectDetections', () => {
  it('keeps boxes at or above the threshold, strongest first, with COCO ids', () => {
    const detections = selectDetections(
      raw([
        { score: 0.3, column: 43, yxyx: [0.1, 0.1, 0.4, 0.3] },
        { score: 0.1, column: 46, yxyx: [0.5, 0.5, 0.9, 0.9] },
        { score: 0.8, column: 76, yxyx: [0.5, 0.6, 0.9, 0.95] },
      ]),
    );
    expect(detections.map((d) => d.cocoLabel)).toEqual(['cell phone', 'bottle']);
    expect(detections[0].cocoId).toBe(77);
    // The boxes went through a Float32Array, as they do coming off the GPU.
    const { box } = detections[1];
    expect(box.x).toBeCloseTo(0.1, 6);
    expect(box.y).toBeCloseTo(0.1, 6);
    expect(box.width).toBeCloseTo(0.2, 6);
    expect(box.height).toBeCloseTo(0.3, 6);
  });

  it('defaults to the recall-friendly 0.2 threshold rather than COCO-SSD 0.5', () => {
    expect(DEFAULT_MIN_SCORE).toBe(0.2);
    expect(selectDetections(raw([{ score: 0.25, yxyx: [0.1, 0.1, 0.5, 0.5] }]))).toHaveLength(1);
  });

  it('suppresses overlapping boxes regardless of their COCO class', () => {
    const detections = selectDetections(
      raw([
        { score: 0.6, column: 43, yxyx: [0.1, 0.1, 0.5, 0.5] },
        { score: 0.5, column: 85, yxyx: [0.12, 0.11, 0.52, 0.5] },
        { score: 0.4, column: 43, yxyx: [0.6, 0.6, 0.9, 0.9] },
      ]),
    );
    expect(detections.map((d) => d.score)).toEqual([expect.closeTo(0.6), expect.closeTo(0.4)]);
  });

  it('drops specks below the minimum area and boxes entirely off-frame', () => {
    const detections = selectDetections(
      raw([
        { score: 0.9, yxyx: [0.5, 0.5, 0.52, 0.52] },
        { score: 0.9, yxyx: [1.1, 0.2, 1.3, 0.4] },
      ]),
    );
    expect(detections).toEqual([]);
  });

  it('caps the number of boxes', () => {
    const rows = Array.from({ length: 10 }, (_, i) => ({
      score: 0.9 - i * 0.01,
      yxyx: [0, i * 0.1, 0.1, i * 0.1 + 0.09],
    }));
    expect(selectDetections({ ...raw(rows), maxDetections: 3, minArea: 0 })).toHaveLength(3);
  });

  it('survives mismatched array lengths and NaN scores', () => {
    const input = raw([
      { score: Number.NaN, yxyx: [0.1, 0.1, 0.5, 0.5] },
      { score: 0.7, yxyx: [0.1, 0.1, 0.5, 0.5] },
    ]);
    expect(selectDetections({ ...input, boxes: input.boxes.subarray(0, 4) })).toEqual([]);
    expect(selectDetections(input)).toHaveLength(1);
  });
});

describe('mergeTiledDetections', () => {
  const detection = (box, score) => ({ box, score, cocoId: 44, cocoLabel: 'bottle' });
  const [topLeft, topRight] = STILL_TILES;

  it('maps a tile detection into frame coordinates', () => {
    const merged = mergeTiledDetections(
      [],
      [{ tile: topRight, detections: [detection({ x: 0.2, y: 0.2, width: 0.4, height: 0.4 }, 0.5)] }],
    );
    expect(merged).toHaveLength(1);
    expect(merged[0].box.x).toBeCloseTo(0.375 + 0.2 * 0.625);
    expect(merged[0].box.width).toBeCloseTo(0.4 * 0.625);
  });

  it('adds what only a tile found to what the full frame found', () => {
    const full = [detection({ x: 0.05, y: 0.5, width: 0.2, height: 0.2 }, 0.4)];
    const merged = mergeTiledDetections(full, [
      { tile: topRight, detections: [detection({ x: 0.3, y: 0.1, width: 0.3, height: 0.3 }, 0.35)] },
    ]);
    expect(merged).toHaveLength(2);
  });

  it('drops a tile box cut off by an inner tile edge, but not by a frame edge', () => {
    const merged = mergeTiledDetections(
      [],
      [
        // Touches the top-left tile's right edge, which is inside the frame: cut in half.
        { tile: topLeft, detections: [detection({ x: 0.7, y: 0.2, width: 0.3, height: 0.2 }, 0.6)] },
        // Touches the top-left tile's left edge, which is the frame's own edge: a real item.
        { tile: topLeft, detections: [detection({ x: 0, y: 0.5, width: 0.2, height: 0.2 }, 0.5)] },
      ],
    );
    expect(merged).toHaveLength(1);
    expect(merged[0].score).toBe(0.5);
  });

  it('suppresses the tile view of an item the full frame already boxed', () => {
    const full = [detection({ x: 0.5, y: 0.1, width: 0.4, height: 0.4 }, 0.6)];
    // The same item seen from the top-right tile, a little tighter: IoU alone would keep it.
    const tileView = {
      x: (0.55 - 0.375) / 0.625,
      y: 0.15 / 0.625,
      width: 0.25 / 0.625,
      height: 0.3 / 0.625,
    };
    const merged = mergeTiledDetections(full, [{ tile: topRight, detections: [detection(tileView, 0.5)] }]);
    expect(merged).toHaveLength(1);
    expect(merged[0].score).toBe(0.6);
  });

  it('keeps the strongest boxes up to the cap', () => {
    const full = Array.from({ length: 8 }, (_, i) =>
      detection({ x: (i % 4) * 0.25, y: i < 4 ? 0 : 0.5, width: 0.2, height: 0.2 }, 0.9 - i * 0.05),
    );
    const merged = mergeTiledDetections(full, [], { maxDetections: 5 });
    expect(merged.map((d) => d.score)).toEqual(full.slice(0, 5).map((d) => d.score));
  });
});
