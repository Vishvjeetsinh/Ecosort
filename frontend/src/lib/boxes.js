/**
 * Box geometry shared by the detector, the tracker and the crop step of Live scan.
 *
 * Every box is `{x, y, width, height}` NORMALISED to the frame (0..1, origin top-left), so
 * the same object serves a 1280x720 webcam frame, a 300x300 detector input and a
 * percentage-positioned overlay without conversion. Pixels only appear at the very end,
 * in toPixelRegion(), where a crop has to be cut out of a real tensor.
 *
 * Pure: no tfjs, no DOM - unit tested directly.
 */

function finite(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function clamp(value, lo, hi) {
  return Math.min(hi, Math.max(lo, value));
}

/**
 * The detector's native `[ymin, xmin, ymax, xmax]` (any order of min/max, possibly a
 * little outside 0..1 - SSD box decoding overshoots the frame edge) -> a clamped box.
 *
 * @returns {{x: number, y: number, width: number, height: number} | null} null when the
 *   box has no area left once clamped.
 */
export function boxFromYxyx(ymin, xmin, ymax, xmax) {
  const top = clamp(Math.min(finite(ymin), finite(ymax)), 0, 1);
  const bottom = clamp(Math.max(finite(ymin), finite(ymax)), 0, 1);
  const left = clamp(Math.min(finite(xmin), finite(xmax)), 0, 1);
  const right = clamp(Math.max(finite(xmin), finite(xmax)), 0, 1);
  const width = right - left;
  const height = bottom - top;
  if (width <= 0 || height <= 0) return null;
  return { x: left, y: top, width, height };
}

export function boxArea(box) {
  if (!box) return 0;
  return Math.max(0, finite(box.width)) * Math.max(0, finite(box.height));
}

/** Intersection over union; 0 for disjoint or degenerate boxes. */
export function iou(a, b) {
  if (!a || !b) return 0;
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  const inter = Math.max(0, right - left) * Math.max(0, bottom - top);
  if (inter <= 0) return 0;
  const union = boxArea(a) + boxArea(b) - inter;
  return union > 0 ? inter / union : 0;
}

/** Linear blend from `a` towards `b`; t=1 lands exactly on `b`. */
export function lerpBox(a, b, t) {
  const k = clamp(finite(t, 1), 0, 1);
  return {
    x: a.x + (b.x - a.x) * k,
    y: a.y + (b.y - a.y) * k,
    width: a.width + (b.width - a.width) * k,
    height: a.height + (b.height - a.height) * k,
  };
}

/** The same box seen in a horizontally flipped picture (the mirrored webcam preview). */
export function mirrorBox(box) {
  return { ...box, x: 1 - box.x - box.width };
}

/**
 * The pixel rectangle the classifier gets to see for one detection.
 *
 * Square by default, for the same reason capture is (imageUtils.canvasFromSource): every
 * classifier here was trained on square, undistorted pictures, so stretching a tall bottle
 * box to 224x224 would show it an object it never saw. The square is grown by `padding` on
 * each side because SSD boxes hug the object, while the training photos have a margin -
 * then shifted, never shrunk, to stay inside the frame, and only shrunk when the frame
 * itself is smaller than the square.
 *
 * @param {{x:number,y:number,width:number,height:number}} box normalised
 * @param {number} frameWidth  pixels
 * @param {number} frameHeight pixels
 * @param {{square?: boolean, padding?: number}} [options]
 * @returns {{x:number, y:number, width:number, height:number}} integer pixels, at least
 *   1x1 and fully inside the frame
 */
export function toPixelRegion(box, frameWidth, frameHeight, { square = true, padding = 0.1 } = {}) {
  const fw = Math.max(1, Math.floor(finite(frameWidth, 1)));
  const fh = Math.max(1, Math.floor(finite(frameHeight, 1)));
  const pad = Math.max(0, finite(padding));

  const cx = (finite(box.x) + finite(box.width) / 2) * fw;
  const cy = (finite(box.y) + finite(box.height) / 2) * fh;
  let w = Math.max(1, finite(box.width) * fw * (1 + 2 * pad));
  let h = Math.max(1, finite(box.height) * fh * (1 + 2 * pad));
  if (square) {
    const side = Math.max(w, h);
    w = side;
    h = side;
  }
  w = Math.min(w, fw);
  h = Math.min(h, fh);
  if (square) {
    // A square that no longer fits either way is cut to the short edge, still square.
    const side = Math.min(w, h);
    w = side;
    h = side;
  }

  const width = Math.max(1, Math.round(w));
  const height = Math.max(1, Math.round(h));
  const x = clamp(Math.round(cx - width / 2), 0, fw - width);
  const y = clamp(Math.round(cy - height / 2), 0, fh - height);
  return { x, y, width, height };
}
