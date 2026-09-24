/**
 * One Live scan frame, start to finish (docs/ARCHITECTURE.md 5.1):
 *
 *   fromPixels(frame) --+--> detector.detect  --> boxes (normalised)
 *                       |                            |  toPixelRegion (square, padded)
 *                       +--> engine.classifyRegions <+  one batched predict for all crops
 *
 * The frame is uploaded to the GPU ONCE and both stages read that same tensor, so every
 * crop is cut from exactly the frame its box was found in - reading the <video> twice
 * would pair boxes from one frame with pixels from the next.
 */

import * as tf from '@tensorflow/tfjs';

import { toPixelRegion } from './boxes.js';

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

/**
 * @param {{source: HTMLVideoElement|HTMLCanvasElement|HTMLImageElement,
 *          detector: {detect: Function, detectStill: Function},
 *          engine: {classifyRegions: Function},
 *          still?: boolean, minScore?: number, maxDetections?: number, topK?: number}} input
 *   `still` trades speed for recall with the detector's tiled pass - for a frozen frame or
 *   an uploaded photo, never for a live frame.
 * @returns {Promise<{observations: Array<object>, detectMs: number, classifyMs: number,
 *          frameWidth: number, frameHeight: number}>}
 *   `observations` is what tracker.update() takes: each detection plus its predictions
 *   and the pixel region the classifier actually saw.
 */
export async function scanFrame({
  source,
  detector,
  engine,
  still = false,
  minScore,
  maxDetections,
  topK = 3,
}) {
  const pixels = tf.browser.fromPixels(source);
  try {
    const [frameHeight, frameWidth] = pixels.shape;

    const detectStarted = nowMs();
    const detect = still ? detector.detectStill : detector.detect;
    const detections = await detect(pixels, { minScore, maxDetections });
    const detectMs = nowMs() - detectStarted;

    const regions = detections.map((d) => toPixelRegion(d.box, frameWidth, frameHeight));
    const classifyStarted = nowMs();
    const classified = await engine.classifyRegions(pixels, regions, { topK });
    const classifyMs = regions.length > 0 ? nowMs() - classifyStarted : 0;

    const observations = detections.map((detection, i) => ({
      ...detection,
      region: regions[i],
      predictions: classified.results[i]?.predictions ?? [],
    }));

    return { observations, detectMs, classifyMs, frameWidth, frameHeight };
  } finally {
    pixels.dispose();
  }
}

export default scanFrame;
