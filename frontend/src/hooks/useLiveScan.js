/**
 * The Live scan loop: detect -> classify -> track, as fast as the GPU allows.
 *
 * One frame is in flight at a time. The next one is only grabbed on the animation frame
 * after the previous finished, so a slow GPU lowers the frame rate instead of queueing
 * work, and a hidden tab (no animation frames) stops scanning by itself.
 *
 * Still scans (a frozen frame, an uploaded photo) take the slower tiled detector pass and
 * go through the same tracker in snap mode, so an item keeps its id - and its selection -
 * when the picture is frozen. `hold` stops the loop from touching the tracker from the
 * moment a still scan is requested, not from the next render, so a frame that was already
 * in flight can never overwrite the still result.
 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { scanFrame } from '../lib/liveScan.js';
import { createTracker } from '../lib/tracker.js';

const EMPTY_STATS = Object.freeze({ fps: null, detectMs: null, classifyMs: null, crops: 0, still: false });

/** Exponential smoothing for the HUD numbers, so they are readable rather than jittery. */
const HUD_ALPHA = 0.2;

function smooth(previous, next) {
  if (!Number.isFinite(previous)) return next;
  if (!Number.isFinite(next)) return previous;
  return previous + (next - previous) * HUD_ALPHA;
}

function nowMs() {
  return typeof performance !== 'undefined' && typeof performance.now === 'function'
    ? performance.now()
    : Date.now();
}

/**
 * @param {{videoRef: {current: HTMLVideoElement|null}, detector: object|null,
 *          engine: object|null, active: boolean, minScore?: number,
 *          maxDetections?: number, topK?: number}} options
 */
export function useLiveScan({ videoRef, detector, engine, active, minScore, maxDetections, topK = 3 }) {
  const [tracks, setTracks] = useState([]);
  const [stats, setStats] = useState(EMPTY_STATS);
  const [error, setError] = useState(null);

  const trackerRef = useRef(null);
  if (trackerRef.current === null) trackerRef.current = createTracker();
  const inFlightRef = useRef(Promise.resolve());
  const holdRef = useRef(false);

  // The still-scan callbacks read the latest models and settings through refs, so their
  // identity never changes and a click handler never closes over a disposed engine.
  const latest = useRef({ detector, engine, minScore, maxDetections, topK });
  latest.current = { detector, engine, minScore, maxDetections, topK };

  useEffect(() => {
    if (!active || !detector || !engine) return undefined;

    let cancelled = false;
    let rafId = 0;
    let lastFrameAt = null;
    holdRef.current = false;
    setError(null);

    const nextAnimationFrame = () =>
      new Promise((resolve) => {
        rafId = requestAnimationFrame(resolve);
      });

    const run = async () => {
      while (!cancelled) {
        await nextAnimationFrame();
        if (cancelled || holdRef.current) continue;
        const video = videoRef.current;
        // HAVE_CURRENT_DATA: a <video> before its first frame reads as a 0x0 image.
        if (!video || video.readyState < 2 || !video.videoWidth) continue;

        const step = scanFrame({ source: video, detector, engine, minScore, maxDetections, topK });
        inFlightRef.current = step.catch(() => undefined);

        let frame;
        try {
          frame = await step;
        } catch (err) {
          // Switching models disposes the old engine mid-frame; the effect restarts with
          // the new one, so that is not an error worth showing.
          if (cancelled || detector.disposed || engine.disposed) return;
          console.error('[ecosort] live scan frame failed:', err);
          setError(err instanceof Error ? err : new Error(String(err)));
          return;
        }
        if (cancelled || holdRef.current) continue;

        const at = nowMs();
        const interval = lastFrameAt === null ? null : at - lastFrameAt;
        lastFrameAt = at;

        setTracks(trackerRef.current.update(frame.observations));
        setStats((prev) => {
          const base = prev.still ? EMPTY_STATS : prev;
          const frameMs = smooth(base.fps ? 1000 / base.fps : null, interval);
          return {
            fps: Number.isFinite(frameMs) && frameMs > 0 ? 1000 / frameMs : null,
            detectMs: smooth(base.detectMs, frame.detectMs),
            classifyMs: smooth(base.classifyMs, frame.classifyMs),
            crops: frame.observations.length,
            still: false,
          };
        });
      }
    };

    void run();
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [active, detector, engine, minScore, maxDetections, topK, videoRef]);

  /**
   * Scan one still picture thoroughly and show its items.
   *
   * @param {HTMLCanvasElement|HTMLImageElement} source
   * @param {{fresh?: boolean}} [options] `fresh` forgets the live tracks first - an uploaded
   *   photo has nothing to do with what the camera saw.
   */
  const scanStill = useCallback(async (source, { fresh = false } = {}) => {
    const { detector: d, engine: e, minScore: min, maxDetections: max, topK: k } = latest.current;
    if (!d || !e) throw new Error('The detector and the classifier must both be loaded first.');

    holdRef.current = true;
    await inFlightRef.current; // never two detector passes at once
    if (fresh) trackerRef.current.reset();

    setError(null);
    const step = scanFrame({ source, detector: d, engine: e, still: true, minScore: min, maxDetections: max, topK: k });
    inFlightRef.current = step.catch(() => undefined);
    const frame = await step;

    const next = trackerRef.current.update(frame.observations, { snap: true });
    setTracks(next);
    setStats({
      fps: null,
      detectMs: frame.detectMs,
      classifyMs: frame.classifyMs,
      crops: frame.observations.length,
      still: true,
    });
    return next;
  }, []);

  /** Back to the camera with a clean slate. */
  const resume = useCallback(() => {
    holdRef.current = false;
    trackerRef.current.reset();
    setTracks([]);
    setStats(EMPTY_STATS);
    setError(null);
  }, []);

  return { tracks, stats, error, scanStill, resume };
}

export default useLiveScan;
