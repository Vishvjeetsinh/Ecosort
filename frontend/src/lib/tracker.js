/**
 * Frame-to-frame identity for Live scan.
 *
 * Raw per-frame results flicker: a box jitters by a few pixels, drops out for one frame,
 * and a bottle that is 48% plastic / 45% glass flips its label every other frame. This
 * turns the stream of per-frame detections into stable *tracks*:
 *
 *   - greedy IoU matching gives each physical item one id for as long as it stays in view
 *     (which is also its React key, so its overlay box animates instead of re-mounting);
 *   - boxes are exponentially smoothed, and a track coasts for a few frames after its
 *     detection disappears instead of blinking out;
 *   - category confidences are exponentially smoothed too, so the label only changes when
 *     the evidence does - a label is a running vote, not the last frame's opinion;
 *   - a track is only shown once it has been seen `minHits` times, which filters the
 *     one-frame false positives a low detector threshold lets through.
 *
 * `snap` mode is for still images (a frozen frame, an uploaded photo): boxes land exactly
 * on the detection, unmatched tracks are dropped and new ones are shown at once.
 *
 * Pure: no tfjs, no DOM - unit tested directly.
 */

import { iou, lerpBox } from './boxes.js';

const DEFAULTS = Object.freeze({
  iouThreshold: 0.3,
  maxMisses: 4,
  minHits: 2,
  boxAlpha: 0.55,
  scoreAlpha: 0.35,
});

/** A category whose smoothed confidence decays below this is forgotten. */
const FORGET_BELOW = 0.005;

function ranked(scores, labels) {
  return [...scores.entries()]
    .sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))
    .map(([category, confidence]) => ({
      category,
      label: labels.get(category) ?? category,
      confidence,
    }));
}

function predictionsOf(observation) {
  return Array.isArray(observation?.predictions) ? observation.predictions : [];
}

/**
 * @param {Partial<typeof DEFAULTS>} [options]
 */
export function createTracker(options = {}) {
  const config = { ...DEFAULTS, ...options };
  let tracks = [];
  let nextId = 1;

  function spawn(observation, confirmed) {
    const scores = new Map();
    const labels = new Map();
    for (const p of predictionsOf(observation)) {
      scores.set(p.category, Number(p.confidence) || 0);
      labels.set(p.category, p.label);
    }
    return {
      id: nextId++,
      box: { ...observation.box },
      detectorScore: observation.score,
      cocoId: observation.cocoId,
      cocoLabel: observation.cocoLabel,
      hits: 1,
      misses: 0,
      confirmed,
      scores,
      labels,
    };
  }

  function absorb(track, observation, snap) {
    track.box = snap ? { ...observation.box } : lerpBox(track.box, observation.box, config.boxAlpha);
    track.detectorScore = observation.score;
    track.cocoId = observation.cocoId;
    track.cocoLabel = observation.cocoLabel;
    track.hits += 1;
    track.misses = 0;
    track.confirmed = track.confirmed || snap || track.hits >= config.minHits;

    // A category this frame did not rank counts as 0 for this frame, so a label that has
    // stopped being suggested fades out rather than lingering at its last value.
    const seen = new Map(predictionsOf(observation).map((p) => [p.category, p]));
    const a = config.scoreAlpha;
    for (const category of new Set([...track.scores.keys(), ...seen.keys()])) {
      const observed = Number(seen.get(category)?.confidence) || 0;
      const next = (1 - a) * (track.scores.get(category) ?? 0) + a * observed;
      if (next < FORGET_BELOW && !seen.has(category)) {
        track.scores.delete(category);
        track.labels.delete(category);
      } else {
        track.scores.set(category, next);
      }
      if (seen.has(category)) track.labels.set(category, seen.get(category).label);
    }
  }

  function snapshot() {
    return tracks
      .filter((track) => track.confirmed)
      .map((track) => {
        const predictions = ranked(track.scores, track.labels);
        return Object.freeze({
          id: track.id,
          box: Object.freeze({ ...track.box }),
          predictions: Object.freeze(predictions),
          top: predictions[0] ?? null,
          detectorScore: track.detectorScore,
          cocoId: track.cocoId,
          cocoLabel: track.cocoLabel,
          hits: track.hits,
          // Coasting: kept on screen from memory while its detection is missing.
          stale: track.misses > 0,
        });
      });
  }

  /**
   * Fold one frame's observations in and return the tracks to show, oldest first so the
   * list beside the overlay does not reshuffle while items stay put.
   *
   * @param {Array<{box: object, score?: number, cocoId?: number, cocoLabel?: string,
   *                predictions?: Array<{category: string, label: string, confidence: number}>}>} observations
   * @param {{snap?: boolean}} [options]
   */
  function update(observations, { snap = false } = {}) {
    const list = Array.isArray(observations) ? observations.filter((o) => o && o.box) : [];

    const pairs = [];
    for (let t = 0; t < tracks.length; t += 1) {
      for (let o = 0; o < list.length; o += 1) {
        const overlap = iou(tracks[t].box, list[o].box);
        if (overlap >= config.iouThreshold) pairs.push({ t, o, overlap });
      }
    }
    pairs.sort((a, b) => b.overlap - a.overlap || a.t - b.t || a.o - b.o);

    const matchedTracks = new Set();
    const matchedObservations = new Set();
    for (const { t, o } of pairs) {
      if (matchedTracks.has(t) || matchedObservations.has(o)) continue;
      matchedTracks.add(t);
      matchedObservations.add(o);
      absorb(tracks[t], list[o], snap);
    }

    const survivors = [];
    for (let t = 0; t < tracks.length; t += 1) {
      const track = tracks[t];
      if (!matchedTracks.has(t)) {
        if (snap) continue;
        track.misses += 1;
        if (track.misses > config.maxMisses) continue;
      }
      survivors.push(track);
    }
    for (let o = 0; o < list.length; o += 1) {
      if (!matchedObservations.has(o)) survivors.push(spawn(list[o], snap || config.minHits <= 1));
    }

    tracks = survivors;
    return snapshot();
  }

  function reset() {
    tracks = [];
  }

  return { update, reset, snapshot };
}

export default createTracker;
