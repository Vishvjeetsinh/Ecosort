import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

import useLiveScan from '../hooks/useLiveScan.js';
import useLocalStorage from '../hooks/useLocalStorage.js';
import useWebcam from '../hooks/useWebcam.js';
import { mirrorBox, toPixelRegion } from '../lib/boxes.js';
import { LOW_CONFIDENCE_THRESHOLD } from '../lib/classifier.js';
import { cropToDataUrl, fileToImageElement, snapshotCanvas } from '../lib/imageUtils.js';
import { BinGuideCard } from './BinGuideCard.jsx';
import ErrorBanner from './ErrorBanner.jsx';
import ScanItemList, { describeBin } from './ScanItemList.jsx';
import Spinner from './Spinner.jsx';

/**
 * Detector score thresholds. Medium is lib/detector.js's measured default; High finds the
 * items that only reach ~0.1-0.2 on unusual backgrounds, at the price of more false boxes
 * (which the classifier then mostly labels as general waste).
 */
const SENSITIVITY = Object.freeze({
  low: { label: 'Low', minScore: 0.35 },
  medium: { label: 'Medium', minScore: 0.2 },
  high: { label: 'High', minScore: 0.12 },
});

/** Enough for a desk full of items; each extra box is one more crop in the batch. */
const MAX_ITEMS = 6;
/** Still frames are scanned at up to this size: detail for the tiles, not a 12 MP photo. */
const STILL_MAX_SIDE = 1280;

function percent(value) {
  return `${Math.round(Math.min(1, Math.max(0, Number(value) || 0)) * 100)}%`;
}

function formatMs(value) {
  return Number.isFinite(value) ? `${Math.round(value)} ms` : '—';
}

/**
 * One tracked item drawn over the picture. Positioned in percentages of a stage that has
 * the frame's own aspect ratio, so no letterbox maths is needed, and keyed by track id so
 * the CSS transition glides the box between frames instead of re-mounting it.
 */
function DetectionBox({ track, bin, mirrored, selected, debug, onSelect }) {
  const box = mirrored ? mirrorBox(track.box) : track.box;
  const top = track.top;
  // Near the top edge the label goes inside the box, or it would be clipped.
  const labelInside = box.y < 0.09;
  const name = top ? bin.category?.shortLabel || bin.category?.label || top.category : 'Unknown';

  return (
    <button
      type="button"
      onClick={() => onSelect(track.id)}
      aria-label={`${top ? bin.category?.label || top.category : 'Unknown item'}, ${percent(top?.confidence)}${
        bin.binLabel ? `, ${bin.binLabel}` : ''
      }`}
      aria-pressed={selected}
      className={`pointer-events-auto absolute rounded-md border-[3px] motion-safe:transition-[left,top,width,height,opacity] motion-safe:duration-150 motion-safe:ease-linear ${
        track.stale ? 'opacity-45' : 'opacity-100'
      } ${selected ? 'z-10' : ''}`}
      style={{
        left: `${box.x * 100}%`,
        top: `${box.y * 100}%`,
        width: `${box.width * 100}%`,
        height: `${box.height * 100}%`,
        borderColor: bin.colorHex,
        boxShadow: selected
          ? `0 0 0 2px #ffffff, 0 0 0 5px ${bin.colorHex}, 0 0 24px 4px ${bin.colorHex}`
          : '0 0 0 1px rgba(0,0,0,0.35)',
      }}
    >
      <span
        className={`absolute left-[-3px] flex max-w-[16rem] items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5 text-left text-xs font-semibold shadow ${
          labelInside ? 'top-0' : 'bottom-full mb-1'
        }`}
        style={{ backgroundColor: bin.colorHex, color: bin.textColorHex }}
      >
        <span aria-hidden="true">{bin.category?.icon}</span>
        <span className="shrink-0">{name}</span>
        <span className="shrink-0 font-mono font-normal tabular-nums opacity-90">{percent(top?.confidence)}</span>
        {/* Bin names can be long ("Orange (facility signage) bin"); they give way first. */}
        {bin.binLabel ? <span className="min-w-0 truncate font-normal opacity-90">→ {bin.binLabel}</span> : null}
      </span>
      {debug ? (
        // Below the box, not inside it: on a small item it would hide the item and its label.
        <span className="absolute left-[-3px] top-full mt-1 whitespace-nowrap rounded bg-slate-950/80 px-1 font-mono text-[10px] text-slate-100">
          #{track.id} {track.cocoLabel} {percent(track.detectorScore)}
        </span>
      ) : null}
      {top && top.confidence < LOW_CONFIDENCE_THRESHOLD ? (
        <span className="absolute right-0 top-0 m-1 rounded bg-amber-400 px-1 text-[10px] font-bold text-amber-950">?</span>
      ) : null}
    </button>
  );
}

function PreparingCard({ title, progress }) {
  const fraction =
    progress && Number.isFinite(progress.fraction) ? Math.round(progress.fraction * 100) : null;
  return (
    <div className="card flex items-center gap-3 p-3 text-sm text-slate-700 dark:text-slate-200">
      <Spinner size={18} label={title} />
      <div className="min-w-0 flex-1">
        <p className="font-medium">{title}</p>
        <p className="truncate text-xs text-slate-500 dark:text-slate-400">
          {progress?.message || 'Loading into your browser. Nothing is uploaded.'}
        </p>
      </div>
      {fraction !== null ? (
        <span className="shrink-0 font-mono text-xs tabular-nums text-slate-500 dark:text-slate-400">{fraction}%</span>
      ) : null}
    </div>
  );
}

function DetectorUnavailable({ error, onRetry }) {
  // DetectorUnavailableError means the files are missing or unreachable; anything else
  // means they loaded but the browser could not run them, where fetching again won't help.
  const missing = error?.name === 'DetectorUnavailableError';
  return (
    <div className="card border-amber-300 bg-amber-50 p-4 dark:border-amber-800 dark:bg-amber-950/40">
      <h2 className="text-base font-semibold text-amber-900 dark:text-amber-100">
        {missing ? 'The object detector is not available' : 'The object detector failed to start'}
      </h2>
      {missing ? (
        <>
          <p className="mt-1 text-sm text-amber-900/90 dark:text-amber-100/90">
            Live scan finds items with a small pretrained detector that lives next to the classifiers in{' '}
            <code className="font-mono">models/detectors/</code>. Nothing is downloaded at runtime, so fetch it once:
          </p>
          <pre className="mt-2 overflow-x-auto rounded-lg bg-amber-100 p-2 font-mono text-xs text-amber-950 dark:bg-amber-900/50 dark:text-amber-50">
{`make fetch-models        # or: node scripts/fetch-detector.mjs`}
          </pre>
        </>
      ) : (
        <p className="mt-1 text-sm text-amber-900/90 dark:text-amber-100/90">
          The model files loaded, but this browser could not run them - usually a WebGL problem such as a
          lost GPU context or hardware acceleration being switched off. Try again, reload the page, or use a
          current Chrome or Edge with hardware acceleration enabled.
        </p>
      )}
      {error?.message ? (
        <p className="mt-2 font-mono text-[11px] text-amber-900/80 dark:text-amber-100/70">{error.message}</p>
      ) : null}
      <button type="button" className="btn btn-ghost mt-3" onClick={onRetry}>
        Check again
      </button>
    </div>
  );
}

/**
 * The Live scan tab: every item in view gets its own box, coloured by the bin it belongs
 * in for the selected region. Detection, classification and tracking all run in this tab.
 *
 * Modes: `live` scans the camera continuously; `frozen` holds one camera frame and scans it
 * thoroughly (tiled detection) so it can be saved; `photo` does the same for an upload.
 */
export default function LiveScanPanel({
  detector,
  engine,
  engineReady,
  engineProgress,
  engineBlocked,
  rules,
  categories,
  mirror: mirrorSetting = true,
  topK = 3,
  onSaveItems,
  saving = false,
}) {
  const [mode, setMode] = useState('live');
  const [still, setStill] = useState(null); // { canvas, dataUrl, width, height, source }
  const [scanningStill, setScanningStill] = useState(false);
  const [stillError, setStillError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [mirror, setMirror] = useState(Boolean(mirrorSetting));
  const [sensitivity, setSensitivity] = useLocalStorage('scanSensitivity', 'medium');
  const [debug, setDebug] = useLocalStorage('scanDebug', false);
  const fileInputRef = useRef(null);
  const sensitivityId = useId();

  useEffect(() => setMirror(Boolean(mirrorSetting)), [mirrorSetting]);

  const minScore = (SENSITIVITY[sensitivity] || SENSITIVITY.medium).minScore;

  // Compile the classifier for multi-item batches before the first frame, once per engine.
  const [warmEngine, setWarmEngine] = useState(null);
  useEffect(() => {
    if (!engine || !engineReady || typeof engine.warmRegionBatches !== 'function') return undefined;
    let cancelled = false;
    const done = () => {
      if (!cancelled) setWarmEngine(engine);
    };
    // A failed warm-up is not fatal: scanning still works, it just stalls on first use.
    engine.warmRegionBatches().then(done, (err) => {
      console.warn('[ecosort] region batch warm-up failed:', err);
      done();
    });
    return () => {
      cancelled = true;
    };
  }, [engine, engineReady]);
  const engineWarm = Boolean(engine) && warmEngine === engine;

  const modelsReady = Boolean(detector.ready && detector.detector && engineReady && engine && engineWarm);

  // The camera stays on while a frame is frozen, so Resume is instant; a photo does not need it.
  const webcam = useWebcam({ enabled: mode !== 'photo', facingMode: 'environment' });
  const { videoRef } = webcam;
  const cameraLive = webcam.status === 'live';

  const [frame, setFrame] = useState({ width: 0, height: 0 });
  const readFrameSize = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    setFrame((prev) =>
      prev.width === video.videoWidth && prev.height === video.videoHeight
        ? prev
        : { width: video.videoWidth, height: video.videoHeight },
    );
  }, [videoRef]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    video.addEventListener('resize', readFrameSize);
    return () => video.removeEventListener('resize', readFrameSize);
  }, [videoRef, readFrameSize]);

  const { tracks, stats, error: scanError, scanStill, resume } = useLiveScan({
    videoRef,
    detector: detector.detector,
    engine,
    active: mode === 'live' && cameraLive && modelsReady,
    minScore,
    maxDetections: MAX_ITEMS,
    topK,
  });

  // Keep a selection only while its item exists; otherwise follow the first item.
  const selected = tracks.find((track) => track.id === selectedId) || tracks[0] || null;

  const runStillScan = useCallback(
    async (snapshot, { fresh }) => {
      setStill(snapshot);
      setStillError(null);
      setScanningStill(true);
      try {
        await scanStill(snapshot.canvas, { fresh });
      } catch (err) {
        console.error('[ecosort] still scan failed:', err);
        setStillError(err instanceof Error ? err : new Error(String(err)));
      } finally {
        setScanningStill(false);
      }
    },
    [scanStill],
  );

  const handleFreeze = useCallback(() => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !modelsReady) return;
    const canvas = snapshotCanvas(video, { maxSide: STILL_MAX_SIDE });
    setMode('frozen');
    void runStillScan(
      { canvas, dataUrl: canvas.toDataURL('image/jpeg', 0.9), width: canvas.width, height: canvas.height, source: 'webcam' },
      { fresh: false },
    );
  }, [modelsReady, runStillScan, videoRef]);

  const handleResume = useCallback(() => {
    resume();
    setStill(null);
    setStillError(null);
    setMode('live');
  }, [resume]);

  const handlePhoto = useCallback(
    async (event) => {
      const file = event.target.files?.[0];
      event.target.value = ''; // the same file can be chosen again
      if (!file || !modelsReady) return;
      try {
        const image = await fileToImageElement(file);
        const canvas = snapshotCanvas(image, { maxSide: STILL_MAX_SIDE });
        setMode('photo');
        await runStillScan(
          { canvas, dataUrl: canvas.toDataURL('image/jpeg', 0.9), width: canvas.width, height: canvas.height, source: 'upload' },
          { fresh: true },
        );
      } catch (err) {
        setStillError(err instanceof Error ? err : new Error(String(err)));
      }
    },
    [modelsReady, runStillScan],
  );

  const handleSave = useCallback(() => {
    if (!still || tracks.length === 0 || !onSaveItems) return;
    const items = tracks
      .filter((track) => track.top)
      .map((track) => {
        let imageDataUrl = null;
        try {
          imageDataUrl = cropToDataUrl(still.canvas, toPixelRegion(track.box, still.width, still.height));
        } catch (err) {
          console.warn('[ecosort] could not crop a thumbnail for item', track.id, err);
        }
        return { predictions: track.predictions, imageDataUrl };
      });
    void onSaveItems(items, {
      source: still.source,
      durationMs: Math.round((stats.detectMs || 0) + (stats.classifyMs || 0)),
    });
  }, [onSaveItems, stats.classifyMs, stats.detectMs, still, tracks]);

  const handleKeyDown = useCallback(
    (event) => {
      if (event.key !== ' ' && event.key !== 'Spacebar') return;
      const tag = event.target?.tagName;
      if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      event.preventDefault();
      if (mode === 'live') handleFreeze();
      else handleResume();
    },
    [handleFreeze, handleResume, mode],
  );

  const showingStill = mode !== 'live' && still;
  const aspect = showingStill
    ? `${still.width} / ${still.height}`
    : frame.width && frame.height
      ? `${frame.width} / ${frame.height}`
      : '16 / 9';
  // A webcam still is shown mirrored like the preview it came from, so nothing jumps on freeze.
  const mirrored = mirror && (mode === 'live' || still?.source === 'webcam');

  const bins = useMemo(
    () => new Map(tracks.map((track) => [track.id, describeBin(rules, categories, track.top?.category)])),
    [categories, rules, tracks],
  );
  const selectedBin = selected ? bins.get(selected.id) : null;

  const statusText = (() => {
    if (detector.status === 'error') return 'Detector unavailable';
    if (engineBlocked) return 'No classifier';
    if (!modelsReady) return 'Loading models…';
    if (mode === 'photo') return 'Photo';
    if (mode === 'frozen') return 'Frozen frame';
    if (cameraLive) return 'Scanning';
    return webcam.status === 'requesting' ? 'Asking for camera permission…' : 'Camera off';
  })();

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      <div className="flex flex-col gap-4 lg:col-span-3">
        {detector.status === 'error' ? (
          <DetectorUnavailable error={detector.error} onRetry={detector.reload} />
        ) : !detector.ready ? (
          <PreparingCard title="Preparing the object detector…" progress={detector.progress} />
        ) : null}
        {engineBlocked ? (
          <ErrorBanner
            title="No classifier is loaded"
            error={new Error('Live scan names each item with the classifier. Open the Classify tab to see how to install one.')}
          />
        ) : !engineReady ? (
          <PreparingCard title="Preparing the classifier…" progress={engineProgress} />
        ) : !engineWarm ? (
          <PreparingCard
            title="Preparing the classifier for several items at once…"
            progress={{ message: 'Compiling GPU shaders for batches of 1, 2, 4 and 6 crops. This happens once.' }}
          />
        ) : null}

        <div
          className="card flex flex-col gap-3 p-4"
          role="group"
          aria-label="Live scan camera"
          tabIndex={0}
          onKeyDown={handleKeyDown}
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200" role="status">
              <span
                className={`inline-block h-2.5 w-2.5 rounded-full ${
                  mode === 'live' && cameraLive && modelsReady ? 'animate-pulse bg-brand-500' : 'bg-slate-400'
                }`}
                aria-hidden="true"
              />
              {statusText}
              {mode === 'live' && Number.isFinite(stats.fps) ? (
                <span className="font-mono text-xs font-normal tabular-nums text-slate-500 dark:text-slate-400">
                  {stats.fps.toFixed(1)} fps
                </span>
              ) : null}
            </p>

            <div className="flex flex-wrap items-center gap-2">
              <label htmlFor={sensitivityId} className="text-xs text-slate-500 dark:text-slate-400">
                Sensitivity
              </label>
              <select
                id={sensitivityId}
                className="input w-auto py-1 text-sm"
                value={SENSITIVITY[sensitivity] ? sensitivity : 'medium'}
                onChange={(event) => setSensitivity(event.target.value)}
              >
                {Object.entries(SENSITIVITY).map(([id, option]) => (
                  <option key={id} value={id}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button type="button" className="btn btn-ghost text-sm" aria-pressed={mirror} onClick={() => setMirror((v) => !v)}>
                {mirror ? 'Mirrored' : 'Mirror'}
              </button>
              <button type="button" className="btn btn-ghost text-sm" aria-pressed={Boolean(debug)} onClick={() => setDebug((v) => !v)}>
                {debug ? 'Debug: on' : 'Debug'}
              </button>
            </div>
          </div>

          <div className="relative w-full overflow-hidden rounded-xl bg-slate-900" style={{ aspectRatio: aspect }}>
            <video
              ref={videoRef}
              className={`absolute inset-0 h-full w-full object-cover ${showingStill ? 'invisible' : ''}`}
              style={mirror ? { transform: 'scaleX(-1)' } : undefined}
              autoPlay
              playsInline
              muted
              aria-label="Live camera preview"
              onLoadedMetadata={readFrameSize}
            />
            {showingStill ? (
              <img
                src={still.dataUrl}
                alt={mode === 'photo' ? 'The uploaded photo' : 'The frozen camera frame'}
                className="absolute inset-0 h-full w-full object-cover"
                style={mirrored ? { transform: 'scaleX(-1)' } : undefined}
              />
            ) : null}

            <div className="pointer-events-none absolute inset-0">
              {tracks.map((track) => (
                <DetectionBox
                  key={track.id}
                  track={track}
                  bin={bins.get(track.id)}
                  mirrored={mirrored}
                  selected={selected?.id === track.id}
                  debug={Boolean(debug)}
                  onSelect={setSelectedId}
                />
              ))}
            </div>

            {scanningStill ? (
              <div className="absolute inset-0 grid place-items-center bg-slate-950/40">
                <div className="flex items-center gap-2 rounded-lg bg-slate-950/80 px-3 py-2 text-sm text-slate-100">
                  <Spinner size={16} label="Scanning the still frame" />
                  Thorough scan…
                </div>
              </div>
            ) : null}

            {mode !== 'photo' && !cameraLive && !showingStill ? (
              <div className="absolute inset-0 grid place-items-center bg-slate-900/90 p-6 text-center">
                {webcam.status === 'requesting' ? (
                  <Spinner label="Waiting for camera permission" />
                ) : (
                  <div className="max-w-md space-y-2">
                    <p className="text-sm text-slate-300">
                      {webcam.error?.message ?? 'The camera is not running.'}
                    </p>
                    {webcam.error?.hint ? <p className="text-xs text-slate-400">{webcam.error.hint}</p> : null}
                    <div className="flex flex-wrap justify-center gap-2 pt-1">
                      <button type="button" className="btn btn-primary" onClick={webcam.retry}>
                        {webcam.status === 'idle' ? 'Start camera' : 'Retry'}
                      </button>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() => fileInputRef.current?.click()}
                        disabled={!modelsReady}
                      >
                        Scan a photo instead
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : null}

            {debug && (Number.isFinite(stats.detectMs) || Number.isFinite(stats.classifyMs)) ? (
              <p className="pointer-events-none absolute bottom-2 right-2 rounded bg-slate-950/75 px-2 py-1 font-mono text-[11px] text-slate-100">
                {stats.still ? 'full + 4 tiles' : 'detect'} {formatMs(stats.detectMs)} · classify {stats.crops}{' '}
                {formatMs(stats.classifyMs)}
                {engine?.backend ? ` · ${engine.backend}` : ''}
              </p>
            ) : null}
          </div>

          {scanError || stillError ? (
            <ErrorBanner
              title="Scanning stopped"
              error={scanError || stillError}
              onRetry={handleResume}
            />
          ) : null}

          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {mode === 'live' ? (
                <>
                  Boxes are coloured by the bin each item goes in. Press
                  <kbd className="mx-1 rounded border border-slate-300 px-1 font-sans text-[0.7rem] dark:border-slate-600">Space</kbd>
                  to freeze a frame and save its items.
                </>
              ) : (
                'Still pictures get a slower, tiled scan that also finds small items.'
              )}
            </p>
            <div className="flex flex-wrap gap-2">
              <input ref={fileInputRef} type="file" accept="image/*" className="hidden" onChange={handlePhoto} />
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => fileInputRef.current?.click()}
                disabled={!modelsReady || scanningStill}
              >
                Scan a photo
              </button>
              {mode === 'live' ? (
                <button
                  type="button"
                  className="btn btn-primary min-w-[8rem]"
                  onClick={handleFreeze}
                  disabled={!modelsReady || !cameraLive}
                >
                  Freeze frame
                </button>
              ) : (
                <>
                  <button
                    type="button"
                    className="btn btn-primary"
                    onClick={handleSave}
                    disabled={saving || scanningStill || tracks.length === 0 || !onSaveItems}
                  >
                    {saving ? 'Saving…' : `Save ${tracks.length === 1 ? '1 item' : `${tracks.length} items`}`}
                  </button>
                  <button type="button" className="btn btn-ghost" onClick={handleResume} disabled={scanningStill}>
                    Resume camera
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="flex flex-col gap-4 lg:col-span-2">
        <ScanItemList
          tracks={tracks}
          rules={rules}
          categories={categories}
          selectedId={selected?.id ?? null}
          onSelect={setSelectedId}
          mode={mode}
          debug={Boolean(debug)}
        />
        {selected?.top ? (
          <BinGuideCard
            guidance={selectedBin?.guidance || null}
            category={selectedBin?.category || null}
            bin={
              selectedBin?.guidance && Array.isArray(rules?.bins)
                ? rules.bins.find((candidate) => candidate.id === selectedBin.guidance.binId) || null
                : null
            }
            region={rules ? rules.region : null}
          />
        ) : null}
      </div>
    </div>
  );
}
