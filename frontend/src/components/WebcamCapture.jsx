import { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import useWebcam from '../hooks/useWebcam.js';
import { canvasFromSource, downscaleToDataUrl } from '../lib/imageUtils.js';
import Spinner from './Spinner.jsx';

/**
 * Fallback square size for the frame handed to the classifier.
 *
 * Models differ: the MobileNets take 224, the Inception family takes 299. Capturing at
 * the active model's own input size means the pixels it sees are real, never upsampled
 * from a smaller grab. 320 is the floor so a capture taken before the engine reports its
 * size still has enough detail for any supported model to downscale from.
 */
const DEFAULT_CAPTURE_SIZE = 320;
/** ~2 fps: responsive enough to feel live, slow enough to leave the main thread to inference. */
const LIVE_INTERVAL_MS = 500;

const STATUS_TEXT = {
  idle: 'Camera off',
  requesting: 'Asking for camera permission…',
  live: 'Camera live',
  denied: 'Permission denied',
  unsupported: 'Camera unavailable',
  error: 'Camera error',
};

const STATUS_DOT = {
  idle: 'bg-slate-400',
  requesting: 'bg-amber-500 animate-pulse',
  live: 'bg-brand-500',
  denied: 'bg-red-500',
  unsupported: 'bg-slate-500',
  error: 'bg-red-500',
};

/**
 * Width of the square crop region, as a percentage of the preview box width.
 *
 * imageUtils centre-crops the *source frame* to a square; the preview box is 16:9 and the
 * video is object-cover, so the square only lines up with the box height when the camera
 * happens to be 16:9 too. Work it out from the real frame size instead of assuming.
 */
function squareGuideWidthPct(frameWidth, frameHeight) {
  if (!frameWidth || !frameHeight) return 56.25; // 9/16 — a square as tall as a 16:9 box
  const boxW = 16;
  const boxH = 9;
  const coverScale = Math.max(boxW / frameWidth, boxH / frameHeight);
  const side = Math.min(frameWidth, frameHeight) * coverScale;
  return Math.min(100, (side / boxW) * 100);
}

export default function WebcamCapture({
  active = true,
  onCapture,
  onError,
  mirrored = false,
  liveMode = false,
  onLiveFrame,
  busy = false,
  captureSize = DEFAULT_CAPTURE_SIZE,
}) {
  const { videoRef, status, error, devices, deviceId, setDeviceId, retry, capabilities } = useWebcam({
    enabled: Boolean(active),
    facingMode: 'environment',
  });

  const [mirror, setMirror] = useState(Boolean(mirrored));
  const [live, setLive] = useState(Boolean(liveMode));
  const [frame, setFrame] = useState({ width: 0, height: 0 });
  const [capturing, setCapturing] = useState(false);
  const selectId = useId();

  const liveAvailable = typeof onLiveFrame === 'function';
  const isLive = status === 'live';
  const canCapture = isLive && !busy && !capturing;

  // Keep the callbacks in refs so the rAF loop below never restarts just because the
  // parent re-rendered with fresh function identities.
  const onLiveFrameRef = useRef(onLiveFrame);
  const onErrorRef = useRef(onError);
  useEffect(() => {
    onLiveFrameRef.current = onLiveFrame;
    onErrorRef.current = onError;
  }, [onLiveFrame, onError]);

  useEffect(() => {
    setMirror(Boolean(mirrored));
  }, [mirrored]);

  useEffect(() => {
    setLive(Boolean(liveMode));
  }, [liveMode]);

  const readFrameSize = useCallback(() => {
    const video = videoRef.current;
    if (!video) return;
    setFrame((prev) =>
      prev.width === video.videoWidth && prev.height === video.videoHeight
        ? prev
        : { width: video.videoWidth, height: video.videoHeight }
    );
  }, [videoRef]);

  // Chrome fires `resize` on the video element when the track renegotiates resolution.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    video.addEventListener('resize', readFrameSize);
    return () => video.removeEventListener('resize', readFrameSize);
  }, [videoRef, readFrameSize]);

  const guideWidthPct = useMemo(() => squareGuideWidthPct(frame.width, frame.height), [frame.width, frame.height]);
  const sideMaskPct = Math.max(0, (100 - guideWidthPct) / 2);

  const handleCapture = useCallback(async () => {
    const video = videoRef.current;
    if (!video || !isLive || busy || capturing) return;
    if (!video.videoWidth || !video.videoHeight) {
      onErrorRef.current?.(
        Object.assign(new Error('The camera has not delivered a frame yet — wait a moment and try again.'), {
          name: 'WebcamFrameError',
        })
      );
      return;
    }
    setCapturing(true);
    try {
      // The mirrored preview is a comfort feature for front cameras; the captured pixels
      // are deliberately left un-mirrored so the model sees the scene as it is.
      const canvas = await canvasFromSource(video, { size: captureSize });
      const dataUrl = await downscaleToDataUrl(video);
      onCapture?.({ canvas, dataUrl });
    } catch (err) {
      console.error('[WebcamCapture] capture failed:', err);
      onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setCapturing(false);
    }
  }, [videoRef, isLive, busy, capturing, onCapture]);

  // Live mode: grab a square every LIVE_INTERVAL_MS. The loop is not even started
  // while `busy`, so frames can never queue up behind a running inference.
  useEffect(() => {
    if (!live || !liveAvailable || !isLive || busy) return undefined;
    let rafId = 0;
    let cancelled = false;
    let lastAt = 0;
    let inFlight = false;

    const tick = (timestamp) => {
      if (cancelled) return;
      rafId = requestAnimationFrame(tick);
      if (inFlight || timestamp - lastAt < LIVE_INTERVAL_MS) return;
      const video = videoRef.current;
      if (!video || !video.videoWidth || !video.videoHeight) return;
      lastAt = timestamp;
      inFlight = true;
      Promise.resolve()
        .then(() => canvasFromSource(video, { size: captureSize }))
        .then((canvas) => {
          if (cancelled) return;
          onLiveFrameRef.current?.({ canvas });
        })
        .catch((err) => {
          console.error('[WebcamCapture] live frame failed:', err);
          if (!cancelled) onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)));
        })
        .finally(() => {
          inFlight = false;
        });
    };

    rafId = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [live, liveAvailable, isLive, busy, videoRef]);

  const handleKeyDown = useCallback(
    (event) => {
      if (event.key !== ' ' && event.key !== 'Spacebar') return;
      // Let space do its normal job when a real control has focus.
      const tag = event.target?.tagName;
      if (tag === 'BUTTON' || tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      event.preventDefault();
      void handleCapture();
    },
    [handleCapture]
  );

  const statusText = STATUS_TEXT[status] ?? 'Camera';
  const resolution = frame.width && frame.height ? `${frame.width}×${frame.height}` : null;

  return (
    <div
      className="card flex flex-col gap-3 p-4"
      role="group"
      aria-label="Webcam capture"
      tabIndex={0}
      onKeyDown={handleKeyDown}
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="flex items-center gap-2 text-sm font-medium text-slate-700 dark:text-slate-200" role="status" aria-live="polite">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${STATUS_DOT[status] ?? 'bg-slate-400'}`} aria-hidden="true" />
          {statusText}
          {resolution && isLive ? <span className="text-xs font-normal text-slate-500 dark:text-slate-400">({resolution})</span> : null}
        </p>

        <div className="flex flex-wrap items-center gap-2">
          {devices.length > 1 ? (
            <>
              <label htmlFor={selectId} className="sr-only">
                Camera
              </label>
              <select
                id={selectId}
                className="input w-auto py-1 text-sm"
                value={deviceId ?? ''}
                onChange={(event) => setDeviceId(event.target.value || null)}
                disabled={busy}
              >
                <option value="">Default camera</option>
                {devices.map((device) => (
                  <option key={device.deviceId || device.label} value={device.deviceId}>
                    {device.label}
                  </option>
                ))}
              </select>
            </>
          ) : null}

          <button
            type="button"
            className="btn btn-ghost text-sm"
            aria-pressed={mirror}
            onClick={() => setMirror((value) => !value)}
          >
            {mirror ? 'Mirrored' : 'Mirror'}
          </button>

          {liveAvailable ? (
            <button
              type="button"
              className="btn btn-ghost text-sm"
              aria-pressed={live}
              onClick={() => setLive((value) => !value)}
              disabled={!isLive}
            >
              {live ? 'Live: on' : 'Live: off'}
            </button>
          ) : null}
        </div>
      </div>

      <div className="relative aspect-video w-full overflow-hidden rounded-xl bg-slate-900">
        <video
          ref={videoRef}
          className="h-full w-full object-cover"
          style={mirror ? { transform: 'scaleX(-1)' } : undefined}
          autoPlay
          playsInline
          muted
          aria-label="Live camera preview"
          onLoadedMetadata={readFrameSize}
        />

        {isLive ? (
          <div className="pointer-events-none absolute inset-0" aria-hidden="true">
            <div className="absolute inset-y-0 left-0 bg-slate-950/55" style={{ width: `${sideMaskPct}%` }} />
            <div className="absolute inset-y-0 right-0 bg-slate-950/55" style={{ width: `${sideMaskPct}%` }} />
            <div
              className="absolute inset-y-0 left-1/2 -translate-x-1/2 rounded-lg border-2 border-white/80 shadow-[0_0_0_1px_rgba(0,0,0,0.35)]"
              style={{ width: `${guideWidthPct}%` }}
            />
          </div>
        ) : (
          <div className="absolute inset-0 grid place-items-center bg-slate-900/90 p-6 text-center">
            {status === 'requesting' ? (
              <Spinner label="Waiting for camera permission" />
            ) : (
              <div className="max-w-md space-y-2">
                <p className="text-sm font-semibold text-slate-100">{statusText}</p>
                <p className="text-sm text-slate-300">
                  {error?.message ?? 'The camera is not running. Press Start camera to try again.'}
                </p>
                {error?.hint ? <p className="text-xs text-slate-400">{error.hint}</p> : null}
                <button type="button" className="btn btn-primary mt-1" onClick={retry}>
                  {status === 'idle' ? 'Start camera' : 'Retry'}
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs text-slate-500 dark:text-slate-400">
          Only the bright square is sent to the model, captured at {captureSize}×{captureSize}. Press
          <kbd className="mx-1 rounded border border-slate-300 px-1 font-sans text-[0.7rem] dark:border-slate-600">Space</kbd>
          to capture when this panel has focus.
        </p>
        <button
          type="button"
          className="btn btn-primary min-w-[9rem] justify-center"
          onClick={handleCapture}
          disabled={!canCapture}
        >
          {capturing ? 'Capturing…' : busy ? 'Classifying…' : 'Capture'}
        </button>
      </div>

      {capabilities?.label && isLive ? (
        <p className="text-xs text-slate-400 dark:text-slate-500">Using: {capabilities.label}</p>
      ) : null}
    </div>
  );
}
