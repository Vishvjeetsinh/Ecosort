import { useCallback, useEffect, useRef, useState } from 'react';
import WebcamCapture from './WebcamCapture.jsx';
import ImageDropzone from './ImageDropzone.jsx';
import PreviewCard from './PreviewCard.jsx';
import ErrorBanner from './ErrorBanner.jsx';
import { canvasFromSource, downscaleToDataUrl, fileToImageElement } from '../lib/imageUtils.js';

/** See WebcamCapture: the floor for a capture taken before the engine reports inputSize. */
const DEFAULT_CAPTURE_SIZE = 320;

const MODES = [
  { id: 'webcam', label: 'Webcam' },
  { id: 'upload', label: 'Upload' },
];

/** Only the facts CapturePanel actually knows; the inference duration is added by the caller. */
function describeSource(width, height) {
  const known = Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0;
  return { capturedAt: Date.now(), ...(known ? { width, height } : null) };
}

const PHOTO_TIPS = [
  'Put the item on a plain, uncluttered background.',
  'Fill the frame — the model only sees a small square crop.',
  'Use even light and avoid harsh shadows or glare.',
  'One item per photo; labels and logos facing the camera help.',
];

export default function CapturePanel({
  mode,
  onModeChange,
  onImageReady,
  busy = false,
  disabled = false,
  error,
  onError,
  mirror = true,
  captureSize = DEFAULT_CAPTURE_SIZE,
}) {
  const currentMode = mode === 'upload' ? 'upload' : 'webcam';
  const [preview, setPreview] = useState(null); // { canvas, dataUrl, source }
  const [localError, setLocalError] = useState(null);
  const [dismissed, setDismissed] = useState(null);
  const captureRegionRef = useRef(null);

  const report = useCallback(
    (err, context) => {
      const normalized = err instanceof Error ? err : new Error(`${context}: ${String(err)}`);
      console.error(`[CapturePanel] ${context}:`, err);
      setLocalError(normalized);
      setDismissed(null);
      onError?.(normalized);
    },
    [onError]
  );

  // A fresh error from the parent should not stay hidden by an earlier dismissal.
  useEffect(() => {
    if (error) setDismissed(null);
  }, [error]);

  const emit = useCallback(
    (payload) => {
      setPreview(payload);
      setLocalError(null);
      onImageReady?.({ canvas: payload.canvas, dataUrl: payload.dataUrl, source: payload.source });
    },
    [onImageReady]
  );

  const handleFile = useCallback(
    async (file) => {
      try {
        const image = await fileToImageElement(file);
        const canvas = await canvasFromSource(image, { size: captureSize });
        const dataUrl = await downscaleToDataUrl(image);
        emit({
          canvas,
          dataUrl,
          source: 'upload',
          meta: describeSource(image?.naturalWidth, image?.naturalHeight),
        });
      } catch (err) {
        report(err, `Could not read "${file?.name || 'that image'}"`);
      }
    },
    [emit, report]
  );

  const handleWebcamCapture = useCallback(
    (payload) => {
      const { canvas, dataUrl } = payload || {};
      if (!canvas) {
        report(new Error('The webcam returned an empty frame.'), 'Webcam capture');
        return;
      }
      emit({ canvas, dataUrl, source: 'webcam', meta: describeSource() });
    },
    [emit, report]
  );

  const handleReclassify = useCallback(() => {
    if (!preview?.canvas) return;
    onImageReady?.({ canvas: preview.canvas, dataUrl: preview.dataUrl, source: preview.source });
  }, [preview, onImageReady]);

  const handleClear = useCallback(() => {
    setPreview(null);
    setLocalError(null);
  }, []);

  const handleUseAnother = useCallback(() => {
    setPreview(null);
    setLocalError(null);
    captureRegionRef.current?.focus();
  }, []);

  const visibleErrorSource = error ?? localError;
  const visibleError = visibleErrorSource && visibleErrorSource !== dismissed ? visibleErrorSource : null;

  return (
    <section className="flex flex-col gap-4" aria-label="Capture an item">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="inline-flex rounded-lg bg-slate-100 p-1 dark:bg-slate-800" role="group" aria-label="Image source">
          {MODES.map((item) => {
            const selected = currentMode === item.id;
            return (
              <button
                key={item.id}
                type="button"
                aria-pressed={selected}
                disabled={disabled}
                onClick={() => onModeChange?.(item.id)}
                className={`rounded-md px-4 py-1.5 text-sm font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-60 ${
                  selected
                    ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-700 dark:text-slate-50'
                    : 'text-slate-600 hover:text-slate-900 dark:text-slate-300 dark:hover:text-white'
                }`}
              >
                {item.label}
              </button>
            );
          })}
        </div>

        {preview ? (
          <button type="button" className="btn btn-ghost text-sm" onClick={handleUseAnother} disabled={busy || disabled}>
            Use another image
          </button>
        ) : null}
      </div>

      {visibleError ? (
        <ErrorBanner
          title="Image capture problem"
          error={visibleError}
          onDismiss={() => setDismissed(visibleErrorSource)}
        />
      ) : null}

      {disabled ? (
        <p className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900 dark:border-amber-500/40 dark:bg-amber-950/40 dark:text-amber-200">
          The classifier is not ready yet, so capturing is switched off. Wait for the model to finish loading — or, if it
          failed, train a custom model (<code>make train</code>) or fetch the MobileNetV2 fallback (<code>make fetch-models</code>).
        </p>
      ) : null}

      <div
        ref={captureRegionRef}
        tabIndex={-1}
        aria-disabled={disabled || undefined}
        className={`flex flex-col gap-4 ${disabled ? 'pointer-events-none opacity-50' : ''}`}
      >
        {currentMode === 'webcam' ? (
          <WebcamCapture
            active={!disabled}
            busy={busy || disabled}
            mirrored={mirror}
            onCapture={handleWebcamCapture}
            onError={(err) => report(err, 'Webcam capture')}
            captureSize={captureSize}
          />
        ) : (
          <ImageDropzone onFile={handleFile} busy={busy || disabled} />
        )}

        <PreviewCard
          dataUrl={preview?.dataUrl}
          source={preview?.source}
          onClear={handleClear}
          onReclassify={handleReclassify}
          busy={busy}
          meta={preview?.meta}
        />
      </div>

      <details className="card p-4 text-sm text-slate-600 dark:text-slate-300">
        <summary className="cursor-pointer font-medium text-slate-800 dark:text-slate-100">
          Tips for a good photo
        </summary>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          {PHOTO_TIPS.map((tip) => (
            <li key={tip}>{tip}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}
