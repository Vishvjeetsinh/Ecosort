import { useCallback, useEffect, useId, useRef, useState } from 'react';
import Spinner from './Spinner.jsx';
import { MAX_IMAGE_BYTES } from '../lib/imageUtils.js';

// Same ceiling the decoder enforces, so a file can never pass this gate and fail later.
const MAX_BYTES = MAX_IMAGE_BYTES;
const MAX_LABEL = `${Math.round(MAX_IMAGE_BYTES / (1024 * 1024))} MB`;

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'an unknown size';
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} bytes`;
}

/**
 * imageUtils.fileToImageElement() insists on a real image/* MIME type, so reject the same
 * set here — the user gets the message next to the drop target instead of in an error banner.
 */
function looksLikeImage(file) {
  return (file?.type || '').toLowerCase().startsWith('image/');
}

function matchesAccept(file, accept) {
  const rules = String(accept || 'image/*')
    .split(',')
    .map((rule) => rule.trim().toLowerCase())
    .filter(Boolean);
  if (rules.length === 0) return true;
  const type = (file.type || '').toLowerCase();
  const name = (file.name || '').toLowerCase();
  return rules.some((rule) => {
    if (rule === '*/*') return true;
    if (rule.startsWith('.')) return name.endsWith(rule);
    if (rule.endsWith('/*')) return type.startsWith(rule.slice(0, -1));
    return type === rule;
  });
}

export default function ImageDropzone({ onFile, busy = false, accept = 'image/*' }) {
  const [dragging, setDragging] = useState(false);
  const [message, setMessage] = useState(null); // { tone: 'error' | 'info', text }
  const inputId = useId();
  // dragenter/dragleave also fire for child nodes; count depth so the highlight does not flicker.
  const dragDepth = useRef(0);

  const onFileRef = useRef(onFile);
  const busyRef = useRef(busy);
  const acceptRef = useRef(accept);
  useEffect(() => {
    onFileRef.current = onFile;
    busyRef.current = busy;
    acceptRef.current = accept;
  }, [onFile, busy, accept]);

  const acceptFiles = useCallback((fileList) => {
    const files = Array.from(fileList || []);
    if (files.length === 0) {
      setMessage({ tone: 'error', text: 'No file was found in that drop — try choosing one instead.' });
      return;
    }
    const file = files[0];

    if (!looksLikeImage(file) || !matchesAccept(file, acceptRef.current)) {
      setMessage({
        tone: 'error',
        text: `"${file.name || 'That file'}" is not an accepted image — the browser reported "${file.type || 'no type at all'}". Use PNG, JPEG, WebP, GIF or AVIF.`,
      });
      return;
    }
    if (file.size > MAX_BYTES) {
      setMessage({
        tone: 'error',
        text: `"${file.name || 'That image'}" is ${formatBytes(file.size)} — the limit is ${MAX_LABEL}. Resize it and try again.`,
      });
      return;
    }

    setMessage(files.length > 1 ? { tone: 'info', text: `Using "${file.name}" — extra files were ignored.` } : null);
    onFileRef.current?.(file);
  }, []);

  // Paste support: works anywhere on the page while this dropzone is mounted.
  useEffect(() => {
    const onPaste = (event) => {
      if (busyRef.current) return;
      const clipboard = event.clipboardData;
      if (!clipboard) return;
      let picked = null;
      for (const item of Array.from(clipboard.items || [])) {
        if (item.kind === 'file' && (item.type || '').startsWith('image/')) {
          picked = item.getAsFile();
          if (picked) break;
        }
      }
      if (!picked) {
        const first = clipboard.files?.[0];
        if (first && looksLikeImage(first)) picked = first;
      }
      if (!picked) return;
      event.preventDefault();
      acceptFiles([picked]);
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [acceptFiles]);

  const handleChange = useCallback(
    (event) => {
      acceptFiles(event.target.files);
      // Reset so picking the very same file again still fires a change event.
      event.target.value = '';
    },
    [acceptFiles]
  );

  const handleDragEnter = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    if (busyRef.current) return;
    dragDepth.current += 1;
    setDragging(true);
  }, []);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    if (busyRef.current) return;
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
  }, []);

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    event.stopPropagation();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  }, []);

  const handleDrop = useCallback(
    (event) => {
      event.preventDefault();
      event.stopPropagation();
      dragDepth.current = 0;
      setDragging(false);
      if (busyRef.current) return;
      acceptFiles(event.dataTransfer?.files);
    },
    [acceptFiles]
  );

  const stateClasses = dragging
    ? 'border-brand-500 bg-brand-50 dark:bg-brand-950/40'
    : 'border-slate-300 hover:border-brand-400 dark:border-slate-600 dark:hover:border-brand-500';

  return (
    <div className="card flex flex-col gap-3 p-4">
      <input
        id={inputId}
        type="file"
        accept={accept}
        className="sr-only peer"
        onChange={handleChange}
        disabled={busy}
      />
      <label
        htmlFor={inputId}
        onDragEnter={handleDragEnter}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={`relative flex min-h-[14rem] cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 text-center transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-brand-500 peer-disabled:cursor-not-allowed peer-disabled:opacity-60 ${stateClasses}`}
      >
        {busy ? (
          <Spinner label="Working on the previous image" />
        ) : (
          <>
            <svg
              className="h-10 w-10 text-slate-400 dark:text-slate-500"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 16V4m0 0L8 8m4-4 4 4" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" />
            </svg>
            <p className="text-sm font-medium text-slate-700 dark:text-slate-200">
              {dragging ? 'Drop the image to classify it' : 'Drop an image here, or click to choose one'}
            </p>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              You can also paste an image with <kbd className="rounded border border-slate-300 px-1 dark:border-slate-600">Ctrl</kbd>
              <span aria-hidden="true"> / </span>
              <kbd className="rounded border border-slate-300 px-1 dark:border-slate-600">⌘</kbd>
              <kbd className="ml-1 rounded border border-slate-300 px-1 dark:border-slate-600">V</kbd>
            </p>
            <p className="text-xs text-slate-400 dark:text-slate-500">PNG, JPEG, WebP, GIF or AVIF · up to {MAX_LABEL}</p>
          </>
        )}
      </label>

      {message ? (
        <p
          role={message.tone === 'error' ? 'alert' : 'status'}
          aria-live="polite"
          className={
            message.tone === 'error'
              ? 'text-sm text-red-600 dark:text-red-400'
              : 'text-sm text-slate-600 dark:text-slate-300'
          }
        >
          {message.text}
        </p>
      ) : null}
    </div>
  );
}
