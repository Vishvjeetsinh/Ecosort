import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

/**
 * Camera plumbing for <WebcamCapture />.
 *
 * Everything that can go wrong with getUserMedia is turned into a status the UI can
 * render and a message a human can act on. The two rules that matter most:
 *   1. a track that is started must always be stopped (no camera light left on), and
 *   2. a getUserMedia() promise that resolves after the component went away must stop
 *      its own tracks, because nobody else is holding a reference to them.
 */

const IDEAL_WIDTH = 1280;
const IDEAL_HEIGHT = 720;

const UNSUPPORTED_INSECURE =
  'The browser blocks camera access on this page because it is not a secure context. ' +
  'getUserMedia only works on http://localhost, http://127.0.0.1 or an HTTPS origin — ' +
  'opening EcoSort over a LAN address such as http://192.168.1.20:5173 will always fail, ' +
  'with no permission prompt at all. Reach the app through localhost (or an HTTPS tunnel), ' +
  'or use the Upload tab instead.';

const UNSUPPORTED_BROWSER =
  'This browser does not expose navigator.mediaDevices.getUserMedia, so the webcam cannot be used. ' +
  'Use a current Chrome, Edge, Firefox or Safari — or switch to the Upload tab.';

function makeError(name, message, hint = null) {
  const error = new Error(message);
  error.name = name;
  error.hint = hint;
  return error;
}

function hasGetUserMedia() {
  return (
    typeof navigator !== 'undefined' &&
    Boolean(navigator.mediaDevices) &&
    typeof navigator.mediaDevices.getUserMedia === 'function'
  );
}

/** localhost is treated as secure by browsers even over plain http. */
function inSecureContext() {
  if (typeof window === 'undefined') return false;
  if (window.isSecureContext) return true;
  const host = window.location?.hostname ?? '';
  return host === 'localhost' || host === '127.0.0.1' || host === '[::1]' || host === '::1';
}

function unsupportedError() {
  return inSecureContext()
    ? makeError('WebcamUnsupportedError', UNSUPPORTED_BROWSER)
    : makeError('WebcamUnsupportedError', UNSUPPORTED_INSECURE, 'Open http://localhost:5173 instead of the machine IP address.');
}

function stopStream(stream) {
  if (!stream || typeof stream.getTracks !== 'function') return;
  for (const track of stream.getTracks()) {
    try {
      track.stop();
    } catch (err) {
      // A track can already be dead (device unplugged); losing the camera light is the
      // only thing that would be unacceptable, and that has already happened here.
      console.warn('[useWebcam] could not stop a %s track:', track.kind, err);
    }
  }
}

function readCapabilities(track) {
  if (!track) return null;
  const settings = typeof track.getSettings === 'function' ? track.getSettings() : {};
  let caps = {};
  if (typeof track.getCapabilities === 'function') {
    try {
      caps = track.getCapabilities() || {};
    } catch (err) {
      console.warn('[useWebcam] getCapabilities() is unavailable on this platform:', err);
    }
  }
  return {
    label: track.label || 'Camera',
    deviceId: settings.deviceId ?? null,
    facingMode: settings.facingMode ?? null,
    width: settings.width ?? null,
    height: settings.height ?? null,
    frameRate: settings.frameRate ?? null,
    hasTorch: Boolean(caps.torch),
    zoom: caps.zoom ?? null,
  };
}

function describeFailure(err) {
  const name = err?.name || 'Error';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return {
        status: 'denied',
        error: makeError(
          'WebcamPermissionError',
          'Camera permission was denied for this site.',
          'Click the camera (or lock) icon on the left of the address bar, set Camera to "Allow", ' +
            'then press Retry. In Chrome you can also use Settings → Privacy and security → Site settings → Camera.'
        ),
      };
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return {
        status: 'error',
        error: makeError(
          'WebcamNotFoundError',
          'No camera was found on this device.',
          'Plug a webcam in and press Retry, or use the Upload tab.'
        ),
      };
    case 'NotReadableError':
    case 'TrackStartError':
      return {
        status: 'error',
        error: makeError(
          'WebcamBusyError',
          'The camera could not be read — another application is most likely using it.',
          'Close video calls or other tabs holding the camera, then press Retry.'
        ),
      };
    case 'OverconstrainedError':
      return {
        status: 'error',
        error: makeError(
          'WebcamConstraintError',
          `This camera cannot satisfy the requested settings (constraint: ${err?.constraint || 'unknown'}), ` +
            'and the relaxed retry failed too.',
          'Pick a different camera from the list, then press Retry.'
        ),
      };
    case 'AbortError':
      return {
        status: 'error',
        error: makeError(
          'WebcamAbortError',
          'The camera was claimed by the system before it finished starting.',
          'Press Retry; if it keeps happening, unplug and replug the camera.'
        ),
      };
    case 'TypeError':
      // getUserMedia throws TypeError when the constraints are empty *or* when the API is
      // shimmed away on an insecure origin — the second case is the common one here.
      return { status: 'unsupported', error: unsupportedError() };
    default:
      return {
        status: 'error',
        error: makeError('WebcamError', `The camera could not be started (${name}): ${err?.message || 'unknown error'}.`),
      };
  }
}

export function useWebcam({ enabled = false, facingMode = 'environment', deviceId: preferredDeviceId = null } = {}) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const mountedRef = useRef(true);
  // Bumped on every start/teardown; a resolved getUserMedia whose token is stale is orphaned.
  const startTokenRef = useRef(0);

  const [stream, setStream] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [devices, setDevices] = useState([]);
  const [deviceId, setDeviceIdState] = useState(preferredDeviceId ?? null);
  const [capabilities, setCapabilities] = useState(null);
  const [attempt, setAttempt] = useState(0);

  const deviceIdRef = useRef(deviceId);
  useEffect(() => {
    deviceIdRef.current = deviceId;
  }, [deviceId]);

  // Mount bookkeeping is declared first so that on a StrictMode remount `mountedRef`
  // is true again before the start effect below runs.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      startTokenRef.current += 1;
      stopStream(streamRef.current);
      streamRef.current = null;
    };
  }, []);

  useEffect(() => {
    setDeviceIdState((prev) => (preferredDeviceId && preferredDeviceId !== prev ? preferredDeviceId : prev));
  }, [preferredDeviceId]);

  const refreshDevices = useCallback(async () => {
    if (!hasGetUserMedia() || typeof navigator.mediaDevices.enumerateDevices !== 'function') return;
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      if (!mountedRef.current) return;
      const cameras = all
        .filter((device) => device.kind === 'videoinput')
        .map((device, index) => ({
          deviceId: device.deviceId,
          // Labels stay empty until permission has been granted at least once.
          label: device.label || `Camera ${index + 1}`,
          groupId: device.groupId || null,
        }));
      setDevices(cameras);
    } catch (err) {
      console.warn('[useWebcam] enumerateDevices() failed; keeping the previous camera list:', err);
    }
  }, []);

  const teardown = useCallback(() => {
    startTokenRef.current += 1;
    stopStream(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    if (!mountedRef.current) return;
    setStream(null);
    setCapabilities(null);
  }, []);

  const stop = useCallback(() => {
    teardown();
    if (!mountedRef.current) return;
    setStatus((prev) => (prev === 'unsupported' ? prev : 'idle'));
    setError((prev) => (prev?.name === 'WebcamUnsupportedError' ? prev : null));
  }, [teardown]);

  const beginStream = useCallback(
    async (targetDeviceId) => {
      if (!hasGetUserMedia()) {
        if (!mountedRef.current) return;
        setStatus('unsupported');
        setError(unsupportedError());
        return;
      }

      // Always release the previous camera before asking for another one: switching
      // devices while the old track is live fails outright on several platforms.
      teardown();
      const token = startTokenRef.current;
      if (!mountedRef.current) return;
      setStatus('requesting');
      setError(null);

      const run = async (relaxed) => {
        const video = relaxed
          ? true
          : {
              ...(targetDeviceId ? { deviceId: { exact: targetDeviceId } } : { facingMode }),
              width: { ideal: IDEAL_WIDTH },
              height: { ideal: IDEAL_HEIGHT },
            };

        let media;
        try {
          media = await navigator.mediaDevices.getUserMedia({ video, audio: false });
        } catch (err) {
          if (!mountedRef.current || token !== startTokenRef.current) return;
          if (err?.name === 'OverconstrainedError' && !relaxed) {
            // The exact deviceId or the 720p ideal was impossible: retry once, bare.
            await run(true);
            return;
          }
          const failure = describeFailure(err);
          setStatus(failure.status);
          setError(failure.error);
          return;
        }

        // The permission prompt can sit open for minutes — by the time it is answered the
        // component may be gone, in which case nothing else will ever stop these tracks.
        if (!mountedRef.current || token !== startTokenRef.current) {
          stopStream(media);
          return;
        }

        streamRef.current = media;
        setStream(media);
        setStatus('live');
        setError(null);
        setCapabilities(readCapabilities(media.getVideoTracks()[0] || null));
        // Device labels are only readable once permission has been granted.
        void refreshDevices();
      };

      await run(false);
    },
    [facingMode, refreshDevices, teardown]
  );

  // Attach the stream to the <video> element and start playback.
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;
    if (video.srcObject !== stream) video.srcObject = stream;
    if (!stream) return undefined;

    let cancelled = false;
    const played = video.play();
    if (played && typeof played.catch === 'function') {
      played.catch((err) => {
        // AbortError just means a newer srcObject superseded this play() call, and a
        // stream that is no longer the active one has already been torn down on purpose.
        if (cancelled || err?.name === 'AbortError' || streamRef.current !== stream) return;
        console.warn('[useWebcam] video.play() was rejected:', err);
        if (!mountedRef.current) return;
        setStatus('error');
        setError(
          makeError(
            'WebcamPlaybackError',
            `The browser refused to play the camera preview (${err?.name || 'error'}).`,
            'Press Retry; the preview is muted and inline, so autoplay should normally be allowed.'
          )
        );
      });
    }
    return () => {
      cancelled = true;
    };
  }, [stream]);

  // A track ends on its own when the device is unplugged or stolen by another app.
  useEffect(() => {
    if (!stream) return undefined;
    const tracks = stream.getVideoTracks();
    const onEnded = () => {
      if (!mountedRef.current) return;
      setStatus('error');
      setError(
        makeError(
          'WebcamEndedError',
          'The camera stopped sending video — it was unplugged or taken over by another application.',
          'Press Retry to reconnect.'
        )
      );
      teardown();
    };
    for (const track of tracks) track.addEventListener('ended', onEnded);
    return () => {
      for (const track of tracks) track.removeEventListener('ended', onEnded);
    };
  }, [stream, teardown]);

  useEffect(() => {
    if (!hasGetUserMedia() || typeof navigator.mediaDevices.addEventListener !== 'function') return undefined;
    const onDeviceChange = () => {
      void refreshDevices();
    };
    navigator.mediaDevices.addEventListener('devicechange', onDeviceChange);
    return () => navigator.mediaDevices.removeEventListener('devicechange', onDeviceChange);
  }, [refreshDevices]);

  // The single owner of start/stop: flipping `enabled`, switching camera, changing
  // facingMode or calling retry() all funnel through here.
  useEffect(() => {
    if (!enabled) {
      stop();
      return undefined;
    }
    void beginStream(deviceId);
    return undefined;
  }, [enabled, deviceId, attempt, beginStream, stop]);

  const start = useCallback(() => {
    setError(null);
    void beginStream(deviceIdRef.current);
  }, [beginStream]);

  const retry = useCallback(() => {
    setError(null);
    setAttempt((value) => value + 1);
  }, []);

  const setDeviceId = useCallback((nextDeviceId) => {
    setDeviceIdState(nextDeviceId || null);
  }, []);

  return useMemo(
    () => ({ videoRef, stream, status, error, devices, deviceId, setDeviceId, start, stop, retry, capabilities }),
    [stream, status, error, devices, deviceId, setDeviceId, start, stop, retry, capabilities]
  );
}

export default useWebcam;
