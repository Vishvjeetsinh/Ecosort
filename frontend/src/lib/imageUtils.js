/**
 * Browser image helpers shared by the capture UI and the classifier.
 *
 * EXIF is deliberately not parsed by hand. `createImageBitmap(file, {
 * imageOrientation: 'from-image' })` bakes the camera's orientation tag into
 * the decoded pixels, which is exactly what we need before drawing to a canvas
 * -- a raw <img> leaves the rotation to CSS and some engines then feed the
 * unrotated buffer to `drawImage`. The <img> path stays as a graceful fallback
 * for browsers without `createImageBitmap`.
 */

export const MAX_IMAGE_BYTES = 20 * 1024 * 1024;

/** Canvas dimensions are capped so a 50-megapixel phone photo cannot OOM the tab. */
const MAX_DECODED_SIDE = 4096;

const RE_ENCODABLE_TYPES = new Set(['image/jpeg', 'image/png', 'image/webp']);

function assertImageFile(file) {
  if (!file || typeof file !== 'object' || typeof file.type !== 'string') {
    throw new TypeError('Expected a File or Blob');
  }
  if (!file.type.startsWith('image/')) {
    throw new Error(
      `Unsupported file type "${file.type || 'unknown'}" — choose a JPEG, PNG, WebP or GIF image.`,
    );
  }
  if (typeof file.size === 'number' && file.size > MAX_IMAGE_BYTES) {
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    throw new Error(`Image is ${mb} MB — the limit is 20 MB. Try a smaller photo.`);
  }
}

function loadImageFromUrl(url, revoke) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => {
      if (revoke) URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = () => {
      if (revoke) URL.revokeObjectURL(url);
      reject(new Error('The image could not be decoded — the file may be corrupt.'));
    };
    img.src = url;
  });
}

function createCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(width));
  canvas.height = Math.max(1, Math.round(height));
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    throw new Error('This browser refused a 2D canvas context, so images cannot be processed.');
  }
  return { canvas, ctx };
}

function canvasToBlob(canvas, type, quality) {
  return new Promise((resolve, reject) => {
    if (typeof canvas.toBlob !== 'function') {
      reject(new Error('canvas.toBlob is unavailable'));
      return;
    }
    canvas.toBlob(
      (blob) => {
        if (blob) resolve(blob);
        else reject(new Error('The browser failed to encode the processed image.'));
      },
      type,
      quality,
    );
  });
}

async function bitmapToImageElement(bitmap, mimeType) {
  const scale = Math.min(1, MAX_DECODED_SIDE / Math.max(bitmap.width, bitmap.height));
  const { canvas, ctx } = createCanvas(bitmap.width * scale, bitmap.height * scale);
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

  const type = RE_ENCODABLE_TYPES.has(mimeType) ? mimeType : 'image/png';
  const quality = type === 'image/jpeg' ? 0.92 : undefined;

  try {
    const blob = await canvasToBlob(canvas, type, quality);
    const url = URL.createObjectURL(blob);
    return await loadImageFromUrl(url, true);
  } catch {
    // toBlob is missing or refused the type: a data URL always works, it is
    // just heavier, so it is the second choice rather than the first.
    return loadImageFromUrl(canvas.toDataURL(type, quality), false);
  }
}

/**
 * Decode a user-supplied file into an orientation-corrected image element.
 *
 * @param {File|Blob} file
 * @returns {Promise<HTMLImageElement>}
 */
export async function fileToImageElement(file) {
  assertImageFile(file);

  let bitmapError = null;
  if (typeof createImageBitmap === 'function') {
    let bitmap = null;
    try {
      bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch (orientationError) {
      // Safari < 16 rejects the options bag; retry without it before giving up.
      try {
        bitmap = await createImageBitmap(file);
      } catch (plainError) {
        bitmapError = new Error(
          `createImageBitmap failed (${String(orientationError?.message ?? orientationError)}; ` +
            `${String(plainError?.message ?? plainError)})`,
          { cause: plainError },
        );
      }
    }
    if (bitmap) {
      try {
        return await bitmapToImageElement(bitmap, file.type);
      } finally {
        if (typeof bitmap.close === 'function') bitmap.close();
      }
    }
  }

  const url = URL.createObjectURL(file);
  try {
    return await loadImageFromUrl(url, true);
  } catch (err) {
    if (bitmapError) {
      throw new Error(`${err.message} (decoder also reported: ${bitmapError.message})`, {
        cause: bitmapError,
      });
    }
    throw err;
  }
}

/**
 * Natural pixel dimensions of anything drawable.
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 * @returns {{width: number, height: number}}
 */
export function getSourceSize(source) {
  if (!source || typeof source !== 'object') {
    throw new TypeError('Expected an image, video, canvas or bitmap source');
  }

  const width =
    Number(source.naturalWidth) || Number(source.videoWidth) || Number(source.width) || 0;
  const height =
    Number(source.naturalHeight) || Number(source.videoHeight) || Number(source.height) || 0;

  if (!width || !height) {
    throw new Error(
      'The image source has no dimensions yet — wait for it to load before classifying.',
    );
  }
  return { width, height };
}

/**
 * Centre-cropped square canvas, scaled with "cover" semantics.
 *
 * Webcam frames are 4:3 or 16:9; stretching them to the model's square input
 * distorts every object in them, so the short edge defines a square that is
 * cropped from the centre instead.
 *
 * This is the whole preprocessing contract's first step, and both sides honour it:
 * the bundled ImageNet models were themselves trained resize-then-centre-crop, and
 * ml/train.py passes `crop_to_aspect_ratio=True` so the custom model sees the same
 * geometry. Change the crop here and you have to change it there in the same commit.
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 * @param {{size?: number}} [options]
 * @returns {HTMLCanvasElement}
 */
export function canvasFromSource(source, { size = 224 } = {}) {
  if (!Number.isFinite(size) || size <= 0) {
    throw new RangeError(`canvasFromSource: size must be a positive number, got ${size}`);
  }

  const { width, height } = getSourceSize(source);
  const side = Math.min(width, height);
  const sx = (width - side) / 2;
  const sy = (height - side) / 2;

  const target = Math.round(size);
  const { canvas, ctx } = createCanvas(target, target);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, sx, sy, side, side, 0, 0, target, target);
  return canvas;
}

/**
 * @param {HTMLCanvasElement} canvas
 * @param {{type?: string, quality?: number}} [options]
 * @returns {string} data URL
 */
export function canvasToDataUrl(canvas, { type = 'image/jpeg', quality = 0.82 } = {}) {
  if (!canvas || typeof canvas.toDataURL !== 'function') {
    throw new TypeError('canvasToDataUrl expects an HTMLCanvasElement');
  }
  const dataUrl = canvas.toDataURL(type, quality);
  if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/')) {
    throw new Error('The browser failed to encode the canvas as an image.');
  }
  return dataUrl;
}

/**
 * Aspect-preserving downscale, used for the history thumbnails that are POSTed
 * to the backend (which caps `imageDataUrl` at 400 000 characters).
 *
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement|ImageBitmap} source
 * @param {{maxSide?: number, type?: string, quality?: number}} [options]
 * @returns {string} data URL
 */
export function downscaleToDataUrl(
  source,
  { maxSide = 320, type = 'image/jpeg', quality = 0.75 } = {},
) {
  if (!Number.isFinite(maxSide) || maxSide <= 0) {
    throw new RangeError(`downscaleToDataUrl: maxSide must be positive, got ${maxSide}`);
  }

  const { width, height } = getSourceSize(source);
  const scale = Math.min(1, maxSide / Math.max(width, height));
  const { canvas, ctx } = createCanvas(width * scale, height * scale);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  return canvasToDataUrl(canvas, { type, quality });
}
