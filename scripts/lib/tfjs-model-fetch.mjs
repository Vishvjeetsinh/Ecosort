// EcoSort - shared downloader for the pretrained TFJS graph models baked into the app.
//
// WHY this exists: EcoSort does zero network I/O at runtime, so every pretrained model's
// weights must already be on disk when the app boots. Each fetch script (fetch-mobilenet.mjs,
// fetch-detector.mjs) is a thin model description handed to runFetcher(); the download,
// verification and atomic install below are shared so the two can never drift apart. It
// deliberately has no npm dependencies so it can run inside a bare `node:22-bookworm-slim`
// build layer before (or entirely without) `npm install`.

import { mkdir, writeFile, readFile, readdir, rename, rm, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

/**
 * @typedef {object} ModelSpec
 * @property {string} name            Short human name for log lines, e.g. "MobileNetV2".
 * @property {string} title           Longer name for the "Fetching ..." line.
 * @property {string} scriptName      Invoking script, for help and retry hints.
 * @property {string} baseUrl         Upstream directory; every remote file is fetched from it.
 * @property {string} dirName         Install path under the models root. May be nested
 *                                    (`detectors/ssdlite_mobilenet_v2`).
 * @property {string[]} remoteFiles   model.json plus every weight shard, in manifest order.
 * @property {number} knownTotalBytes Verified total of the weight shards; a second opinion
 *                                    when the weightsManifest arithmetic cannot be trusted.
 * @property {(downloadBytes: number|null) => object} buildMetadata  metadata.json contents.
 * @property {(fetchedAt: string) => string} buildSourceText         SOURCE.txt contents.
 * @property {string} failureNote     Printed when the install fails.
 */

const RETRY_DELAYS_MS = [500, 1500, 4500];
const REQUEST_TIMEOUT_MS = 120_000;

// On-disk width of one element per TFJS dtype. `quantization.dtype`, when present, wins:
// quantised weights are stored narrow on disk and widened at load time.
const DTYPE_BYTES = {
  float32: 4,
  float64: 8,
  float16: 2,
  int32: 4,
  int64: 8,
  uint32: 4,
  uint16: 2,
  uint8: 1,
  int8: 1,
  bool: 1,
  complex64: 8,
  complex128: 16,
};

const LIB_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(LIB_DIR, '..', '..');

let quiet = false;

function log(message) {
  if (!quiet) process.stdout.write(`${message}\n`);
}

function warn(message) {
  process.stderr.write(`${message}\n`);
}

export function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MiB`;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function printHelp(spec) {
  process.stdout.write(
    [
      `${spec.scriptName} - download the pretrained ${spec.title} TFJS graph model for EcoSort`,
      '',
      'Usage:',
      `  node scripts/${spec.scriptName} [options]`,
      '',
      'Options:',
      `  --dest <dir>   Models root that will contain ${spec.dirName}/ (default: <repo>/models)`,
      '  --force        Re-download even when a complete model is already present',
      '  --quiet        Suppress progress output (errors are still printed)',
      '  -h, --help     Show this help',
      '',
      `Source: ${spec.baseUrl}`,
      '',
    ].join('\n'),
  );
}

function parseArgs(argv) {
  const options = { dest: null, force: false, quiet: false, help: false };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];

    if (arg === '--force') {
      options.force = true;
    } else if (arg === '--quiet' || arg === '-q') {
      options.quiet = true;
    } else if (arg === '--help' || arg === '-h') {
      options.help = true;
    } else if (arg === '--dest') {
      const value = argv[i + 1];
      if (!value || value.startsWith('-')) {
        throw new Error('--dest requires a directory argument');
      }
      options.dest = value;
      i += 1;
    } else if (arg.startsWith('--dest=')) {
      const value = arg.slice('--dest='.length);
      if (!value) throw new Error('--dest requires a directory argument');
      options.dest = value;
    } else {
      throw new Error(`unknown argument: ${arg} (try --help)`);
    }
  }

  return options;
}

/** Element count * element width, or null when the dtype has no fixed width. */
function weightByteLength(spec) {
  const dtype = spec?.quantization?.dtype ?? spec?.dtype;
  const width = DTYPE_BYTES[dtype];
  if (width === undefined) return null;

  const shape = Array.isArray(spec.shape) ? spec.shape : [];
  let count = 1;
  for (const dim of shape) {
    if (!Number.isFinite(dim) || dim < 0) return null;
    count *= dim;
  }
  return count * width;
}

/**
 * Pull the shard list and the expected concatenated byte length out of a model.json.
 * `totalBytes` is null when at least one weight uses a non-fixed-width dtype.
 */
export function summariseManifest(modelJson) {
  const manifest = modelJson?.weightsManifest;
  if (!Array.isArray(manifest) || manifest.length === 0) {
    throw new Error('model.json has no weightsManifest');
  }

  const shards = [];
  let totalBytes = 0;
  let computable = true;

  for (const group of manifest) {
    if (!Array.isArray(group?.paths) || group.paths.length === 0) {
      throw new Error('weightsManifest group is missing its "paths" array');
    }
    for (const shardPath of group.paths) {
      if (typeof shardPath !== 'string' || shardPath.length === 0) {
        throw new Error('weightsManifest contains an empty shard path');
      }
      if (!shards.includes(shardPath)) shards.push(shardPath);
    }

    const weights = Array.isArray(group.weights) ? group.weights : [];
    if (weights.length === 0) {
      throw new Error('weightsManifest group is missing its "weights" array');
    }
    for (const spec of weights) {
      const size = weightByteLength(spec);
      if (size === null) computable = false;
      else totalBytes += size;
    }
  }

  return { shards, totalBytes: computable ? totalBytes : null };
}

async function readModelJson(dir) {
  const modelJsonPath = path.join(dir, 'model.json');
  let text;
  try {
    text = await readFile(modelJsonPath, 'utf8');
  } catch (err) {
    throw new Error(`cannot read ${modelJsonPath}: ${err.message}`, { cause: err });
  }
  try {
    return JSON.parse(text);
  } catch (err) {
    throw new Error(`${modelJsonPath} is not valid JSON: ${err.message}`, { cause: err });
  }
}

async function shardSizes(dir, shards) {
  const sizes = new Map();
  for (const shard of shards) {
    const shardPath = path.join(dir, shard);
    let info;
    try {
      info = await stat(shardPath);
    } catch (err) {
      throw new Error(`missing weight shard ${shard} (${err.code ?? err.message})`, {
        cause: err,
      });
    }
    if (!info.isFile()) throw new Error(`weight shard ${shard} is not a regular file`);
    if (info.size === 0) throw new Error(`weight shard ${shard} is empty`);
    sizes.set(shard, info.size);
  }
  return sizes;
}

/**
 * Cheap completeness probe used for idempotency: model.json parses and every shard it
 * names exists with a non-zero length. Deliberately skips byte totals - that is the job
 * of verifyModelDir immediately after a download.
 */
async function isModelComplete(dir) {
  try {
    const modelJson = await readModelJson(dir);
    const { shards } = summariseManifest(modelJson);
    await shardSizes(dir, shards);
    return true;
  } catch {
    return false;
  }
}

/** Strict post-download verification. Throws with a human-readable reason on failure. */
async function verifyModelDir(dir, knownTotalBytes) {
  const modelJson = await readModelJson(dir);
  const { shards, totalBytes } = summariseManifest(modelJson);
  const sizes = await shardSizes(dir, shards);

  let actualBytes = 0;
  for (const size of sizes.values()) actualBytes += size;

  if (totalBytes === null) {
    // No usable dtype arithmetic - fall back to the verified upstream byte count.
    if (actualBytes !== knownTotalBytes) {
      throw new Error(
        `weight shards total ${actualBytes} bytes, expected ${knownTotalBytes} ` +
          '(and the manifest dtypes were not size-computable)',
      );
    }
  } else if (actualBytes !== totalBytes) {
    if (actualBytes === knownTotalBytes) {
      warn(
        `warning: shard bytes (${actualBytes}) do not match the weightsManifest sum ` +
          `(${totalBytes}), but do match the verified upstream size - accepting.`,
      );
    } else {
      throw new Error(
        `weight shards total ${actualBytes} bytes, but the weightsManifest implies ` +
          `${totalBytes} bytes - the download is incomplete or corrupt`,
      );
    }
  }

  return { shards, actualBytes, expectedBytes: totalBytes ?? knownTotalBytes };
}

async function downloadOnce(url, destPath, { label, position, count }) {
  const response = await fetch(url, {
    redirect: 'follow',
    signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error('response had no body');
  }

  const declared = Number(response.headers.get('content-length'));
  const expected = Number.isFinite(declared) && declared > 0 ? declared : null;
  const showProgress = !quiet && process.stdout.isTTY;

  const chunks = [];
  let received = 0;

  for await (const chunk of response.body) {
    chunks.push(chunk);
    received += chunk.length;
    if (showProgress) {
      const suffix = expected ? ` / ${formatBytes(expected)}` : '';
      process.stdout.write(
        `\r  [${position}/${count}] ${label} ... ${formatBytes(received)}${suffix}      `,
      );
    }
  }

  if (expected !== null && received !== expected) {
    throw new Error(`truncated: received ${received} of ${expected} bytes`);
  }

  await writeFile(destPath, Buffer.concat(chunks));

  if (showProgress) process.stdout.write('\r');
  log(`  [${position}/${count}] ${label} - ${received} bytes (${formatBytes(received)})`);

  return received;
}

async function downloadWithRetry(url, destPath, meta) {
  let lastError = null;

  for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt += 1) {
    if (attempt > 0) {
      const delay = RETRY_DELAYS_MS[attempt - 1];
      warn(
        `  retry ${attempt}/${RETRY_DELAYS_MS.length} for ${meta.label} in ${delay}ms - ` +
          `${lastError.message}`,
      );
      await sleep(delay);
    }
    try {
      return await downloadOnce(url, destPath, meta);
    } catch (err) {
      lastError = err;
    }
  }

  throw new Error(
    `failed to download ${meta.label} after ${RETRY_DELAYS_MS.length + 1} attempts: ` +
      `${lastError.message}`,
    { cause: lastError },
  );
}

/** Remove leftover temp dirs from crashed runs so repeated failures cannot pile up. */
async function cleanStaleTempDirs(parentDir, baseName) {
  let entries;
  try {
    entries = await readdir(parentDir, { withFileTypes: true });
  } catch {
    return; // parentDir may not exist yet - nothing stale to clean
  }

  const prefix = `.${baseName}.tmp-`;
  for (const entry of entries) {
    if (!entry.isDirectory() || !entry.name.startsWith(prefix)) continue;
    if (entry.name === `${prefix}${process.pid}`) continue;
    try {
      await rm(path.join(parentDir, entry.name), { recursive: true, force: true });
    } catch (err) {
      warn(`warning: could not remove stale temp dir ${entry.name}: ${err.message}`);
    }
  }
}

/** Fill in descriptor files next to an already-present model without re-downloading. */
async function ensureSidecarFiles(spec, dir) {
  const wrote = [];
  const metadataPath = path.join(dir, 'metadata.json');
  const sourcePath = path.join(dir, 'SOURCE.txt');

  try {
    await stat(metadataPath);
  } catch {
    let bytes = null;
    try {
      const { actualBytes } = await verifyModelDir(dir, spec.knownTotalBytes);
      bytes = actualBytes + (await stat(path.join(dir, 'model.json'))).size;
    } catch {
      // Backfilling a descriptor must never fail over a size we could not total; the
      // backend weighs the directory itself when downloadBytes is absent.
    }
    await writeFile(metadataPath, `${JSON.stringify(spec.buildMetadata(bytes), null, 2)}\n`);
    wrote.push('metadata.json');
  }

  try {
    await stat(sourcePath);
  } catch {
    await writeFile(sourcePath, spec.buildSourceText(new Date().toISOString()));
    wrote.push('SOURCE.txt');
  }

  return wrote;
}

async function install(spec, options) {
  const destRoot = options.dest
    ? path.resolve(process.cwd(), options.dest)
    : path.join(REPO_ROOT, 'models');
  const finalDir = path.join(destRoot, spec.dirName);
  const parentDir = path.dirname(finalDir);
  const baseName = path.basename(finalDir);

  if (!options.force && (await isModelComplete(finalDir))) {
    const wrote = await ensureSidecarFiles(spec, finalDir);
    log(`${spec.name} already present at ${finalDir} - skipping download.`);
    if (wrote.length > 0) log(`  wrote missing descriptor(s): ${wrote.join(', ')}`);
    log('  (use --force to re-download)');
    return;
  }

  await mkdir(parentDir, { recursive: true });
  await cleanStaleTempDirs(parentDir, baseName);

  // Download into a sibling temp dir and rename on success, so an interrupted run can
  // never leave behind a half-written model that the idempotency probe would accept.
  const tmpDir = path.join(parentDir, `.${baseName}.tmp-${process.pid}`);
  await rm(tmpDir, { recursive: true, force: true });
  await mkdir(tmpDir, { recursive: true });

  try {
    log(`Fetching ${spec.title} into ${finalDir}`);
    log(`  source: ${spec.baseUrl}`);

    let downloaded = 0;
    for (const [index, name] of spec.remoteFiles.entries()) {
      downloaded += await downloadWithRetry(`${spec.baseUrl}${name}`, path.join(tmpDir, name), {
        label: name,
        position: index + 1,
        count: spec.remoteFiles.length,
      });
    }

    const fetchedAt = new Date().toISOString();
    await writeFile(path.join(tmpDir, 'SOURCE.txt'), spec.buildSourceText(fetchedAt));

    const { shards, actualBytes } = await verifyModelDir(tmpDir, spec.knownTotalBytes);

    // Written after verification so downloadBytes can state what the browser actually
    // fetches: every weight shard plus model.json itself.
    const modelJsonBytes = (await stat(path.join(tmpDir, 'model.json'))).size;
    await writeFile(
      path.join(tmpDir, 'metadata.json'),
      `${JSON.stringify(spec.buildMetadata(actualBytes + modelJsonBytes), null, 2)}\n`,
    );

    // rename(2) refuses to replace a non-empty directory, so the old copy goes first.
    await rm(finalDir, { recursive: true, force: true });
    await rename(tmpDir, finalDir);

    log(
      `Verified ${shards.length} weight shard(s), ${actualBytes} bytes ` +
        `(${formatBytes(actualBytes)}); downloaded ${formatBytes(downloaded)} in total.`,
    );
    log(`${spec.name} ready at ${finalDir}`);
  } finally {
    // A no-op once the rename succeeded; the safety net on every failure path.
    await rm(tmpDir, { recursive: true, force: true }).catch((err) => {
      warn(`warning: could not remove temp dir ${tmpDir}: ${err.message}`);
    });
  }
}

/**
 * Parse argv, install `spec`, and turn any failure into a non-zero exit code with a
 * retry hint. Never throws: the scripts that call it are plain top-level awaits.
 *
 * @param {ModelSpec} spec
 * @param {string[]} [argv]
 */
export async function runFetcher(spec, argv = process.argv.slice(2)) {
  try {
    const options = parseArgs(argv);
    if (options.help) {
      printHelp(spec);
      return;
    }
    quiet = options.quiet;
    await install(spec, options);
  } catch (err) {
    warn(`${spec.scriptName.replace(/\.mjs$/, '')}: ${err.message}`);
    if (err.cause && err.cause !== err && err.cause.message !== err.message) {
      warn(`  caused by: ${err.cause.message}`);
    }
    warn(spec.failureNote);
    warn(`Retry with network access:  node scripts/${spec.scriptName} --force`);
    process.exitCode = 1;
  }
}
