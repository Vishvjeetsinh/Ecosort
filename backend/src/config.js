import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * Paths are derived from this module's own location rather than process.cwd() so the
 * server behaves identically whether it is started from the repo root, from backend/,
 * by nodemon, or by `node /app/src/server.js` inside the container.
 */
const MODULE_DIR = path.dirname(fileURLToPath(import.meta.url)); // <repo>/backend/src
const PACKAGE_ROOT = path.resolve(MODULE_DIR, '..'); // <repo>/backend
const REPO_ROOT = path.resolve(PACKAGE_ROOT, '..'); // <repo>

/** Treat an empty/whitespace-only env var as "not set" — compose passes empty strings. */
function readEnv(name, fallback = undefined) {
  const raw = process.env[name];
  if (raw === undefined || raw === null) return fallback;
  const trimmed = String(raw).trim();
  return trimmed === '' ? fallback : trimmed;
}

function readIntEnv(name, fallback, { min = Number.MIN_SAFE_INTEGER, max = Number.MAX_SAFE_INTEGER } = {}) {
  const raw = readEnv(name);
  if (raw === undefined) return fallback;
  const parsed = Number.parseInt(raw, 10);
  if (!Number.isInteger(parsed)) {
    throw new Error(`Invalid ${name}: expected an integer, received "${raw}"`);
  }
  if (parsed < min || parsed > max) {
    throw new Error(`Invalid ${name}: ${parsed} is outside the allowed range ${min}..${max}`);
  }
  return parsed;
}

/**
 * Relative directory values resolve against the repo root, because the documented host
 * defaults in .env.example are written repo-relative ("./models", "./backend/data")
 * while compose/Dockerfile pass absolute container paths ("/models", "/data").
 */
function resolveDir(value, fallbackAbsolute) {
  if (value === undefined) return fallbackAbsolute;
  return path.isAbsolute(value) ? path.normalize(value) : path.resolve(REPO_ROOT, value);
}

function resolveDbFile(value, dataDir) {
  if (value === undefined) return path.join(dataDir, 'ecosort.db');
  if (value === ':memory:') return value; // better-sqlite3 sentinel, never a filesystem path
  return path.isAbsolute(value) ? path.normalize(value) : path.resolve(dataDir, value);
}

function parseCorsOrigin(value) {
  const parts = value
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
  return parts.length === 0 ? ['http://localhost:5173'] : parts;
}

function readPackageVersion() {
  const pkgPath = path.join(PACKAGE_ROOT, 'package.json');
  try {
    const parsed = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    return typeof parsed.version === 'string' ? parsed.version : '0.0.0';
  } catch (err) {
    // /api/health must still answer if package.json is unreadable; surface the reason once.
    process.stderr.write(`[ecosort:config] could not read version from ${pkgPath}: ${err.message}\n`);
    return '0.0.0';
  }
}

const nodeEnv = readEnv('NODE_ENV', 'development');
const dataDir = resolveDir(readEnv('DATA_DIR'), path.join(PACKAGE_ROOT, 'data'));
const dbFile = resolveDbFile(readEnv('DB_FILE'), dataDir);
const corsOrigins = parseCorsOrigin(readEnv('CORS_ORIGIN', 'http://localhost:5173'));

/** SQLite cannot create its own parent directory — do it before anything opens the db. */
function ensureDataDir() {
  if (dbFile === ':memory:') return;
  try {
    fs.mkdirSync(dataDir, { recursive: true });
  } catch (err) {
    throw new Error(`Cannot create DATA_DIR "${dataDir}": ${err.message}`);
  }
}

ensureDataDir();

export const config = Object.freeze({
  nodeEnv,
  isProduction: nodeEnv === 'production',
  isTest: nodeEnv === 'test',
  isDevelopment: nodeEnv !== 'production' && nodeEnv !== 'test',

  version: readPackageVersion(),

  backendPort: readIntEnv('BACKEND_PORT', 4000, { min: 0, max: 65535 }),
  frontendPort: readIntEnv('FRONTEND_PORT', 5173, { min: 0, max: 65535 }),

  packageRoot: PACKAGE_ROOT,
  repoRoot: REPO_ROOT,
  dataDir,
  dbFile,
  isMemoryDb: dbFile === ':memory:',

  modelsDir: resolveDir(readEnv('MODELS_DIR'), path.join(REPO_ROOT, 'models')),
  bundledModelsDir: resolveDir(readEnv('BUNDLED_MODELS_DIR'), path.join(REPO_ROOT, 'models')),

  corsOrigins: Object.freeze(corsOrigins),
  corsAllowAll: corsOrigins.includes('*'),

  jsonBodyLimit: readEnv('JSON_BODY_LIMIT', '2mb'),
  maxImageDataUrl: readIntEnv('MAX_IMAGE_DATA_URL', 400000, { min: 0, max: 5000000 }),

  logLevel: (readEnv('LOG_LEVEL', 'info') || 'info').toLowerCase(),
  defaultRegion: readEnv('DEFAULT_REGION', 'us-generic'),
});

/** A log-safe snapshot: only resolved paths and non-secret scalars. */
export function describeConfig() {
  return {
    nodeEnv: config.nodeEnv,
    version: config.version,
    backendPort: config.backendPort,
    dataDir: config.dataDir,
    dbFile: config.dbFile,
    modelsDir: config.modelsDir,
    modelsDirExists: fs.existsSync(config.modelsDir),
    bundledModelsDir: config.bundledModelsDir,
    bundledModelsDirExists: fs.existsSync(config.bundledModelsDir),
    corsOrigins: config.corsAllowAll ? '*' : [...config.corsOrigins],
    maxImageDataUrl: config.maxImageDataUrl,
    jsonBodyLimit: config.jsonBodyLimit,
    logLevel: config.logLevel,
    defaultRegion: config.defaultRegion,
  };
}

export default config;
