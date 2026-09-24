import util from 'node:util';

/**
 * Deliberately dependency-free: the logger is imported by the error handler and by the
 * bootstrap path, so it must never be able to fail for lack of a transport.
 */
const LEVELS = Object.freeze({ error: 0, warn: 1, info: 2, debug: 3 });
const DEFAULT_LEVEL = 'info';

const ESC = String.fromCharCode(27);
const ANSI = Object.freeze({
  reset: ESC + '[0m',
  dim: ESC + '[2m',
  bold: ESC + '[1m',
  red: ESC + '[31m',
  yellow: ESC + '[33m',
  cyan: ESC + '[36m',
  grey: ESC + '[90m',
});

const LEVEL_COLOR = Object.freeze({
  error: ANSI.red,
  warn: ANSI.yellow,
  info: ANSI.cyan,
  debug: ANSI.grey,
});

function normaliseLevel(value, fallback = DEFAULT_LEVEL) {
  const key = String(value ?? '').trim().toLowerCase();
  return Object.prototype.hasOwnProperty.call(LEVELS, key) ? key : fallback;
}

let activeLevel = normaliseLevel(process.env.LOG_LEVEL);
const asJson = process.env.NODE_ENV === 'production';
const useColor = !asJson && process.env.NO_COLOR === undefined && process.stdout.isTTY === true;

export function setLogLevel(level) {
  activeLevel = normaliseLevel(level, activeLevel);
  return activeLevel;
}

export function getLogLevel() {
  return activeLevel;
}

/** Errors carry no enumerable properties — unwrap them so stacks survive serialisation. */
function serialiseError(err) {
  return {
    name: err.name,
    message: err.message,
    ...(err.code ? { code: err.code } : {}),
    ...(err.stack ? { stack: err.stack } : {}),
  };
}

function normaliseMeta(meta) {
  if (meta === undefined || meta === null) return undefined;
  if (meta instanceof Error) return { error: serialiseError(meta) };
  if (typeof meta !== 'object') return { value: meta };
  const out = {};
  for (const [key, value] of Object.entries(meta)) {
    out[key] = value instanceof Error ? serialiseError(value) : value;
  }
  return out;
}

/** JSON.stringify throws on cycles and BigInt; a logger must never throw. */
function safeStringify(value) {
  const seen = new WeakSet();
  try {
    return JSON.stringify(value, (_key, val) => {
      if (typeof val === 'bigint') return val.toString();
      if (typeof val === 'function') return `[Function ${val.name || 'anonymous'}]`;
      if (typeof val === 'object' && val !== null) {
        if (seen.has(val)) return '[Circular]';
        seen.add(val);
      }
      return val;
    });
  } catch (err) {
    return JSON.stringify({ logSerializationError: String(err && err.message) });
  }
}

function write(level, scope, message, meta) {
  if (LEVELS[level] > LEVELS[activeLevel]) return;

  const timestamp = new Date().toISOString();
  const normalisedMeta = normaliseMeta(meta);
  const stream = level === 'error' || level === 'warn' ? process.stderr : process.stdout;

  if (asJson) {
    const payload = { time: timestamp, level, scope, msg: String(message), ...(normalisedMeta ?? {}) };
    stream.write(safeStringify(payload) + '\n');
    return;
  }

  const clock = timestamp.slice(11, 23);
  const tag = level.toUpperCase().padEnd(5, ' ');
  const head = useColor
    ? `${ANSI.dim}${clock}${ANSI.reset} ${LEVEL_COLOR[level]}${tag}${ANSI.reset} ${ANSI.bold}[${scope}]${ANSI.reset}`
    : `${clock} ${tag} [${scope}]`;
  const tail =
    normalisedMeta === undefined
      ? ''
      : ' ' + util.inspect(normalisedMeta, { depth: 4, colors: useColor, breakLength: 120, compact: 3 });

  stream.write(`${head} ${message}${tail}\n`);
}

export function createLogger(scope = 'ecosort') {
  const name = String(scope);
  return {
    scope: name,
    error: (message, meta) => write('error', name, message, meta),
    warn: (message, meta) => write('warn', name, message, meta),
    info: (message, meta) => write('info', name, message, meta),
    debug: (message, meta) => write('debug', name, message, meta),
    child: (childScope) => createLogger(`${name}:${childScope}`),
  };
}

export const logger = createLogger('ecosort');

export default logger;
