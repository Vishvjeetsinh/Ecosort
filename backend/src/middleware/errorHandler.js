import { config } from '../config.js';
import { createLogger } from '../logger.js';
import { HttpError, formatZodIssues, isZodError } from './validate.js';

const log = createLogger('http');

const STATUS_CODES = Object.freeze({
  400: 'bad_request',
  404: 'not_found',
  413: 'payload_too_large',
});

/** body-parser tags its failures with `err.type`; map them onto the documented codes. */
const BODY_PARSER_ERRORS = Object.freeze({
  'entity.too.large': { status: 413, code: 'payload_too_large', message: 'Request body is too large' },
  'entity.parse.failed': { status: 400, code: 'bad_request', message: 'Request body is not valid JSON' },
  'entity.verify.failed': { status: 400, code: 'bad_request', message: 'Request body failed verification' },
  'encoding.unsupported': { status: 400, code: 'bad_request', message: 'Unsupported content encoding' },
  'charset.unsupported': { status: 400, code: 'bad_request', message: 'Unsupported charset' },
  'request.aborted': { status: 400, code: 'bad_request', message: 'Request aborted by the client' },
  'request.size.invalid': { status: 400, code: 'bad_request', message: 'Request size did not match Content-Length' },
  'parameters.too.many': { status: 400, code: 'bad_request', message: 'Too many request parameters' },
});

function classify(err) {
  if (err instanceof HttpError) {
    return { status: err.status, code: err.code, message: err.message, details: err.details ?? null };
  }

  if (isZodError(err)) {
    return { status: 400, code: 'bad_request', message: 'Request failed validation', details: formatZodIssues(err) };
  }

  const bodyParser = err && typeof err.type === 'string' ? BODY_PARSER_ERRORS[err.type] : undefined;
  if (bodyParser) {
    const details =
      err.type === 'entity.too.large' ? { limit: err.limit ?? config.jsonBodyLimit, received: err.length ?? null } : null;
    return { ...bodyParser, details };
  }

  const status = Number.isInteger(err?.status) ? err.status : Number.isInteger(err?.statusCode) ? err.statusCode : 500;
  if (status >= 400 && status < 500) {
    return {
      status,
      code: STATUS_CODES[status] ?? 'bad_request',
      message: typeof err?.message === 'string' && err.message ? err.message : 'Request could not be processed',
      details: null,
    };
  }

  return { status: 500, code: 'internal_error', message: 'Internal server error', details: null };
}

/** Terminal Express error middleware — must keep all four parameters to be recognised. */
export function errorHandler(err, req, res, next) {
  const resolved = classify(err);

  if (resolved.status >= 500) {
    log.error('Unhandled request failure', {
      method: req.method,
      url: req.originalUrl,
      status: resolved.status,
      error: err instanceof Error ? err : new Error(String(err)),
    });
  } else {
    log.warn('Request rejected', {
      method: req.method,
      url: req.originalUrl,
      status: resolved.status,
      code: resolved.code,
      message: resolved.message,
    });
  }

  // The socket may already be half-written (e.g. an aborted upload); hand back to Express.
  if (res.headersSent) {
    next(err);
    return;
  }

  const details =
    resolved.status >= 500 && !config.isProduction && err instanceof Error
      ? { stack: err.stack, name: err.name, message: err.message }
      : resolved.details;

  res.status(resolved.status).json({
    error: { code: resolved.code, message: resolved.message, details: details ?? null },
  });
}

export default errorHandler;
