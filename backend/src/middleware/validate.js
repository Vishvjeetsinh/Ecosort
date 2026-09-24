/**
 * Request validation built on zod, plus the HttpError type every layer throws.
 *
 * IMPORTANT for route authors: `validateQuery` writes the parsed value to
 * `req.validatedQuery`, NOT to `req.query`. In Express 4.21 `req.query` is a lazy getter
 * defined on the request prototype, so assigning to it either throws in strict mode or is
 * silently dropped. Routes MUST read `req.validatedQuery`.
 */

export class HttpError extends Error {
  constructor(status, code, message, details = null) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.code = code;
    this.details = details ?? null;
    this.expose = status < 500;
    if (Error.captureStackTrace) Error.captureStackTrace(this, HttpError);
  }
}

export const badRequest = (message = 'Invalid request', details = null) =>
  new HttpError(400, 'bad_request', message, details);

export const notFound = (message = 'Resource not found', details = null) =>
  new HttpError(404, 'not_found', message, details);

export const payloadTooLarge = (message = 'Payload too large', details = null) =>
  new HttpError(413, 'payload_too_large', message, details);

export function isZodError(err) {
  return Boolean(err) && (err.name === 'ZodError' || Array.isArray(err.issues));
}

function formatPath(segments) {
  if (!Array.isArray(segments) || segments.length === 0) return '';
  return segments.reduce((acc, segment) => {
    if (typeof segment === 'number') return `${acc}[${segment}]`;
    return acc === '' ? String(segment) : `${acc}.${segment}`;
  }, '');
}

/** Flattens a ZodError into a transport-friendly list for the error envelope's `details`. */
export function formatZodIssues(error) {
  const issues = Array.isArray(error?.issues) ? error.issues : [];
  return issues.map((issue) => ({
    path: formatPath(issue.path),
    code: issue.code ?? 'invalid',
    message: issue.message ?? 'Invalid value',
  }));
}

function assertSchema(schema, factoryName) {
  if (!schema || typeof schema.safeParse !== 'function') {
    throw new TypeError(`${factoryName}(schema) requires a zod schema`);
  }
}

function runValidator(schema, value, label) {
  const result = schema.safeParse(value);
  if (!result.success) {
    throw badRequest(`Invalid request ${label}`, formatZodIssues(result.error));
  }
  return result.data;
}

export function validateBody(schema) {
  assertSchema(schema, 'validateBody');
  return function validateBodyMiddleware(req, _res, next) {
    req.body = runValidator(schema, req.body ?? {}, 'body');
    next();
  };
}

export function validateQuery(schema) {
  assertSchema(schema, 'validateQuery');
  return function validateQueryMiddleware(req, _res, next) {
    // See the module header: req.query is not writable, so the parsed value lands here.
    req.validatedQuery = runValidator(schema, req.query ?? {}, 'query');
    next();
  };
}

export function validateParams(schema) {
  assertSchema(schema, 'validateParams');
  return function validateParamsMiddleware(req, _res, next) {
    const parsed = runValidator(schema, req.params ?? {}, 'parameters');
    req.params = parsed;
    req.validatedParams = parsed; // stable alias for handlers mounted deeper in a router
    next();
  };
}
