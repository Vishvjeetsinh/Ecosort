/**
 * The only module in the frontend that talks to the backend.
 *
 * Every function returns parsed JSON and throws `ApiError` on failure, so callers
 * never have to inspect a Response. Empty when same-origin (the Vite dev server
 * proxies /api to the backend); set VITE_API_BASE only for an unusual deployment.
 */
// `import.meta.env` only exists under a Vite/Vitest transform; guard it so this
// module can also be imported by plain Node tooling without throwing.
const BASE = (import.meta.env && import.meta.env.VITE_API_BASE) || '';

const NETWORK_MESSAGE =
  'Cannot reach the EcoSort backend. It may not be running — start it with `docker compose up --build`, or check that port 4000 is reachable.';

export class ApiError extends Error {
  constructor(message, { status = 0, code = 'unknown_error', details = null } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.details = details;
  }

  /** True when the request never reached the server (backend down, DNS, offline). */
  get isNetworkError() {
    return this.status === 0;
  }
}

function buildQuery(query) {
  if (!query) return '';
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === '') continue;
    if (typeof value === 'boolean') {
      params.set(key, value ? 'true' : 'false');
      continue;
    }
    params.set(key, String(value));
  }
  const qs = params.toString();
  return qs ? `?${qs}` : '';
}

function isAbort(err) {
  return Boolean(err) && (err.name === 'AbortError' || err.code === 20);
}

/**
 * @param {string} path  API path beginning with `/`.
 * @param {{method?:string, body?:unknown, signal?:AbortSignal, query?:object}} [options]
 */
async function request(path, { method = 'GET', body, signal, query } = {}) {
  const url = `${BASE}${path}${buildQuery(query)}`;
  const init = {
    method,
    signal,
    headers: { Accept: 'application/json' },
  };

  if (body !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(body);
  }

  let response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    // Aborts are cooperative cancellation, not failures — let callers detect them.
    if (isAbort(cause)) throw cause;
    throw new ApiError(NETWORK_MESSAGE, {
      status: 0,
      code: 'network_error',
      details: cause instanceof Error ? cause.message : String(cause),
    });
  }

  if (response.status === 204) return null;

  let text = '';
  try {
    text = await response.text();
  } catch (cause) {
    if (isAbort(cause)) throw cause;
    throw new ApiError('The connection dropped while reading the response.', {
      status: response.status,
      code: 'network_error',
      details: cause instanceof Error ? cause.message : String(cause),
    });
  }

  let payload = null;
  let parseFailed = false;
  if (text.length > 0) {
    try {
      payload = JSON.parse(text);
    } catch {
      parseFailed = true;
    }
  }

  if (!response.ok) {
    const envelope = payload && typeof payload === 'object' ? payload.error : null;
    throw new ApiError(
      (envelope && envelope.message) || `The backend returned HTTP ${response.status}.`,
      {
        status: response.status,
        code: (envelope && envelope.code) || `http_${response.status}`,
        details: envelope ? (envelope.details ?? null) : text.slice(0, 500) || null,
      },
    );
  }

  if (parseFailed) {
    throw new ApiError('The backend returned a response that is not valid JSON.', {
      status: response.status,
      code: 'bad_response',
      details: text.slice(0, 200),
    });
  }

  return payload;
}

/* ------------------------------------------------------------------ health */

export function getHealth(signal) {
  return request('/api/health', { signal });
}

export function getModelStatus(signal) {
  return request('/api/model/status', { signal });
}

/* ------------------------------------------------------------------ models */

/** The model registry (ARCHITECTURE 2.4): `{ models, defaultModelId, active }`. */
export function listModels(signal) {
  return request('/api/models', { signal });
}

/** One registry entry by id: `{ model }`. 404 when the id is not installed. */
export function getModel(id, signal) {
  if (!id) {
    return Promise.reject(
      new ApiError('A model id is required to look one up.', {
        status: 0,
        code: 'bad_request',
      }),
    );
  }
  return request(`/api/models/${encodeURIComponent(id)}`, { signal });
}

/* -------------------------------------------------- categories & rules */

export function getCategories(signal) {
  return request('/api/categories', { signal });
}

/** The region catalogue: `{ defaultRegion, regions }`. */
export function getRules(signal) {
  return request('/api/rules', { signal });
}

/** Full rule set for one region: `{ region, bins, categories }`. */
export function getRegionRules(regionId, signal) {
  if (!regionId) {
    return Promise.reject(
      new ApiError('A region must be selected before its rules can be loaded.', {
        status: 0,
        code: 'bad_request',
      }),
    );
  }
  return request(`/api/rules/${encodeURIComponent(regionId)}`, { signal });
}

/** Guidance for one category in one region: `{ region, category, guidance, bin }`. */
export function getGuidance(regionId, categoryId, signal) {
  if (!regionId || !categoryId) {
    return Promise.reject(
      new ApiError('Both a region and a category are required to load guidance.', {
        status: 0,
        code: 'bad_request',
      }),
    );
  }
  return request(
    `/api/rules/${encodeURIComponent(regionId)}/${encodeURIComponent(categoryId)}`,
    { signal },
  );
}

/* -------------------------------------------------------- classifications */

export function createClassification(body, signal) {
  return request('/api/classifications', { method: 'POST', body, signal });
}

/**
 * @param {{limit?:number, offset?:number, category?:string, source?:string,
 *          modelKind?:string, regionId?:string, from?:string, to?:string,
 *          includeImage?:boolean}} [filters]
 */
export function listClassifications(filters = {}, signal) {
  return request('/api/classifications', { query: filters, signal });
}

export function getClassification(id, signal) {
  return request(`/api/classifications/${encodeURIComponent(id)}`, { signal });
}

/** @param {{correctedCategory?:string|null, notes?:string|null}} patch */
export function updateClassification(id, patch, signal) {
  return request(`/api/classifications/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: patch,
    signal,
  });
}

export function deleteClassification(id, signal) {
  return request(`/api/classifications/${encodeURIComponent(id)}`, {
    method: 'DELETE',
    signal,
  });
}

/** Deletes every stored classification. The backend demands `confirm=true`. */
export function clearClassifications(signal) {
  return request('/api/classifications', {
    method: 'DELETE',
    query: { confirm: true },
    signal,
  });
}

/* ------------------------------------------------------------------ stats */

export function getStats({ regionId, days } = {}, signal) {
  return request('/api/stats', { query: { regionId, days }, signal });
}

export { BASE as API_BASE };
