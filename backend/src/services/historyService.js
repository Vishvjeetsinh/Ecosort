/**
 * Classification history: every SQL statement in the application lives in this module.
 *
 * Statements are prepared lazily and keyed to the database handle they were compiled against, so
 * closing/reopening the database (tests, `make db-reset`) transparently recompiles them instead
 * of throwing "database connection is closed".
 */
import { getDb } from '../db.js';
import { createLogger } from '../logger.js';
import { badRequest } from '../middleware/validate.js';
import { isValidCategory } from './categoryService.js';

const logger = createLogger('history');

const DATE_ONLY = /^\d{4}-\d{2}-\d{2}$/;

const BASE_COLUMNS = [
  'id',
  'created_at',
  'top_category',
  'top_label',
  'top_confidence',
  'predictions_json',
  'raw_labels_json',
  'source',
  'model_kind',
  'region_id',
  'corrected_category',
  'notes',
  'duration_ms',
];

const IMAGE_COLUMN = 'image_data_url';
const ALL_COLUMNS = [...BASE_COLUMNS, IMAGE_COLUMN];

/**
 * created_at is always `new Date().toISOString()`, i.e. a fixed-width UTC string, so ISO strings
 * compare correctly with plain `<`/`>` and `substr(created_at, 1, 10)` is the UTC calendar day.
 */
const WINDOW_SQL = 'WHERE created_at >= @from AND created_at < @to';

let prepared = null;

function statements() {
  const db = getDb();
  if (prepared !== null && prepared.db === db) return prepared;

  prepared = {
    db,
    insert: db.prepare(
      `INSERT INTO classifications (
         created_at, top_category, top_label, top_confidence, predictions_json, raw_labels_json,
         source, model_kind, region_id, image_data_url, corrected_category, notes, duration_ms
       ) VALUES (
         @created_at, @top_category, @top_label, @top_confidence, @predictions_json, @raw_labels_json,
         @source, @model_kind, @region_id, @image_data_url, @corrected_category, @notes, @duration_ms
       )`,
    ),
    selectById: db.prepare(`SELECT ${ALL_COLUMNS.join(', ')} FROM classifications WHERE id = ?`),
    deleteById: db.prepare('DELETE FROM classifications WHERE id = ?'),
    deleteAll: db.prepare('DELETE FROM classifications'),
    countAll: db.prepare('SELECT COUNT(*) AS total FROM classifications'),
    lastCreatedAt: db.prepare('SELECT MAX(created_at) AS last FROM classifications'),
    windowTotals: db.prepare(
      `SELECT COUNT(*) AS total, AVG(top_confidence) AS avgConfidence FROM classifications ${WINDOW_SQL}`,
    ),
    windowByCategory: db.prepare(
      `SELECT COALESCE(corrected_category, top_category) AS category, COUNT(*) AS count
         FROM classifications ${WINDOW_SQL}
        GROUP BY category
        ORDER BY count DESC, category ASC`,
    ),
    windowBySource: db.prepare(
      `SELECT source, COUNT(*) AS count FROM classifications ${WINDOW_SQL}
        GROUP BY source ORDER BY count DESC, source ASC`,
    ),
    windowByModelKind: db.prepare(
      `SELECT model_kind AS modelKind, COUNT(*) AS count FROM classifications ${WINDOW_SQL}
        GROUP BY model_kind ORDER BY count DESC, modelKind ASC`,
    ),
    windowByDay: db.prepare(
      `SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count FROM classifications ${WINDOW_SQL}
        GROUP BY day ORDER BY day ASC`,
    ),
    windowTopLabels: db.prepare(
      `SELECT top_label AS label, COUNT(*) AS count FROM classifications ${WINDOW_SQL}
        GROUP BY top_label ORDER BY count DESC, label ASC LIMIT @limit`,
    ),
  };
  return prepared;
}

/** Test helper: forces the next call to recompile against the current database handle. */
export function resetPreparedStatements() {
  prepared = null;
}

function parseJsonColumn(value, fallback, rowId, column) {
  if (value === null || value === undefined) return fallback;
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) {
      logger.warn('Discarding a non-array JSON column', { id: rowId, column });
      return fallback;
    }
    return parsed;
  } catch (err) {
    // One corrupt row must not break the whole history listing.
    logger.warn('Discarding a corrupt JSON column', { id: rowId, column, message: err.message });
    return fallback;
  }
}

/** Maps a DB row onto the HistoryItem shape from ARCHITECTURE section 4. */
export function mapRow(row) {
  if (!row) return null;
  return {
    id: row.id,
    createdAt: row.created_at,
    topCategory: row.top_category,
    topLabel: row.top_label,
    topConfidence: row.top_confidence,
    predictions: parseJsonColumn(row.predictions_json, [], row.id, 'predictions_json'),
    rawLabels: parseJsonColumn(row.raw_labels_json, null, row.id, 'raw_labels_json'),
    source: row.source,
    modelKind: row.model_kind,
    regionId: row.region_id,
    imageDataUrl: row.image_data_url ?? null,
    correctedCategory: row.corrected_category ?? null,
    notes: row.notes ?? null,
    durationMs: row.duration_ms ?? null,
    effectiveCategory: row.corrected_category ?? row.top_category,
  };
}

function toIntegerOrNull(value) {
  if (value === null || value === undefined) return null;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? Math.round(numeric) : null;
}

function clampInt(value, { min, max, fallback }) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(max, Math.max(min, Math.trunc(numeric)));
}

/**
 * Widens a date boundary to a full UTC instant so a plain `2026-09-18` filter includes the whole
 * day, and normalises offset timestamps to Z so string comparison stays correct.
 */
function normaliseBoundary(value, { endOfDay }) {
  if (typeof value !== 'string' || value.trim() === '') return null;
  const trimmed = value.trim();
  if (DATE_ONLY.test(trimmed)) {
    return endOfDay ? `${trimmed}T23:59:59.999Z` : `${trimmed}T00:00:00.000Z`;
  }
  const parsed = Date.parse(trimmed);
  if (Number.isNaN(parsed)) {
    throw badRequest(`"${value}" is not an ISO-8601 date`, { field: endOfDay ? 'to' : 'from' });
  }
  return new Date(parsed).toISOString();
}

function buildWhere(filters) {
  const clauses = [];
  const params = {};

  if (typeof filters.category === 'string' && filters.category !== '') {
    // A human correction supersedes the model's answer everywhere the category is used.
    clauses.push('COALESCE(corrected_category, top_category) = @category');
    params.category = filters.category;
  }
  if (typeof filters.source === 'string' && filters.source !== '') {
    clauses.push('source = @source');
    params.source = filters.source;
  }
  if (typeof filters.modelKind === 'string' && filters.modelKind !== '') {
    clauses.push('model_kind = @modelKind');
    params.modelKind = filters.modelKind;
  }
  if (typeof filters.regionId === 'string' && filters.regionId !== '') {
    clauses.push('region_id = @regionId');
    params.regionId = filters.regionId;
  }

  const from = normaliseBoundary(filters.from, { endOfDay: false });
  if (from !== null) {
    clauses.push('created_at >= @from');
    params.from = from;
  }
  const to = normaliseBoundary(filters.to, { endOfDay: true });
  if (to !== null) {
    clauses.push('created_at <= @to');
    params.to = to;
  }

  return { sql: clauses.length === 0 ? '' : `WHERE ${clauses.join(' AND ')}`, params };
}

export function createClassification(input) {
  const predictions = Array.isArray(input?.predictions) ? input.predictions : [];
  if (predictions.length === 0) {
    throw badRequest('predictions must contain at least one entry', { field: 'predictions' });
  }

  // Never trust a client-supplied "top" triple: it is always predictions[0].
  const top = predictions[0];
  const row = {
    created_at: new Date().toISOString(),
    top_category: top.category,
    top_label: top.label,
    top_confidence: top.confidence,
    predictions_json: JSON.stringify(predictions),
    raw_labels_json: Array.isArray(input.rawLabels) ? JSON.stringify(input.rawLabels) : null,
    source: input.source,
    model_kind: input.modelKind,
    region_id: input.regionId,
    image_data_url: typeof input.imageDataUrl === 'string' ? input.imageDataUrl : null,
    corrected_category: null,
    notes: typeof input.notes === 'string' && input.notes !== '' ? input.notes : null,
    duration_ms: toIntegerOrNull(input.durationMs),
  };

  let info;
  try {
    info = statements().insert.run(row);
  } catch (err) {
    // CHECK-constraint violations mean a schema/validation mismatch, which is a client error.
    if (typeof err.code === 'string' && err.code.startsWith('SQLITE_CONSTRAINT')) {
      throw badRequest(`Classification rejected by the database: ${err.message}`);
    }
    throw err;
  }

  const item = getClassification(Number(info.lastInsertRowid));
  if (item === null) {
    throw new Error(`Inserted classification ${info.lastInsertRowid} could not be read back`);
  }
  return item;
}

export function listClassifications(filters = {}) {
  const limit = clampInt(filters.limit, { min: 1, max: 100, fallback: 25 });
  const offset = clampInt(filters.offset, { min: 0, max: Number.MAX_SAFE_INTEGER, fallback: 0 });
  const includeImage = filters.includeImage !== false;
  const columns = includeImage ? ALL_COLUMNS : BASE_COLUMNS;

  const { sql, params } = buildWhere(filters);
  const { db } = statements();

  const rows = db
    .prepare(
      `SELECT ${columns.join(', ')} FROM classifications ${sql}
        ORDER BY created_at DESC, id DESC
        LIMIT @limit OFFSET @offset`,
    )
    .all({ ...params, limit, offset });

  const countStatement = db.prepare(`SELECT COUNT(*) AS total FROM classifications ${sql}`);
  // better-sqlite3 rejects bound values for a statement that has no parameters.
  const { total } = Object.keys(params).length === 0 ? countStatement.get() : countStatement.get(params);

  return { items: rows.map(mapRow), total, limit, offset };
}

export function getClassification(id) {
  const numericId = Number(id);
  if (!Number.isInteger(numericId) || numericId <= 0) return null;
  return mapRow(statements().selectById.get(numericId));
}

export function updateClassification(id, patch = {}) {
  const existing = getClassification(id);
  if (existing === null) return null;

  const assignments = [];
  const params = { id: Number(id) };
  const has = (key) => Object.prototype.hasOwnProperty.call(patch, key) && patch[key] !== undefined;

  if (has('correctedCategory')) {
    const value = patch.correctedCategory;
    if (value !== null && !isValidCategory(value)) {
      throw badRequest(`Unknown category "${value}"`, { field: 'correctedCategory' });
    }
    assignments.push('corrected_category = @corrected_category');
    params.corrected_category = value ?? null; // an explicit null clears the correction
  }
  if (has('notes')) {
    const value = patch.notes;
    assignments.push('notes = @notes');
    params.notes = typeof value === 'string' && value !== '' ? value : null;
  }

  if (assignments.length === 0) return existing;

  statements()
    .db.prepare(`UPDATE classifications SET ${assignments.join(', ')} WHERE id = @id`)
    .run(params);

  return getClassification(id);
}

export function deleteClassification(id) {
  const numericId = Number(id);
  if (!Number.isInteger(numericId) || numericId <= 0) return false;
  return statements().deleteById.run(numericId).changes > 0;
}

export function deleteAllClassifications() {
  return statements().deleteAll.run().changes;
}

export function countClassifications() {
  return statements().countAll.get().total;
}

export function getLastClassifiedAt() {
  return statements().lastCreatedAt.get().last ?? null;
}

/**
 * Every aggregate the stats service needs, over the half-open window [from, to).
 * Grouping is by *effective* category so human corrections are reflected in the charts.
 */
export function getWindowAggregates({ from, to, topLabelLimit = 10 }) {
  if (typeof from !== 'string' || typeof to !== 'string') {
    throw new TypeError('getWindowAggregates requires ISO-8601 "from" and "to" strings');
  }
  const range = { from, to };
  const stmt = statements();
  const totals = stmt.windowTotals.get(range);

  return {
    totalInWindow: totals.total,
    // AVG() is null on an empty set.
    avgConfidence: totals.avgConfidence ?? 0,
    byCategory: stmt.windowByCategory.all(range),
    bySource: stmt.windowBySource.all(range),
    byModelKind: stmt.windowByModelKind.all(range),
    byDay: stmt.windowByDay.all(range),
    topLabels: stmt.windowTopLabels.all({ ...range, limit: clampInt(topLabelLimit, { min: 1, max: 50, fallback: 10 }) }),
  };
}
