import { Router } from 'express';
import { z } from 'zod';

import { config } from '../config.js';
import { createLogger } from '../logger.js';
import { asyncHandler } from '../middleware/asyncHandler.js';
import { badRequest, notFound, payloadTooLarge, validateBody, validateParams, validateQuery } from '../middleware/validate.js';
import { categoryIds } from '../services/categoryService.js';
import {
  createClassification,
  deleteAllClassifications,
  deleteClassification,
  getClassification,
  listClassifications,
  updateClassification,
} from '../services/historyService.js';
import { isValidRegion } from '../services/rulesService.js';

const logger = createLogger('classifications');

const CATEGORY_VALUES = [...categoryIds];
const SOURCES = ['webcam', 'upload'];
const MODEL_KINDS = ['custom', 'fallback'];

// Single character class, anchored: linear-time even on a 400 000-character payload.
const IMAGE_DATA_URL = /^data:image\/(png|jpeg|jpg|webp);base64,[A-Za-z0-9+/=]+$/;

const TRUTHY = ['true', '1', 'yes', 'on'];
const FALSY = ['false', '0', 'no', 'off'];

/** An empty query value means "not set"; without this `?category=` would be a 400. */
const optional = (schema) => z.preprocess((value) => (value === '' ? undefined : value), schema);

const booleanFlag = (fallback) =>
  z.preprocess((value) => {
    if (value === undefined || value === '') return fallback;
    if (typeof value === 'boolean') return value;
    if (typeof value === 'string') {
      const normalised = value.trim().toLowerCase();
      if (TRUTHY.includes(normalised)) return true;
      if (FALSY.includes(normalised)) return false;
    }
    return value; // anything else falls through to z.boolean() and becomes a validation error
  }, z.boolean());

const isoDate = z.string().trim().refine(
  (value) => /^\d{4}-\d{2}-\d{2}$/.test(value) || !Number.isNaN(Date.parse(value)),
  { message: 'must be an ISO-8601 date such as 2026-09-18 or 2026-09-18T12:00:00.000Z' },
);

const predictionSchema = z.object({
  category: z.enum(CATEGORY_VALUES),
  label: z.string().trim().min(1).max(200),
  confidence: z.number().finite().min(0).max(1),
});

const rawLabelSchema = z.object({
  label: z.string().trim().min(1).max(200),
  confidence: z.number().finite().min(0).max(1),
  index: z.number().int().min(0).max(100000),
});

// The contract's POST body carries no top* fields — they are derived from predictions[0].
const createBody = z.object({
  predictions: z.array(predictionSchema).min(1).max(10),
  source: z.enum(SOURCES),
  modelKind: z.enum(MODEL_KINDS),
  regionId: z
    .string()
    .trim()
    .min(1)
    .max(64)
    .refine(isValidRegion, { message: 'regionId must be a known region (see GET /api/rules)' }),
  rawLabels: z.array(rawLabelSchema).max(100).nullish(),
  imageDataUrl: z
    .string()
    .regex(IMAGE_DATA_URL, 'imageDataUrl must be a base64 data URL of a PNG, JPEG or WEBP image')
    .nullish(),
  notes: z.string().trim().max(500).nullish(),
  durationMs: z.number().finite().min(0).max(3600000).nullish(),
});

const patchBody = z
  .object({
    correctedCategory: z.enum(CATEGORY_VALUES).nullable().optional(),
    notes: z.string().trim().max(500).nullable().optional(),
  })
  .refine((value) => value.correctedCategory !== undefined || value.notes !== undefined, {
    message: 'provide at least one of "correctedCategory" or "notes"',
  });

const listQuery = z.object({
  limit: optional(z.coerce.number().int().min(1).max(100).default(25)),
  offset: optional(z.coerce.number().int().min(0).default(0)),
  category: optional(z.enum(CATEGORY_VALUES).optional()),
  source: optional(z.enum(SOURCES).optional()),
  modelKind: optional(z.enum(MODEL_KINDS).optional()),
  regionId: optional(z.string().trim().min(1).max(64).optional()),
  from: optional(isoDate.optional()),
  to: optional(isoDate.optional()),
  includeImage: booleanFlag(true),
});

// Numeric-only so /api/classifications/abc is a clean 400 instead of a NaN lookup.
const idParams = z.object({ id: z.coerce.number().int().positive() });

/**
 * Runs before body validation so an oversized capture is always a 413, even when the rest of the
 * body is also malformed — and so the regex never walks a multi-megabyte string.
 */
function enforceImageBudget(req, _res, next) {
  const value = req.body?.imageDataUrl;
  if (typeof value === 'string' && value.length > config.maxImageDataUrl) {
    next(
      payloadTooLarge(
        `imageDataUrl is ${value.length} characters; the limit is ${config.maxImageDataUrl}. Capture or re-encode the image at a smaller size.`,
        { limit: config.maxImageDataUrl, received: value.length },
      ),
    );
    return;
  }
  next();
}

export const classificationsRouter = Router();

classificationsRouter.post(
  '/',
  enforceImageBudget,
  validateBody(createBody),
  asyncHandler(async (req, res) => {
    const item = createClassification(req.body);
    res.status(201).json({ item });
  }),
);

classificationsRouter.get(
  '/',
  validateQuery(listQuery),
  asyncHandler(async (req, res) => {
    const filters = req.validatedQuery;
    const { items, total, limit, offset } = listClassifications(filters);
    res.set('Cache-Control', 'no-store');
    res.json({ items, total, limit, offset });
  }),
);

classificationsRouter.delete(
  '/',
  asyncHandler(async (req, res) => {
    const confirm = String(req.query?.confirm ?? '').trim().toLowerCase();
    if (!TRUTHY.includes(confirm)) {
      throw badRequest(
        'Refusing to erase the whole history: repeat the request with the "confirm=true" query parameter.',
        { required: 'confirm=true' },
      );
    }
    const deleted = deleteAllClassifications();
    logger.info('History cleared', { deleted });
    res.status(204).end();
  }),
);

classificationsRouter.get(
  '/:id',
  validateParams(idParams),
  asyncHandler(async (req, res) => {
    const { id } = req.validatedParams;
    const item = getClassification(id);
    if (item === null) {
      throw notFound(`No classification with id ${id}`, { id });
    }
    res.set('Cache-Control', 'no-store');
    res.json({ item });
  }),
);

classificationsRouter.patch(
  '/:id',
  validateParams(idParams),
  validateBody(patchBody),
  asyncHandler(async (req, res) => {
    const { id } = req.validatedParams;
    const item = updateClassification(id, req.body);
    if (item === null) {
      throw notFound(`No classification with id ${id}`, { id });
    }
    res.json({ item });
  }),
);

classificationsRouter.delete(
  '/:id',
  validateParams(idParams),
  asyncHandler(async (req, res) => {
    const { id } = req.validatedParams;
    if (!deleteClassification(id)) {
      throw notFound(`No classification with id ${id}`, { id });
    }
    res.status(204).end();
  }),
);

export default classificationsRouter;
