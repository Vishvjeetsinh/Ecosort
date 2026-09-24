import { Router } from 'express';
import { z } from 'zod';

import { asyncHandler } from '../middleware/asyncHandler.js';
import { validateQuery } from '../middleware/validate.js';
import { isValidRegion } from '../services/rulesService.js';
import { getStats } from '../services/statsService.js';

/** Compose and hand-written links send `?regionId=` for "unset"; treat that as absent. */
const optional = (schema) => z.preprocess((value) => (value === '' ? undefined : value), schema);

const statsQuery = z.object({
  regionId: optional(
    z
      .string()
      .trim()
      .min(1)
      .max(64)
      .refine(isValidRegion, { message: 'regionId must be a known region (see GET /api/rules)' })
      .optional(),
  ),
  days: optional(z.coerce.number().int().min(1).max(365).default(30)),
});

export const statsRouter = Router();

statsRouter.get(
  '/',
  validateQuery(statsQuery),
  asyncHandler(async (req, res) => {
    const { regionId, days } = req.validatedQuery;
    res.set('Cache-Control', 'no-store');
    res.json(getStats({ regionId, days }));
  }),
);

export default statsRouter;
