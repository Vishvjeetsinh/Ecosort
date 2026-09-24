import { Router } from 'express';
import { z } from 'zod';

import { asyncHandler } from '../middleware/asyncHandler.js';
import { notFound, validateParams } from '../middleware/validate.js';
import { isValidCategory } from '../services/categoryService.js';
import {
  defaultRegion,
  getGuidance,
  getRegionRules,
  isValidRegion,
  listRegions,
} from '../services/rulesService.js';

const idSchema = z.string().trim().min(1).max(64);

const regionParams = z.object({ regionId: idSchema });
const guidanceParams = z.object({ regionId: idSchema, categoryId: idSchema });

export const rulesRouter = Router();

// Bundled data that only changes with a new image build.
rulesRouter.use((_req, res, next) => {
  res.set('Cache-Control', 'public, max-age=300');
  next();
});

rulesRouter.get(
  '/',
  asyncHandler(async (_req, res) => {
    res.json({ defaultRegion, regions: listRegions() });
  }),
);

rulesRouter.get(
  '/:regionId',
  validateParams(regionParams),
  asyncHandler(async (req, res) => {
    const { regionId } = req.validatedParams;
    const payload = getRegionRules(regionId);
    if (payload === null) {
      throw notFound(`Unknown region "${regionId}"`, { regionId });
    }
    res.json(payload);
  }),
);

rulesRouter.get(
  '/:regionId/:categoryId',
  validateParams(guidanceParams),
  asyncHandler(async (req, res) => {
    const { regionId, categoryId } = req.validatedParams;
    if (!isValidRegion(regionId)) {
      throw notFound(`Unknown region "${regionId}"`, { regionId });
    }
    if (!isValidCategory(categoryId)) {
      throw notFound(`Unknown category "${categoryId}"`, { categoryId });
    }
    const payload = getGuidance(regionId, categoryId);
    if (payload === null) {
      throw notFound(`No guidance for "${categoryId}" in region "${regionId}"`, { regionId, categoryId });
    }
    res.json(payload);
  }),
);

export default rulesRouter;
