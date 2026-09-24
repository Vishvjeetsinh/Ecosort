import { Router } from 'express';
import { z } from 'zod';

import { asyncHandler } from '../middleware/asyncHandler.js';
import { notFound, validateParams } from '../middleware/validate.js';
import { getModel, getModelStatus } from '../services/modelService.js';

/**
 * A model id is a directory name under the models root, so it is also a path segment: anything
 * outside this alphabet is rejected before it can reach the filesystem. `..%2Fetc` therefore ends
 * as a 400 from the validator rather than as a probe of the container's filesystem.
 */
const modelParams = z.object({
  id: z
    .string()
    .regex(/^[A-Za-z0-9_.-]{1,64}$/, 'A model id may only contain letters, digits, "_", "." and "-"'),
});

export const modelRouter = Router();

modelRouter.get(
  '/status',
  asyncHandler(async (_req, res) => {
    // Never cached: this flips the moment the user finishes training a model into MODELS_DIR.
    res.set('Cache-Control', 'no-store');
    res.json(getModelStatus());
  }),
);

/**
 * The picker's endpoints (ARCHITECTURE section 2.4). Same data as /api/model/status, minus the
 * two-slot compatibility blocks and the probed paths — the picker re-fetches this on every open,
 * and a ~100 MB model appearing mid-session must show up straight away.
 */
export const modelsRouter = Router();

modelsRouter.use((_req, res, next) => {
  res.set('Cache-Control', 'no-store');
  next();
});

modelsRouter.get(
  '/',
  asyncHandler(async (_req, res) => {
    const { models, defaultModelId, active } = getModelStatus();
    res.json({ models, defaultModelId, active });
  }),
);

modelsRouter.get(
  '/:id',
  validateParams(modelParams),
  asyncHandler(async (req, res) => {
    const { id } = req.validatedParams;
    const model = getModel(id);
    if (model === null) {
      throw notFound(`Unknown model "${id}"`, { id });
    }
    res.json({ model });
  }),
);

export default modelRouter;
