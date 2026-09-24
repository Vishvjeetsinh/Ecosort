import { Router } from 'express';

import { asyncHandler } from '../middleware/asyncHandler.js';
import { listCategories } from '../services/categoryService.js';

export const categoriesRouter = Router();

categoriesRouter.get(
  '/',
  asyncHandler(async (_req, res) => {
    // Bundled data that only changes with a new image build.
    res.set('Cache-Control', 'public, max-age=300');
    res.json({ categories: listCategories() });
  }),
);

export default categoriesRouter;
