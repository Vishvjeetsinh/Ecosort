import { Router } from 'express';

import categoriesRouter from './categories.js';
import classificationsRouter from './classifications.js';
import healthRouter from './health.js';
import modelRouter, { modelsRouter } from './model.js';
import rulesRouter from './rules.js';
import statsRouter from './stats.js';

/** Mounted at /api by createApp(); paths here are relative to that prefix. */
export const apiRouter = Router();

apiRouter.use('/health', healthRouter);
apiRouter.use('/model', modelRouter);
apiRouter.use('/models', modelsRouter);
apiRouter.use('/categories', categoriesRouter);
apiRouter.use('/rules', rulesRouter);
apiRouter.use('/classifications', classificationsRouter);
apiRouter.use('/stats', statsRouter);

export default apiRouter;
