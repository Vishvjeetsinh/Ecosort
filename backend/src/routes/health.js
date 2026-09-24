import { Router } from 'express';

import { config } from '../config.js';
import { createLogger } from '../logger.js';
import { asyncHandler } from '../middleware/asyncHandler.js';
import { countClassifications } from '../services/historyService.js';

const logger = createLogger('health');

export const healthRouter = Router();

healthRouter.get(
  '/',
  asyncHandler(async (_req, res) => {
    // A broken database is reported as db.ok=false with a 200, never a 500: the container
    // healthcheck has to tell "process up, storage degraded" apart from "process down".
    let db = { ok: true, path: config.dbFile, classifications: 0 };
    try {
      db.classifications = countClassifications();
    } catch (err) {
      logger.error('Health check could not read the database', { message: err.message });
      db = { ok: false, path: config.dbFile, classifications: 0 };
    }

    res.set('Cache-Control', 'no-store');
    res.json({
      status: 'ok',
      version: config.version,
      uptimeSeconds: Math.round(process.uptime()),
      timestamp: new Date().toISOString(),
      db,
    });
  }),
);

export default healthRouter;
