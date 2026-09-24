import fs from 'node:fs';
import path from 'node:path';
import cors from 'cors';
import express from 'express';
import helmet from 'helmet';
import { config } from './config.js';
import { createLogger } from './logger.js';
import { errorHandler } from './middleware/errorHandler.js';
import { notFoundHandler } from './middleware/notFound.js';
import { requestLogger } from './middleware/requestLogger.js';
import { apiRouter } from './routes/index.js';

const log = createLogger('app');

/** TFJS weight shards are extensionless, so `send` cannot infer a media type for them. */
const WEIGHT_SHARD_PATTERN = /group\d+-shard\d+of\d+/;

function modelStaticOptions() {
  return {
    fallthrough: true, // a miss must fall through to the next mount, then to the 404 handler
    index: false,
    etag: true,
    maxAge: 0, // models are replaced in place by `make fetch-models` / ml training runs
    dotfiles: 'ignore',
    setHeaders(res, filePath) {
      // Without CORP the browser blocks these responses when the page is served from :5173.
      res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
      if (WEIGHT_SHARD_PATTERN.test(path.basename(filePath))) {
        res.setHeader('Content-Type', 'application/octet-stream');
      }
    },
  };
}

function mountModelDirectories(app) {
  const mounted = [];
  const skipped = [];

  // MODELS_DIR first so a bind-mounted (freshly trained) model shadows the baked-in copy.
  for (const dir of [config.modelsDir, config.bundledModelsDir]) {
    if (mounted.includes(dir)) continue; // on the host both default to <repo>/models
    if (!fs.existsSync(dir)) {
      skipped.push(dir);
      continue;
    }
    app.use('/models', express.static(dir, modelStaticOptions()));
    mounted.push(dir);
  }

  log.info('Model static mounts resolved', { mounted, skipped });
  return mounted;
}

export function createApp() {
  const app = express();

  app.disable('x-powered-by');
  app.set('trust proxy', true); // compose/nginx sits in front in the prod profile

  app.use(
    helmet({
      // This process serves an API and binary model files, never HTML; a CSP would only
      // break the cross-origin model fetches the Vite dev server makes.
      contentSecurityPolicy: false,
      crossOriginEmbedderPolicy: false,
      crossOriginResourcePolicy: { policy: 'cross-origin' },
    }),
  );

  app.use(
    cors({
      origin: config.corsAllowAll ? '*' : [...config.corsOrigins],
      methods: ['GET', 'HEAD', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Accept'],
      credentials: false,
      maxAge: 600,
    }),
  );

  // Captured images arrive as base64 data URLs inside the JSON body.
  app.use(express.json({ limit: config.jsonBodyLimit }));
  app.use(requestLogger);

  const modelMounts = mountModelDirectories(app);

  app.get('/', (_req, res) => {
    res.json({
      service: 'ecosort-backend',
      version: config.version,
      description: 'Offline waste-image classification API for EcoSort.',
      endpoints: {
        health: '/api/health',
        modelStatus: '/api/model/status',
        categories: '/api/categories',
        rules: '/api/rules',
        ruleForRegion: '/api/rules/:regionId',
        ruleForCategory: '/api/rules/:regionId/:categoryId',
        classifications: '/api/classifications',
        classification: '/api/classifications/:id',
        stats: '/api/stats',
        models: '/models/<file>',
      },
      models: { mounted: modelMounts, custom: '/models/custom/model.json', fallback: '/models/mobilenet_v2/model.json' },
    });
  });

  app.use('/api', apiRouter);

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}

export default createApp;
