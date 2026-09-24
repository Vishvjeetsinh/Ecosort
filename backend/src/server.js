import http from 'node:http';
import { createApp } from './app.js';
import { config, describeConfig } from './config.js';
import { closeDb, getDb, runMigrations } from './db.js';
import { createLogger } from './logger.js';

const log = createLogger('server');

const FORCE_EXIT_MS = 10_000;

let httpServer = null;
let shuttingDown = false;

function shutdown(reason, exitCode = 0) {
  if (shuttingDown) return;
  shuttingDown = true;
  log.info('Shutting down', { reason });

  // Never let a wedged socket keep the container alive forever.
  const forceTimer = setTimeout(() => {
    log.error('Graceful shutdown timed out, forcing exit', { afterMs: FORCE_EXIT_MS });
    process.exit(exitCode === 0 ? 1 : exitCode);
  }, FORCE_EXIT_MS);
  forceTimer.unref();

  const finish = () => {
    try {
      closeDb();
    } catch (err) {
      log.error('Failed to close the database cleanly', err);
      clearTimeout(forceTimer);
      process.exit(1);
      return;
    }
    clearTimeout(forceTimer);
    log.info('Shutdown complete');
    process.exit(exitCode);
  };

  if (httpServer === null) {
    finish();
    return;
  }

  httpServer.close((err) => {
    if (err) log.error('HTTP server did not close cleanly', err);
    finish();
  });

  // Keep-alive connections would otherwise hold server.close() open for their full idle life.
  if (typeof httpServer.closeIdleConnections === 'function') httpServer.closeIdleConnections();
}

function installProcessHandlers() {
  process.on('SIGINT', () => shutdown('SIGINT'));
  process.on('SIGTERM', () => shutdown('SIGTERM'));

  process.on('unhandledRejection', (reason) => {
    log.error('Unhandled promise rejection', reason instanceof Error ? reason : new Error(String(reason)));
    shutdown('unhandledRejection', 1);
  });

  process.on('uncaughtException', (err) => {
    log.error('Uncaught exception', err);
    shutdown('uncaughtException', 1);
  });
}

function start() {
  installProcessHandlers();

  log.info('Starting EcoSort backend', describeConfig());

  // Fail fast at boot rather than on the first request if the volume is unwritable.
  runMigrations(getDb());

  const app = createApp();
  httpServer = http.createServer(app);

  httpServer.on('error', (err) => {
    if (err.code === 'EADDRINUSE') {
      log.error(`Port ${config.backendPort} is already in use. Stop the other process or set BACKEND_PORT.`, {
        port: config.backendPort,
      });
    } else {
      log.error('HTTP server error', err);
    }
    shutdown('server-error', 1);
  });

  httpServer.listen(config.backendPort, '0.0.0.0', () => {
    log.info(`EcoSort API listening on http://0.0.0.0:${config.backendPort}`, {
      env: config.nodeEnv,
      version: config.version,
      db: config.dbFile,
    });
  });
}

try {
  start();
} catch (err) {
  log.error('Backend failed to start', err instanceof Error ? err : new Error(String(err)));
  try {
    closeDb();
  } catch (closeErr) {
    log.error('Database close during failed startup also failed', closeErr);
  }
  process.exit(1);
}
