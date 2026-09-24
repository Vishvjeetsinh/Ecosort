import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';
import { config } from './config.js';
import { createLogger } from './logger.js';

const log = createLogger('db');

/**
 * Ordered migration list. To add schema changes later, append a new entry with the next
 * version number — nothing else in this file needs to change, and previously applied
 * versions are skipped by the runner.
 */
export const MIGRATIONS = Object.freeze([
  Object.freeze({
    version: 1,
    name: 'create_classifications',
    up(db) {
      db.exec(`
        CREATE TABLE IF NOT EXISTS classifications (
          id                 INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at         TEXT    NOT NULL,
          top_category       TEXT    NOT NULL,
          top_label          TEXT    NOT NULL,
          top_confidence     REAL    NOT NULL,
          predictions_json   TEXT    NOT NULL,
          raw_labels_json    TEXT,
          source             TEXT    NOT NULL CHECK (source IN ('webcam','upload')),
          model_kind         TEXT    NOT NULL CHECK (model_kind IN ('custom','fallback')),
          region_id          TEXT    NOT NULL,
          image_data_url     TEXT,
          corrected_category TEXT,
          notes              TEXT,
          duration_ms        INTEGER
        );

        CREATE INDEX IF NOT EXISTS idx_classifications_created_at   ON classifications(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_classifications_top_category ON classifications(top_category);
        CREATE INDEX IF NOT EXISTS idx_classifications_region       ON classifications(region_id);
      `);
    },
  }),
]);

let instance = null;

function applyPragmas(db) {
  // WAL is meaningless for an in-memory database and SQLite silently keeps "memory" mode.
  if (!config.isMemoryDb) {
    const mode = db.pragma('journal_mode = WAL', { simple: true });
    if (String(mode).toLowerCase() !== 'wal') {
      // Common on some bind-mounted / network filesystems — journalling still works.
      log.warn('WAL journal mode unavailable, continuing with fallback', { mode });
    }
    db.pragma('synchronous = NORMAL');
  }
  db.pragma('foreign_keys = ON');
  db.pragma('busy_timeout = 5000');
}

function ensureMigrationsTable(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS schema_migrations (
      version     INTEGER PRIMARY KEY,
      name        TEXT NOT NULL,
      applied_at  TEXT NOT NULL
    );
  `);
}

/** Idempotent: safe to call on every boot and from tests. Returns the versions applied now. */
export function runMigrations(db) {
  if (!db || typeof db.prepare !== 'function') {
    throw new TypeError('runMigrations(db) requires an open better-sqlite3 database');
  }

  ensureMigrationsTable(db);

  const applied = new Set(
    db.prepare('SELECT version FROM schema_migrations').all().map((row) => row.version),
  );
  const record = db.prepare(
    'INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)',
  );
  const pending = [...MIGRATIONS].sort((a, b) => a.version - b.version).filter((m) => !applied.has(m.version));
  const executed = [];

  for (const migration of pending) {
    const apply = db.transaction(() => {
      migration.up(db);
      record.run(migration.version, migration.name, new Date().toISOString());
    });
    try {
      apply();
    } catch (err) {
      // A half-applied schema is unrecoverable at runtime; fail loudly at boot instead.
      throw new Error(`Migration ${migration.version} (${migration.name}) failed: ${err.message}`, { cause: err });
    }
    executed.push(migration.version);
    log.info('Applied migration', { version: migration.version, name: migration.name });
  }

  if (executed.length === 0) log.debug('Schema already up to date', { version: MIGRATIONS.at(-1)?.version ?? 0 });
  return executed;
}

function openDatabase() {
  if (!config.isMemoryDb) {
    // DATA_DIR is created by config.js, but DB_FILE may point somewhere else entirely.
    fs.mkdirSync(path.dirname(config.dbFile), { recursive: true });
  }

  const options = config.logLevel === 'debug' ? { verbose: (sql) => log.debug('sql', { sql }) } : {};

  let db;
  try {
    db = new Database(config.dbFile, options);
  } catch (err) {
    throw new Error(`Cannot open SQLite database at "${config.dbFile}": ${err.message}`, { cause: err });
  }

  applyPragmas(db);
  runMigrations(db);
  log.info('Database ready', { file: config.dbFile, memory: config.isMemoryDb });
  return db;
}

/** Lazy singleton so importing a route module never touches the filesystem. */
export function getDb() {
  if (instance === null || !instance.open) {
    instance = openDatabase();
  }
  return instance;
}

export function closeDb() {
  if (instance !== null) {
    if (instance.open) {
      if (!config.isMemoryDb) {
        // Fold the WAL back into the main file so the volume holds one consistent db.
        try {
          instance.pragma('wal_checkpoint(TRUNCATE)');
        } catch (err) {
          log.warn('WAL checkpoint on shutdown failed', { message: err.message });
        }
      }
      instance.close();
    }
    instance = null;
    log.debug('Database closed');
  }
}

/** Test helper: empties every data table and restarts AUTOINCREMENT ids from 1. */
export function resetDb() {
  const db = getDb();
  const reset = db.transaction(() => {
    db.exec('DELETE FROM classifications');
    // sqlite_sequence only exists once an AUTOINCREMENT table has been written to.
    const hasSequence = db
      .prepare("SELECT 1 AS present FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_sequence'")
      .get();
    if (hasSequence) db.prepare("DELETE FROM sqlite_sequence WHERE name = 'classifications'").run();
  });
  reset();
  log.debug('Database reset');
  return db;
}

export default getDb;
