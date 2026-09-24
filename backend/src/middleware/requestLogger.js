import morgan from 'morgan';
import { config } from '../config.js';
import { createLogger } from '../logger.js';

const log = createLogger('http');

const HUMAN_FORMAT = ':method :url :status :res[content-length]B - :response-time ms';

function toStructuredMeta(tokens, req, res) {
  return {
    method: tokens.method(req, res),
    url: tokens.url(req, res),
    status: Number(tokens.status(req, res)) || 0,
    lengthBytes: Number(tokens.res(req, res, 'content-length')) || 0,
    durationMs: Number(tokens['response-time'](req, res)) || 0,
    remoteAddr: tokens['remote-addr'](req, res),
    userAgent: tokens['user-agent'](req, res),
  };
}

function build() {
  if (config.isProduction) {
    // Hand morgan's fields to the logger as metadata so production output stays one JSON
    // object per line instead of a JSON object wrapped in a text log line.
    return morgan((tokens, req, res) => JSON.stringify(toStructuredMeta(tokens, req, res)), {
      stream: {
        write: (line) => {
          try {
            log.info('request', JSON.parse(line));
          } catch (err) {
            log.info('request', { raw: line.trim(), parseError: err.message });
          }
        },
      },
    });
  }

  return morgan(HUMAN_FORMAT, {
    stream: { write: (line) => log.info(line.trim()) },
  });
}

/**
 * Silent under NODE_ENV=test so `node --test` output stays readable; a no-op middleware is
 * cheaper and clearer than morgan's skip option.
 */
export const requestLogger = config.isTest
  ? function noRequestLogging(_req, _res, next) {
      next();
    }
  : build();

export default requestLogger;
