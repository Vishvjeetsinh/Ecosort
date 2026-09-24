/**
 * The waste taxonomy (docs/ARCHITECTURE.md section 3).
 *
 * The *content* of a category is data; the *set of ids* is contract. Rules, statistics, the DB
 * and the trained model all key off those ids, so a drifted data file is a startup failure here
 * rather than a confusing 500 three layers away.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { createLogger } from '../logger.js';

const logger = createLogger('categories');

const CANONICAL_CATEGORY_IDS = [
  'plastic',
  'paper',
  'cardboard',
  'glass',
  'metal',
  'organic',
  'ewaste',
  'hazardous',
  'textile',
  'trash',
];

const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
const REQUIRED_STRING_FIELDS = [
  'id',
  'label',
  'shortLabel',
  'description',
  'icon',
  'colorHex',
  'textColorHex',
];

// Loaded with fs rather than an import assertion: the assertion syntax changed twice across Node
// releases and this file has to keep loading unchanged on Node 22 and whatever comes next.
const DATA_FILE = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  'data',
  'waste-categories.json',
);

function invalid(message) {
  return new Error(`EcoSort waste categories (${DATA_FILE}): ${message}`);
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim() !== '';
}

function readCategories() {
  let text;
  try {
    text = fs.readFileSync(DATA_FILE, 'utf8');
  } catch (err) {
    throw invalid(`cannot be read - ${err.message}`);
  }

  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    throw invalid(`is not valid JSON - ${err.message}`);
  }

  if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.categories)) {
    throw invalid('must be an object with a "categories" array');
  }

  const problems = [];
  const seen = new Set();

  parsed.categories.forEach((entry, index) => {
    const where = `categories[${index}]`;
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      problems.push(`${where} must be an object`);
      return;
    }
    for (const field of REQUIRED_STRING_FIELDS) {
      if (!isNonEmptyString(entry[field])) {
        problems.push(`${where}.${field} must be a non-empty string`);
      }
    }
    for (const field of ['colorHex', 'textColorHex']) {
      if (typeof entry[field] === 'string' && !HEX_COLOR.test(entry[field])) {
        problems.push(`${where}.${field} must look like "#1a2b3c", got "${entry[field]}"`);
      }
    }
    if (!Array.isArray(entry.examples) || !entry.examples.every(isNonEmptyString)) {
      problems.push(`${where}.examples must be an array of non-empty strings`);
    }
    if (typeof entry.id === 'string') {
      if (seen.has(entry.id)) problems.push(`${where}.id "${entry.id}" is duplicated`);
      seen.add(entry.id);
    }
  });

  const missing = CANONICAL_CATEGORY_IDS.filter((id) => !seen.has(id));
  const unexpected = [...seen].filter((id) => !CANONICAL_CATEGORY_IDS.includes(id));
  if (missing.length > 0) {
    problems.push(`missing canonical categories: ${missing.join(', ')}`);
  }
  if (unexpected.length > 0) {
    problems.push(
      `contains categories outside the ARCHITECTURE section 3 taxonomy: ${unexpected.join(', ')}`,
    );
  }

  if (problems.length > 0) {
    throw invalid(problems.join('; '));
  }

  return Object.freeze(
    parsed.categories.map((entry) =>
      Object.freeze({
        id: entry.id,
        label: entry.label,
        shortLabel: entry.shortLabel,
        description: entry.description,
        icon: entry.icon,
        colorHex: entry.colorHex,
        textColorHex: entry.textColorHex,
        examples: Object.freeze([...entry.examples]),
      }),
    ),
  );
}

const categories = readCategories();
const byId = new Map(categories.map((category) => [category.id, category]));

logger.info(`Loaded ${categories.length} waste categories`);

/** The ten canonical category ids, in data-file order. */
export const categoryIds = Object.freeze(categories.map((category) => category.id));

export function listCategories() {
  return categories;
}

export function getCategory(id) {
  if (typeof id !== 'string') return null;
  return byId.get(id) ?? null;
}

export function isValidCategory(id) {
  return typeof id === 'string' && byId.has(id);
}
