/**
 * Regional recycling rules (docs/ARCHITECTURE.md sections 3 and 4).
 *
 * Everything is validated and normalised once at load: the frontend renders bin colours straight
 * from this data, and a guidance entry pointing at a bin that does not exist would surface as a
 * blank card instead of an error. Better to refuse to boot.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { config } from '../config.js';
import { createLogger } from '../logger.js';
import { categoryIds, getCategory, isValidCategory } from './categoryService.js';

const logger = createLogger('rules');

const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
const REQUIRED_REGION_FIELDS = ['name', 'country', 'authority', 'updated'];
const REQUIRED_BIN_FIELDS = ['name', 'colorName', 'colorHex', 'textColorHex', 'description'];
const STRING_LIST_FIELDS = ['prepSteps', 'acceptedExamples', 'rejectedExamples'];

// See categoryService.js for why this is fs.readFileSync and not an import assertion.
const DATA_FILE = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  'data',
  'recycling-rules.json',
);

function invalid(message) {
  return new Error(`EcoSort recycling rules (${DATA_FILE}): ${message}`);
}

function isNonEmptyString(value) {
  return typeof value === 'string' && value.trim() !== '';
}

function stringOr(value, fallback) {
  return isNonEmptyString(value) ? value : fallback;
}

function stringList(value) {
  return Object.freeze(Array.isArray(value) ? value.filter(isNonEmptyString).map(String) : []);
}

function parseBins(region, where, problems) {
  const bins = new Map();
  region.bins.forEach((bin, index) => {
    const at = `${where}.bins[${index}]`;
    if (!bin || typeof bin !== 'object' || Array.isArray(bin)) {
      problems.push(`${at} must be an object`);
      return;
    }
    if (!isNonEmptyString(bin.id)) {
      problems.push(`${at}.id must be a non-empty string`);
      return;
    }
    if (bins.has(bin.id)) {
      problems.push(`${at}.id "${bin.id}" is duplicated within the region`);
      return;
    }
    for (const field of REQUIRED_BIN_FIELDS) {
      if (!isNonEmptyString(bin[field])) problems.push(`${at}.${field} must be a non-empty string`);
    }
    for (const field of ['colorHex', 'textColorHex']) {
      if (typeof bin[field] === 'string' && !HEX_COLOR.test(bin[field])) {
        problems.push(`${at}.${field} must look like "#1a2b3c", got "${bin[field]}"`);
      }
    }
    if (!Array.isArray(bin.accepts)) {
      problems.push(`${at}.accepts must be an array of category ids`);
    } else {
      for (const categoryId of bin.accepts) {
        if (!isValidCategory(categoryId)) {
          problems.push(`${at}.accepts contains unknown category "${categoryId}"`);
        }
      }
    }

    bins.set(
      bin.id,
      Object.freeze({
        id: bin.id,
        name: stringOr(bin.name, bin.id),
        colorName: stringOr(bin.colorName, 'unspecified'),
        colorHex: stringOr(bin.colorHex, '#64748b'),
        textColorHex: stringOr(bin.textColorHex, '#ffffff'),
        accepts: Object.freeze(Array.isArray(bin.accepts) ? bin.accepts.filter(isValidCategory) : []),
        description: stringOr(bin.description, ''),
      }),
    );
  });
  return bins;
}

function parseGuidance(region, where, bins, problems) {
  const source = region.categories;
  if (!source || typeof source !== 'object' || Array.isArray(source)) {
    problems.push(`${where}.categories must be an object keyed by category id`);
    return Object.freeze({});
  }

  const unknownKeys = Object.keys(source).filter((key) => !isValidCategory(key));
  if (unknownKeys.length > 0) {
    problems.push(`${where}.categories has keys outside the taxonomy: ${unknownKeys.join(', ')}`);
  }

  const guidance = {};
  for (const categoryId of categoryIds) {
    const at = `${where}.categories.${categoryId}`;
    const entry = Object.prototype.hasOwnProperty.call(source, categoryId) ? source[categoryId] : undefined;
    if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
      problems.push(`${at} is missing — every region must cover all ${categoryIds.length} categories`);
      continue;
    }
    const bin = isNonEmptyString(entry.binId) ? bins.get(entry.binId) : undefined;
    if (!bin) {
      problems.push(
        `${at}.binId "${entry.binId}" is not one of this region's bins (${[...bins.keys()].join(', ') || 'none'})`,
      );
      continue;
    }
    if (typeof entry.recyclable !== 'boolean') {
      problems.push(`${at}.recyclable must be a boolean`);
    }
    if (!isNonEmptyString(entry.disposal)) {
      problems.push(`${at}.disposal must be a non-empty string`);
    }
    for (const field of STRING_LIST_FIELDS) {
      if (!Array.isArray(entry[field])) problems.push(`${at}.${field} must be an array of strings`);
    }

    // Bin colours are denormalised into the guidance so the frontend can paint a result card
    // from one object; the bin remains the single source of truth when a field is omitted.
    guidance[categoryId] = Object.freeze({
      categoryId,
      binId: bin.id,
      binName: stringOr(entry.binName, bin.name),
      colorName: stringOr(entry.colorName, bin.colorName),
      colorHex: stringOr(entry.colorHex, bin.colorHex),
      textColorHex: stringOr(entry.textColorHex, bin.textColorHex),
      recyclable: entry.recyclable === true,
      disposal: stringOr(entry.disposal, ''),
      prepSteps: stringList(entry.prepSteps),
      acceptedExamples: stringList(entry.acceptedExamples),
      rejectedExamples: stringList(entry.rejectedExamples),
      notes: typeof entry.notes === 'string' ? entry.notes : '',
      dropOff: isNonEmptyString(entry.dropOff) ? entry.dropOff : null,
    });
  }
  return Object.freeze(guidance);
}

function readRules() {
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

  if (!parsed || typeof parsed !== 'object' || !Array.isArray(parsed.regions) || parsed.regions.length === 0) {
    throw invalid('must be an object with a non-empty "regions" array');
  }

  const problems = [];
  const regions = new Map();

  parsed.regions.forEach((region, index) => {
    const where = `regions[${index}]`;
    if (!region || typeof region !== 'object' || Array.isArray(region)) {
      problems.push(`${where} must be an object`);
      return;
    }
    if (!isNonEmptyString(region.id)) {
      problems.push(`${where}.id must be a non-empty string`);
      return;
    }
    if (regions.has(region.id)) {
      problems.push(`${where}.id "${region.id}" is duplicated`);
      return;
    }
    for (const field of REQUIRED_REGION_FIELDS) {
      if (!isNonEmptyString(region[field])) problems.push(`${where}.${field} must be a non-empty string`);
    }
    if (!Array.isArray(region.bins) || region.bins.length === 0) {
      problems.push(`${where}.bins must be a non-empty array`);
      return;
    }

    const bins = parseBins(region, where, problems);
    const guidance = parseGuidance(region, where, bins, problems);

    regions.set(
      region.id,
      Object.freeze({
        summary: Object.freeze({
          id: region.id,
          name: stringOr(region.name, region.id),
          country: stringOr(region.country, ''),
          authority: stringOr(region.authority, ''),
          updated: stringOr(region.updated, ''),
          notes: typeof region.notes === 'string' ? region.notes : '',
        }),
        bins: Object.freeze([...bins.values()]),
        binsById: bins,
        categories: guidance,
      }),
    );
  });

  const fileDefault = isNonEmptyString(parsed.defaultRegion) ? parsed.defaultRegion : null;
  if (!fileDefault) {
    problems.push('defaultRegion must be a non-empty string');
  } else if (!regions.has(fileDefault)) {
    problems.push(`defaultRegion "${fileDefault}" is not one of the defined regions`);
  }

  if (problems.length > 0) {
    throw invalid(problems.join('; '));
  }

  return { regions, fileDefault };
}

const { regions, fileDefault } = readRules();

/**
 * DEFAULT_REGION wins over the data file so an operator can retarget a deployment without
 * editing bundled data — but only if it actually exists, otherwise the app would start with
 * every region lookup failing.
 */
function resolveDefaultRegion() {
  const configured = config.defaultRegion;
  if (isNonEmptyString(configured) && regions.has(configured)) return configured;
  if (isNonEmptyString(configured) && configured !== fileDefault) {
    logger.warn('DEFAULT_REGION is not a known region — using the data file default instead', {
      configured,
      using: fileDefault,
      known: [...regions.keys()],
    });
  }
  return fileDefault;
}

export const defaultRegion = resolveDefaultRegion();

const regionSummaries = Object.freeze([...regions.values()].map((region) => region.summary));

logger.info(`Loaded recycling rules for ${regions.size} region(s)`, {
  regions: [...regions.keys()],
  defaultRegion,
});

/** Region ids that produced an unknown-region warning already, to keep stats runs quiet. */
const warnedUnknownRegions = new Set();

export function listRegions() {
  return regionSummaries;
}

export function getRegion(id) {
  if (typeof id !== 'string') return null;
  return regions.get(id)?.summary ?? null;
}

export function isValidRegion(id) {
  return typeof id === 'string' && regions.has(id);
}

/** Payload for GET /api/rules/:regionId. Null when the region is unknown. */
export function getRegionRules(id) {
  const region = typeof id === 'string' ? regions.get(id) : undefined;
  if (!region) return null;
  return { region: region.summary, bins: region.bins, categories: region.categories };
}

/** Payload for GET /api/rules/:regionId/:categoryId. Null when either id is unknown. */
export function getGuidance(regionId, categoryId) {
  const region = typeof regionId === 'string' ? regions.get(regionId) : undefined;
  // isValidCategory first: `categories` is a plain object, so an id like "constructor" would
  // otherwise resolve to something inherited.
  if (!region || !isValidCategory(categoryId)) return null;
  const guidance = region.categories[categoryId];
  if (!guidance) return null;
  return {
    region: region.summary,
    category: getCategory(categoryId),
    guidance,
    bin: region.binsById.get(guidance.binId) ?? null,
  };
}

/** Used by the stats service; unknown ids are "not recyclable" rather than an exception. */
export function isRecyclable(regionId, categoryId) {
  const region = typeof regionId === 'string' ? regions.get(regionId) : undefined;
  if (!region) {
    if (typeof regionId === 'string' && !warnedUnknownRegions.has(regionId)) {
      warnedUnknownRegions.add(regionId);
      logger.warn('Recyclability asked for an unknown region', { regionId });
    }
    return false;
  }
  if (!isValidCategory(categoryId)) return false;
  return region.categories[categoryId]?.recyclable === true;
}
