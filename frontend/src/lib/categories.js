/**
 * Helpers over the category list returned by `GET /api/categories`.
 *
 * Colours always come from data (`colorHex`), never from generated Tailwind class
 * names — those would be purged at build time because Tailwind only scans source.
 */

/** The canonical taxonomy ids from docs/ARCHITECTURE.md §3, in display order. */
export const CATEGORY_IDS = Object.freeze([
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
]);

/**
 * A local copy of the taxonomy used only until `GET /api/categories` resolves (and
 * if it fails outright), so predictions never render grey and unlabelled. The
 * backend's `waste-categories.json` remains the single source of truth.
 */
export const FALLBACK_CATEGORIES = Object.freeze(
  [
    {
      id: 'plastic',
      label: 'Plastic',
      shortLabel: 'Plastic',
      description: 'Bottles, tubs, film and packaging made from plastic.',
      icon: '🧴',
      colorHex: '#2563eb',
      textColorHex: '#ffffff',
      examples: ['Drink bottles', 'Yoghurt tubs', 'Shampoo bottles'],
    },
    {
      id: 'paper',
      label: 'Paper',
      shortLabel: 'Paper',
      description: 'Newspaper, office paper, magazines and envelopes.',
      icon: '📄',
      colorHex: '#0ea5e9',
      textColorHex: '#ffffff',
      examples: ['Newspaper', 'Printer paper', 'Magazines'],
    },
    {
      id: 'cardboard',
      label: 'Cardboard',
      shortLabel: 'Card',
      description: 'Corrugated boxes, cartons and egg boxes.',
      icon: '📦',
      colorHex: '#b45309',
      textColorHex: '#ffffff',
      examples: ['Shipping boxes', 'Cereal boxes', 'Egg cartons'],
    },
    {
      id: 'glass',
      label: 'Glass',
      shortLabel: 'Glass',
      description: 'Glass bottles and jars.',
      icon: '🫙',
      colorHex: '#059669',
      textColorHex: '#ffffff',
      examples: ['Wine bottles', 'Jam jars', 'Sauce jars'],
    },
    {
      id: 'metal',
      label: 'Metal',
      shortLabel: 'Metal',
      description: 'Cans, tins, foil and empty aerosols.',
      icon: '🥫',
      colorHex: '#64748b',
      textColorHex: '#ffffff',
      examples: ['Drink cans', 'Food tins', 'Aluminium foil'],
    },
    {
      id: 'organic',
      label: 'Organic / Food',
      shortLabel: 'Organic',
      description: 'Food scraps, garden waste and certified compostables.',
      icon: '🍎',
      colorHex: '#65a30d',
      textColorHex: '#ffffff',
      examples: ['Fruit peel', 'Coffee grounds', 'Garden trimmings'],
    },
    {
      id: 'ewaste',
      label: 'Electronics',
      shortLabel: 'E-waste',
      description: 'Devices, cables, chargers and screens.',
      icon: '🔌',
      colorHex: '#7c3aed',
      textColorHex: '#ffffff',
      examples: ['Old phones', 'Charging cables', 'Keyboards'],
    },
    {
      id: 'hazardous',
      label: 'Hazardous',
      shortLabel: 'Hazard',
      description: 'Batteries, paint, chemicals, bulbs and medicines.',
      icon: '⚠️',
      colorHex: '#dc2626',
      textColorHex: '#ffffff',
      examples: ['Batteries', 'Paint tins', 'Fluorescent bulbs'],
    },
    {
      id: 'textile',
      label: 'Textiles',
      shortLabel: 'Textile',
      description: 'Clothes, shoes and fabric.',
      icon: '👕',
      colorHex: '#db2777',
      textColorHex: '#ffffff',
      examples: ['T-shirts', 'Shoes', 'Bed linen'],
    },
    {
      id: 'trash',
      label: 'General Waste',
      shortLabel: 'Trash',
      description: 'Non-recyclable landfill items, and the "not sure" bucket.',
      icon: '🗑️',
      colorHex: '#44403c',
      textColorHex: '#ffffff',
      examples: ['Crisp packets', 'Used tissues', 'Broken ceramics'],
    },
  ].map((category) => Object.freeze(category)),
);

const FALLBACK_INDEX = Object.freeze(
  Object.fromEntries(FALLBACK_CATEGORIES.map((category) => [category.id, category])),
);

/** Build an `{ [id]: category }` map. Accepts null/garbage and returns `{}`. */
export function indexById(categories) {
  if (!Array.isArray(categories)) return {};
  const index = {};
  for (const category of categories) {
    if (category && typeof category.id === 'string') index[category.id] = category;
  }
  return index;
}

/**
 * Resolve one category. Accepts either the array from the API or an index map, and
 * degrades to the local copy so the UI is never colourless mid-load.
 */
export function categoryById(categories, id) {
  if (!id) return null;
  if (Array.isArray(categories)) {
    const hit = categories.find((category) => category && category.id === id);
    if (hit) return hit;
  } else if (categories && typeof categories === 'object' && categories[id]) {
    return categories[id];
  }
  return FALLBACK_INDEX[id] || null;
}

export function categoryLabel(categories, id) {
  const category = categoryById(categories, id);
  if (category && category.label) return category.label;
  return typeof id === 'string' && id.length > 0 ? id : 'Unknown';
}

export function categoryColor(categories, id) {
  const category = categoryById(categories, id);
  // slate-500: a neutral that stays legible against both themes.
  return (category && category.colorHex) || '#64748b';
}

export function categoryTextColor(categories, id) {
  const category = categoryById(categories, id);
  return (category && category.textColorHex) || '#ffffff';
}

/** Always returns a usable list, in canonical order, whatever the API did. */
export function categoriesOrFallback(categories) {
  if (Array.isArray(categories) && categories.length > 0) return categories;
  return FALLBACK_CATEGORIES;
}
