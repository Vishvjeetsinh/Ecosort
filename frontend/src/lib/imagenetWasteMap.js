/**
 * ImageNet-1k -> EcoSort waste taxonomy.
 *
 * The fallback engine is a generic ImageNet classifier, so its vocabulary is
 * "what is in the picture", not "which bin does this go in". This table is the
 * bridge: every ImageNet class that a person could plausibly point a camera at
 * while sorting household waste gets a waste category and a `weight`.
 *
 * `weight` is a confidence multiplier, NOT a probability:
 *   1.0        the class *is* the waste item ("water bottle" -> plastic)
 *   0.6 - 0.9  the class is usually made of / usually accompanies that stream
 *   0.2 - 0.5  a weak hint (a tool made of metal, a ceramic that is not glass)
 *
 * Classes that are animals, people, plants in the wild, landscapes, buildings
 * and vehicles are deliberately LEFT OUT: their probability mass then shows up
 * in `unmatchedMass` and the UI can honestly say "that does not look like a
 * piece of waste" instead of inventing a bin.
 *
 * Keys must be byte-exact entries of IMAGENET_CLASSES (synonyms included) --
 * `__tests__/imagenetWasteMap.test.js` enforces that.
 */

import { IMAGENET_CLASSES } from './imagenetClasses.js';

/** The 10 canonical waste ids from docs/ARCHITECTURE.md section 3. */
export const WASTE_CATEGORY_IDS = Object.freeze([
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
 * Human labels for the 10 ids. The backend serves richer category records from
 * `GET /api/categories`, but the classifier must be able to label a prediction
 * before any network round-trip has completed, so the labels live here too.
 */
export const WASTE_CATEGORY_LABELS = Object.freeze({
  plastic: 'Plastic',
  paper: 'Paper',
  cardboard: 'Cardboard',
  glass: 'Glass',
  metal: 'Metal',
  organic: 'Organic / Food',
  ewaste: 'Electronics',
  hazardous: 'Hazardous',
  textile: 'Textiles',
  trash: 'General Waste',
});

export const IMAGENET_WASTE_MAP = {
  // --- Plastic -------------------------------------------------------------
  'water bottle': { category: 'plastic', weight: 1 },
  'pop bottle, soda bottle': { category: 'plastic', weight: 1 },
  'pill bottle': { category: 'plastic', weight: 0.9 },
  'plastic bag': { category: 'plastic', weight: 1 },
  'water jug': { category: 'plastic', weight: 0.7 },
  'soap dispenser': { category: 'plastic', weight: 0.8 },
  'bucket, pail': { category: 'plastic', weight: 0.6 },
  'packet': { category: 'plastic', weight: 0.6 },
  'lotion': { category: 'plastic', weight: 0.5 },
  'shower curtain': { category: 'plastic', weight: 0.5 },
  'shower cap': { category: 'plastic', weight: 0.5 },
  'lens cap, lens cover': { category: 'plastic', weight: 0.6 },
  'ping-pong ball': { category: 'plastic', weight: 0.6 },
  'measuring cup': { category: 'plastic', weight: 0.5 },
  'pot, flowerpot': { category: 'plastic', weight: 0.5 },
  'tub, vat': { category: 'plastic', weight: 0.5 },
  'tray': { category: 'plastic', weight: 0.5 },
  'spatula': { category: 'plastic', weight: 0.5 },
  'nipple': { category: 'plastic', weight: 0.5 },
  'binder, ring-binder': { category: 'plastic', weight: 0.5 },
  'ballpoint, ballpoint pen, ballpen, Biro': { category: 'plastic', weight: 0.6 },
  'fountain pen': { category: 'plastic', weight: 0.5 },
  'hair slide': { category: 'plastic', weight: 0.5 },
  'sunglasses, dark glasses, shades': { category: 'plastic', weight: 0.5 },
  'sunglass': { category: 'plastic', weight: 0.4 },
  'rain barrel': { category: 'plastic', weight: 0.4 },
  'shopping basket': { category: 'plastic', weight: 0.4 },
  'pencil box, pencil case': { category: 'plastic', weight: 0.4 },
  'pencil sharpener': { category: 'plastic', weight: 0.4 },
  'rule, ruler': { category: 'plastic', weight: 0.4 },
  'snorkel': { category: 'plastic', weight: 0.4 },
  'window shade': { category: 'plastic', weight: 0.3 },
  'slide rule, slipstick': { category: 'plastic', weight: 0.3 },
  'pick, plectrum, plectron': { category: 'plastic', weight: 0.3 },
  'pinwheel': { category: 'plastic', weight: 0.3 },
  'reel': { category: 'plastic', weight: 0.3 },
  'toyshop': { category: 'plastic', weight: 0.3 },

  // --- Paper ---------------------------------------------------------------
  'envelope': { category: 'paper', weight: 1 },
  'toilet tissue, toilet paper, bathroom tissue': { category: 'paper', weight: 0.9 },
  'comic book': { category: 'paper', weight: 0.9 },
  'paper towel': { category: 'paper', weight: 0.8 },
  'crossword puzzle, crossword': { category: 'paper', weight: 0.8 },
  'book jacket, dust cover, dust jacket, dust wrapper': { category: 'paper', weight: 0.8 },
  'menu': { category: 'paper', weight: 0.8 },
  'library': { category: 'paper', weight: 0.4 },
  'bookshop, bookstore, bookstall': { category: 'paper', weight: 0.4 },

  // --- Cardboard -----------------------------------------------------------
  'carton': { category: 'cardboard', weight: 1 },
  'crate': { category: 'cardboard', weight: 0.45 },
  'jigsaw puzzle': { category: 'cardboard', weight: 0.5 },

  // --- Glass ---------------------------------------------------------------
  'beer bottle': { category: 'glass', weight: 1 },
  'wine bottle': { category: 'glass', weight: 1 },
  'beer glass': { category: 'glass', weight: 1 },
  'goblet': { category: 'glass', weight: 1 },
  'beaker': { category: 'glass', weight: 0.9 },
  'vase': { category: 'glass', weight: 0.8 },
  'whiskey jug': { category: 'glass', weight: 0.7 },
  'Petri dish': { category: 'glass', weight: 0.6 },
  'cocktail shaker': { category: 'glass', weight: 0.6 },
  'red wine': { category: 'glass', weight: 0.6 },
  'perfume, essence': { category: 'glass', weight: 0.5 },
  'saltshaker, salt shaker': { category: 'glass', weight: 0.5 },
  'hourglass': { category: 'glass', weight: 0.5 },
  'mixing bowl': { category: 'glass', weight: 0.4 },
  'pitcher, ewer': { category: 'glass', weight: 0.4 },
  'car mirror': { category: 'glass', weight: 0.3 },
  "loupe, jeweler's loupe": { category: 'glass', weight: 0.3 },
  'china cabinet, china closet': { category: 'glass', weight: 0.25 },

  // --- Metal ---------------------------------------------------------------
  'frying pan, frypan, skillet': { category: 'metal', weight: 0.9 },
  'wok': { category: 'metal', weight: 0.8 },
  'milk can': { category: 'metal', weight: 0.8 },
  'nail': { category: 'metal', weight: 0.7 },
  'screw': { category: 'metal', weight: 0.7 },
  'Dutch oven': { category: 'metal', weight: 0.6 },
  'caldron, cauldron': { category: 'metal', weight: 0.6 },
  'padlock': { category: 'metal', weight: 0.6 },
  'buckle': { category: 'metal', weight: 0.6 },
  'safety pin': { category: 'metal', weight: 0.6 },
  'brass, memorial tablet, plaque': { category: 'metal', weight: 0.6 },
  'chain': { category: 'metal', weight: 0.6 },
  'combination lock': { category: 'metal', weight: 0.5 },
  'bottlecap': { category: 'metal', weight: 0.5 },
  'ladle': { category: 'metal', weight: 0.5 },
  'strainer': { category: 'metal', weight: 0.5 },
  'cleaver, meat cleaver, chopper': { category: 'metal', weight: 0.5 },
  'steel drum': { category: 'metal', weight: 0.5 },
  'coffeepot': { category: 'metal', weight: 0.4 },
  'corkscrew, bottle screw': { category: 'metal', weight: 0.4 },
  'hammer': { category: 'metal', weight: 0.4 },
  'hatchet': { category: 'metal', weight: 0.4 },
  'screwdriver': { category: 'metal', weight: 0.4 },
  'hook, claw': { category: 'metal', weight: 0.4 },
  'thimble': { category: 'metal', weight: 0.4 },
  'radiator': { category: 'metal', weight: 0.4 },
  'safe': { category: 'metal', weight: 0.4 },
  'dumbbell': { category: 'metal', weight: 0.4 },
  'barbell': { category: 'metal', weight: 0.4 },
  'chain mail, ring mail, mail, chain armor, chain armour, ring armor, ring armour': {
    category: 'metal',
    weight: 0.4,
  },
  'necklace': { category: 'metal', weight: 0.4 },
  'chime, bell, gong': { category: 'metal', weight: 0.4 },
  'gong, tam-tam': { category: 'metal', weight: 0.35 },
  'manhole cover': { category: 'metal', weight: 0.35 },
  'shopping cart': { category: 'metal', weight: 0.35 },
  'shovel': { category: 'metal', weight: 0.35 },
  'breastplate, aegis, egis': { category: 'metal', weight: 0.4 },
  'cuirass': { category: 'metal', weight: 0.3 },
  // A tool, not waste: seeing it merely suggests tins are nearby.
  'can opener, tin opener': { category: 'metal', weight: 0.3 },
  'letter opener, paper knife, paperknife': { category: 'metal', weight: 0.3 },
  "plane, carpenter's plane, woodworking plane": { category: 'metal', weight: 0.3 },
  "carpenter's kit, tool kit": { category: 'metal', weight: 0.3 },
  'file, file cabinet, filing cabinet': { category: 'metal', weight: 0.3 },
  'folding chair': { category: 'metal', weight: 0.3 },
  'plate rack': { category: 'metal', weight: 0.3 },
  'chainlink fence': { category: 'metal', weight: 0.3 },
  'window screen': { category: 'metal', weight: 0.3 },
  'grille, radiator grille': { category: 'metal', weight: 0.3 },
  'disk brake, disc brake': { category: 'metal', weight: 0.3 },
  'mailbox, letter box': { category: 'metal', weight: 0.3 },
  'crutch': { category: 'metal', weight: 0.3 },
  'tripod': { category: 'metal', weight: 0.3 },
  'whistle': { category: 'metal', weight: 0.3 },
  'harmonica, mouth organ, harp, mouth harp': { category: 'metal', weight: 0.3 },

  // --- Organic / food ------------------------------------------------------
  'banana': { category: 'organic', weight: 1 },
  'orange': { category: 'organic', weight: 1 },
  'lemon': { category: 'organic', weight: 1 },
  'fig': { category: 'organic', weight: 1 },
  'pineapple, ananas': { category: 'organic', weight: 1 },
  'strawberry': { category: 'organic', weight: 1 },
  'Granny Smith': { category: 'organic', weight: 1 },
  'jackfruit, jak, jack': { category: 'organic', weight: 1 },
  'custard apple': { category: 'organic', weight: 1 },
  'pomegranate': { category: 'organic', weight: 1 },
  'corn': { category: 'organic', weight: 1 },
  'head cabbage': { category: 'organic', weight: 1 },
  'broccoli': { category: 'organic', weight: 1 },
  'cauliflower': { category: 'organic', weight: 1 },
  'zucchini, courgette': { category: 'organic', weight: 1 },
  'spaghetti squash': { category: 'organic', weight: 1 },
  'acorn squash': { category: 'organic', weight: 1 },
  'butternut squash': { category: 'organic', weight: 1 },
  'cucumber, cuke': { category: 'organic', weight: 1 },
  'artichoke, globe artichoke': { category: 'organic', weight: 1 },
  'bell pepper': { category: 'organic', weight: 1 },
  'mushroom': { category: 'organic', weight: 1 },
  'cardoon': { category: 'organic', weight: 0.9 },
  'pizza, pizza pie': { category: 'organic', weight: 1 },
  'cheeseburger': { category: 'organic', weight: 1 },
  'hotdog, hot dog, red hot': { category: 'organic', weight: 1 },
  'bagel, beigel': { category: 'organic', weight: 1 },
  'pretzel': { category: 'organic', weight: 1 },
  'French loaf': { category: 'organic', weight: 1 },
  'mashed potato': { category: 'organic', weight: 1 },
  'meat loaf, meatloaf': { category: 'organic', weight: 1 },
  'carbonara': { category: 'organic', weight: 1 },
  'potpie': { category: 'organic', weight: 1 },
  'burrito': { category: 'organic', weight: 1 },
  'guacamole': { category: 'organic', weight: 1 },
  'dough': { category: 'organic', weight: 0.9 },
  'consomme': { category: 'organic', weight: 0.9 },
  'hot pot, hotpot': { category: 'organic', weight: 0.8 },
  'trifle': { category: 'organic', weight: 0.9 },
  'ice cream, icecream': { category: 'organic', weight: 0.9 },
  'ice lolly, lolly, lollipop, popsicle': { category: 'organic', weight: 0.8 },
  'chocolate sauce, chocolate syrup': { category: 'organic', weight: 0.8 },
  'eggnog': { category: 'organic', weight: 0.8 },
  'espresso': { category: 'organic', weight: 0.7 },
  'ear, spike, capitulum': { category: 'organic', weight: 0.7 },
  "jack-o'-lantern": { category: 'organic', weight: 0.7 },
  'honeycomb': { category: 'organic', weight: 0.5 },
  'hay': { category: 'organic', weight: 0.8 },
  'acorn': { category: 'organic', weight: 0.6 },
  'hip, rose hip, rosehip': { category: 'organic', weight: 0.6 },
  'buckeye, horse chestnut, conker': { category: 'organic', weight: 0.6 },
  'hen-of-the-woods, hen of the woods, Polyporus frondosus, Grifola frondosa': {
    category: 'organic',
    weight: 0.6,
  },
  'bolete': { category: 'organic', weight: 0.5 },
  'agaric': { category: 'organic', weight: 0.5 },
  'coral fungus': { category: 'organic', weight: 0.4 },
  'gyromitra': { category: 'organic', weight: 0.4 },
  'stinkhorn, carrion fungus': { category: 'organic', weight: 0.4 },
  'earthstar': { category: 'organic', weight: 0.4 },
  'daisy': { category: 'organic', weight: 0.4 },
  "yellow lady's slipper, yellow lady-slipper, Cypripedium calceolus, Cypripedium parviflorum": {
    category: 'organic',
    weight: 0.3,
  },
  'drumstick': { category: 'organic', weight: 0.3 },
  'butcher shop, meat market': { category: 'organic', weight: 0.4 },
  'grocery store, grocery, food market, market': { category: 'organic', weight: 0.3 },
  'bakery, bakeshop, bakehouse': { category: 'organic', weight: 0.3 },
  'confectionery, confectionary, candy store': { category: 'organic', weight: 0.3 },
  'restaurant, eating house, eating place, eatery': { category: 'organic', weight: 0.3 },
  'greenhouse, nursery, glasshouse': { category: 'organic', weight: 0.25 },
  'barrow, garden cart, lawn cart, wheelbarrow': { category: 'organic', weight: 0.25 },

  // --- Electronics ---------------------------------------------------------
  'cellular telephone, cellular phone, cellphone, cell, mobile phone': {
    category: 'ewaste',
    weight: 1,
  },
  'laptop, laptop computer': { category: 'ewaste', weight: 1 },
  'notebook, notebook computer': { category: 'ewaste', weight: 1 },
  'desktop computer': { category: 'ewaste', weight: 1 },
  'hand-held computer, hand-held microcomputer': { category: 'ewaste', weight: 1 },
  'monitor': { category: 'ewaste', weight: 1 },
  'screen, CRT screen': { category: 'ewaste', weight: 1 },
  'television, television system': { category: 'ewaste', weight: 1 },
  'iPod': { category: 'ewaste', weight: 1 },
  'remote control, remote': { category: 'ewaste', weight: 1 },
  'computer keyboard, keypad': { category: 'ewaste', weight: 1 },
  'mouse, computer mouse': { category: 'ewaste', weight: 1 },
  'printer': { category: 'ewaste', weight: 1 },
  'modem': { category: 'ewaste', weight: 1 },
  'hard disc, hard disk, fixed disk': { category: 'ewaste', weight: 1 },
  'cassette player': { category: 'ewaste', weight: 1 },
  'CD player': { category: 'ewaste', weight: 1 },
  'tape player': { category: 'ewaste', weight: 1 },
  'radio, wireless': { category: 'ewaste', weight: 1 },
  'microwave, microwave oven': { category: 'ewaste', weight: 1 },
  'toaster': { category: 'ewaste', weight: 1 },
  'dial telephone, dial phone': { category: 'ewaste', weight: 0.9 },
  'digital watch': { category: 'ewaste', weight: 0.9 },
  'joystick': { category: 'ewaste', weight: 0.9 },
  'microphone, mike': { category: 'ewaste', weight: 0.9 },
  'loudspeaker, speaker, speaker unit, loudspeaker system, speaker system': {
    category: 'ewaste',
    weight: 0.9,
  },
  'electric fan, blower': { category: 'ewaste', weight: 0.9 },
  'vacuum, vacuum cleaner': { category: 'ewaste', weight: 0.9 },
  'hand blower, blow dryer, blow drier, hair dryer, hair drier': {
    category: 'ewaste',
    weight: 0.9,
  },
  'reflex camera': { category: 'ewaste', weight: 0.9 },
  'photocopier': { category: 'ewaste', weight: 0.9 },
  'Polaroid camera, Polaroid Land camera': { category: 'ewaste', weight: 0.8 },
  'espresso maker': { category: 'ewaste', weight: 0.8 },
  'washer, automatic washer, washing machine': { category: 'ewaste', weight: 0.8 },
  'dishwasher, dish washer, dishwashing machine': { category: 'ewaste', weight: 0.8 },
  'refrigerator, icebox': { category: 'ewaste', weight: 0.8 },
  'iron, smoothing iron': { category: 'ewaste', weight: 0.8 },
  'power drill': { category: 'ewaste', weight: 0.8 },
  'oscilloscope, scope, cathode-ray oscilloscope, CRO': { category: 'ewaste', weight: 0.8 },
  'projector': { category: 'ewaste', weight: 0.8 },
  'digital clock': { category: 'ewaste', weight: 0.8 },
  'home theater, home theatre': { category: 'ewaste', weight: 0.8 },
  'cassette': { category: 'ewaste', weight: 0.7 },
  'space bar': { category: 'ewaste', weight: 0.7 },
  'typewriter keyboard': { category: 'ewaste', weight: 0.7 },
  'stopwatch, stop watch': { category: 'ewaste', weight: 0.7 },
  'waffle iron': { category: 'ewaste', weight: 0.7 },
  'space heater': { category: 'ewaste', weight: 0.7 },
  'pay-phone, pay-station': { category: 'ewaste', weight: 0.6 },
  'sewing machine': { category: 'ewaste', weight: 0.6 },
  'stove': { category: 'ewaste', weight: 0.6 },
  'Crock Pot': { category: 'ewaste', weight: 0.6 },
  'table lamp': { category: 'ewaste', weight: 0.6 },
  'wall clock': { category: 'ewaste', weight: 0.6 },
  'switch, electric switch, electrical switch': { category: 'ewaste', weight: 0.6 },
  'analog clock': { category: 'ewaste', weight: 0.5 },
  'rotisserie': { category: 'ewaste', weight: 0.5 },
  'scale, weighing machine': { category: 'ewaste', weight: 0.5 },
  'chain saw, chainsaw': { category: 'ewaste', weight: 0.5 },
  'electric guitar': { category: 'ewaste', weight: 0.5 },
  'spotlight, spot': { category: 'ewaste', weight: 0.45 },
  'lawn mower, mower': { category: 'ewaste', weight: 0.4 },
  'entertainment center': { category: 'ewaste', weight: 0.4 },
  'binoculars, field glasses, opera glasses': { category: 'ewaste', weight: 0.3 },
  'barometer': { category: 'ewaste', weight: 0.3 },
  'magnetic compass': { category: 'ewaste', weight: 0.3 },
  'odometer, hodometer, mileometer, milometer': { category: 'ewaste', weight: 0.3 },
  'stethoscope': { category: 'ewaste', weight: 0.3 },
  'scoreboard': { category: 'ewaste', weight: 0.3 },
  'solar dish, solar collector, solar furnace': { category: 'ewaste', weight: 0.3 },
  'vending machine': { category: 'ewaste', weight: 0.3 },
  // A screenshot of a web page: a device is almost certainly in frame.
  'web site, website, internet site, site': { category: 'ewaste', weight: 0.2 },

  // --- Hazardous -----------------------------------------------------------
  'syringe': { category: 'hazardous', weight: 1 },
  'lighter, light, igniter, ignitor': { category: 'hazardous', weight: 0.9 },
  'hair spray': { category: 'hazardous', weight: 0.8 },
  'oil filter': { category: 'hazardous', weight: 0.7 },
  'sunscreen, sunblock, sun blocker': { category: 'hazardous', weight: 0.6 },
  'matchstick': { category: 'hazardous', weight: 0.5 },
  'medicine chest, medicine cabinet': { category: 'hazardous', weight: 0.4 },
  'lipstick, lip rouge': { category: 'hazardous', weight: 0.4 },
  'face powder': { category: 'hazardous', weight: 0.35 },
  'paintbrush': { category: 'hazardous', weight: 0.35 },
  'torch': { category: 'hazardous', weight: 0.35 },

  // --- Textiles ------------------------------------------------------------
  'jersey, T-shirt, tee shirt': { category: 'textile', weight: 1 },
  'sweatshirt': { category: 'textile', weight: 1 },
  'jean, blue jean, denim': { category: 'textile', weight: 1 },
  'cardigan': { category: 'textile', weight: 1 },
  'sock': { category: 'textile', weight: 1 },
  'running shoe': { category: 'textile', weight: 1 },
  'bath towel': { category: 'textile', weight: 1 },
  'wool, woolen, woollen': { category: 'textile', weight: 1 },
  'sandal': { category: 'textile', weight: 0.9 },
  'handkerchief, hankie, hanky, hankey': { category: 'textile', weight: 0.9 },
  'brassiere, bra, bandeau': { category: 'textile', weight: 0.9 },
  'bikini, two-piece': { category: 'textile', weight: 0.9 },
  'swimming trunks, bathing trunks': { category: 'textile', weight: 0.9 },
  'maillot': { category: 'textile', weight: 0.9 },
  'maillot, tank suit': { category: 'textile', weight: 0.9 },
  'miniskirt, mini': { category: 'textile', weight: 0.9 },
  'kimono': { category: 'textile', weight: 0.9 },
  'cloak': { category: 'textile', weight: 0.9 },
  'gown': { category: 'textile', weight: 0.9 },
  'abaya': { category: 'textile', weight: 0.9 },
  "academic gown, academic robe, judge's robe": { category: 'textile', weight: 0.9 },
  'lab coat, laboratory coat': { category: 'textile', weight: 0.9 },
  'fur coat': { category: 'textile', weight: 0.9 },
  'trench coat': { category: 'textile', weight: 0.9 },
  "pajama, pyjama, pj's, jammies": { category: 'textile', weight: 0.9 },
  'suit, suit of clothes': { category: 'textile', weight: 0.9 },
  'sarong': { category: 'textile', weight: 0.9 },
  'mitten': { category: 'textile', weight: 0.9 },
  'quilt, comforter, comfort, puff': { category: 'textile', weight: 0.9 },
  'hoopskirt, crinoline': { category: 'textile', weight: 0.8 },
  'overskirt': { category: 'textile', weight: 0.8 },
  'poncho': { category: 'textile', weight: 0.8 },
  'stole': { category: 'textile', weight: 0.8 },
  'vestment': { category: 'textile', weight: 0.8 },
  'military uniform': { category: 'textile', weight: 0.8 },
  'velvet': { category: 'textile', weight: 0.8 },
  'apron': { category: 'textile', weight: 0.8 },
  'bib': { category: 'textile', weight: 0.8 },
  'dishrag, dishcloth': { category: 'textile', weight: 0.8 },
  'Windsor tie': { category: 'textile', weight: 0.8 },
  'cowboy boot': { category: 'textile', weight: 0.8 },
  'cowboy hat, ten-gallon hat': { category: 'textile', weight: 0.8 },
  'bonnet, poke bonnet': { category: 'textile', weight: 0.8 },
  'sombrero': { category: 'textile', weight: 0.7 },
  'Loafer': { category: 'textile', weight: 0.8 },
  'clog, geta, patten, sabot': { category: 'textile', weight: 0.6 },
  'Christmas stocking': { category: 'textile', weight: 0.7 },
  'pillow': { category: 'textile', weight: 0.7 },
  'sleeping bag': { category: 'textile', weight: 0.7 },
  'ski mask': { category: 'textile', weight: 0.7 },
  'feather boa, boa': { category: 'textile', weight: 0.7 },
  'bow tie, bow-tie, bowtie': { category: 'textile', weight: 0.7 },
  'teddy, teddy bear': { category: 'textile', weight: 0.8 },
  'backpack, back pack, knapsack, packsack, rucksack, haversack': {
    category: 'textile',
    weight: 0.7,
  },
  'purse': { category: 'textile', weight: 0.6 },
  'doormat, welcome mat': { category: 'textile', weight: 0.6 },
  'prayer rug, prayer mat': { category: 'textile', weight: 0.8 },
  'mortarboard': { category: 'textile', weight: 0.6 },
  'bearskin, busby, shako': { category: 'textile', weight: 0.5 },
  'bathing cap, swimming cap': { category: 'textile', weight: 0.5 },
  'bulletproof vest': { category: 'textile', weight: 0.5 },
  'knee pad': { category: 'textile', weight: 0.5 },
  'mosquito net': { category: 'textile', weight: 0.5 },
  'mountain tent': { category: 'textile', weight: 0.5 },
  'parachute, chute': { category: 'textile', weight: 0.6 },
  'wallet, billfold, notecase, pocketbook': { category: 'textile', weight: 0.5 },
  'wig': { category: 'textile', weight: 0.5 },
  'theater curtain, theatre curtain': { category: 'textile', weight: 0.4 },
  'studio couch, day bed': { category: 'textile', weight: 0.4 },
  'mailbag, postbag': { category: 'textile', weight: 0.4 },
  'seat belt, seatbelt': { category: 'textile', weight: 0.4 },
  'bolo tie, bolo, bola tie, bola': { category: 'textile', weight: 0.4 },
  'shoe shop, shoe-shop, shoe store': { category: 'textile', weight: 0.4 },
  'hamper': { category: 'textile', weight: 0.3 },
  'wardrobe, closet, press': { category: 'textile', weight: 0.3 },
  'holster': { category: 'textile', weight: 0.3 },
  'punching bag, punch bag, punching ball, punchball': { category: 'textile', weight: 0.3 },
  'muzzle': { category: 'textile', weight: 0.3 },
  'knot': { category: 'textile', weight: 0.25 },

  // --- General waste (landfill: mixed materials, ceramics, soiled items) ----
  'diaper, nappy, napkin': { category: 'trash', weight: 0.9 },
  'Band Aid': { category: 'trash', weight: 0.8 },
  'candle, taper, wax light': { category: 'trash', weight: 0.7 },
  'rubber eraser, rubber, pencil eraser': { category: 'trash', weight: 0.5 },
  'lampshade, lamp shade': { category: 'trash', weight: 0.5 },
  'umbrella': { category: 'trash', weight: 0.5 },
  'balloon': { category: 'trash', weight: 0.5 },
  'plate': { category: 'trash', weight: 0.4 },
  'soup bowl': { category: 'trash', weight: 0.35 },
  'coffee mug': { category: 'trash', weight: 0.4 },
  'teapot': { category: 'trash', weight: 0.4 },
  'cup': { category: 'trash', weight: 0.3 },
  'mortar': { category: 'trash', weight: 0.3 },
  'wooden spoon': { category: 'trash', weight: 0.4 },
  'toilet seat': { category: 'trash', weight: 0.4 },
  'swab, swob, mop': { category: 'trash', weight: 0.4 },
  'broom': { category: 'trash', weight: 0.4 },
  "plunger, plumber's helper": { category: 'trash', weight: 0.3 },
  'mask': { category: 'trash', weight: 0.4 },
  'gasmask, respirator, gas helmet': { category: 'trash', weight: 0.3 },
  'oxygen mask': { category: 'trash', weight: 0.3 },
  'neck brace': { category: 'trash', weight: 0.3 },
  'mousetrap': { category: 'trash', weight: 0.3 },
  'piggy bank, penny bank': { category: 'trash', weight: 0.3 },
  'birdhouse': { category: 'trash', weight: 0.25 },
  'abacus': { category: 'trash', weight: 0.25 },
  'maraca': { category: 'trash', weight: 0.25 },
  'barrel, cask': { category: 'trash', weight: 0.3 },
  'crash helmet': { category: 'trash', weight: 0.4 },
  'football helmet': { category: 'trash', weight: 0.35 },
  'soccer ball': { category: 'trash', weight: 0.4 },
  'basketball': { category: 'trash', weight: 0.4 },
  'baseball': { category: 'trash', weight: 0.4 },
  'tennis ball': { category: 'trash', weight: 0.4 },
  'golf ball': { category: 'trash', weight: 0.4 },
  'volleyball': { category: 'trash', weight: 0.35 },
  'rugby ball': { category: 'trash', weight: 0.35 },
  'croquet ball': { category: 'trash', weight: 0.3 },
  'puck, hockey puck': { category: 'trash', weight: 0.3 },
  'racket, racquet': { category: 'trash', weight: 0.3 },
  // A bin in frame means waste is in frame, but says nothing about the stream.
  'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin': {
    category: 'trash',
    weight: 0.35,
  },
  'garbage truck, dustcart': { category: 'trash', weight: 0.4 },
};

/** Ids that actually occur in the map, in canonical order. */
export const MAPPED_CATEGORY_IDS = Object.freeze(
  WASTE_CATEGORY_IDS.filter((id) =>
    Object.values(IMAGENET_WASTE_MAP).some((entry) => entry.category === id),
  ),
);

/**
 * @param {string} name exact ImageNet class name
 * @returns {{category: string, weight: number}|null}
 */
export function lookupImagenetClass(name) {
  if (typeof name !== 'string') return null;
  return Object.prototype.hasOwnProperty.call(IMAGENET_WASTE_MAP, name)
    ? IMAGENET_WASTE_MAP[name]
    : null;
}

/**
 * Build the index -> entry lookup used by the inference hot path.
 *
 * The aggregation runs over 1000 probabilities per frame, and in webcam live
 * mode that is several times a second; an integer Map lookup avoids hashing a
 * 60-character class name a thousand times per frame.
 *
 * @param {string[]} [classNames]
 * @returns {Map<number, {category: string, weight: number, name: string}>}
 */
export function buildIndexMap(classNames = IMAGENET_CLASSES) {
  if (!Array.isArray(classNames)) {
    throw new TypeError('buildIndexMap expects an array of class names');
  }
  const index = new Map();
  for (let i = 0; i < classNames.length; i += 1) {
    const entry = lookupImagenetClass(classNames[i]);
    if (entry) {
      index.set(i, { category: entry.category, weight: entry.weight, name: classNames[i] });
    }
  }
  return index;
}

/**
 * Coverage diagnostics, surfaced in SettingsPanel and asserted by the tests.
 * @returns {{mapped: number, total: number, byCategory: Record<string, number>}}
 */
export function mapCoverage() {
  const byCategory = {};
  for (const id of WASTE_CATEGORY_IDS) byCategory[id] = 0;

  let mapped = 0;
  for (const entry of Object.values(IMAGENET_WASTE_MAP)) {
    mapped += 1;
    byCategory[entry.category] = (byCategory[entry.category] ?? 0) + 1;
  }

  return { mapped, total: IMAGENET_CLASSES.length, byCategory };
}

export default IMAGENET_WASTE_MAP;
