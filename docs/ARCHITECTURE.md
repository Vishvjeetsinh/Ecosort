# EcoSort — Architecture & Interface Contract

This document is the **binding contract** between every part of the system. Anything
written in here is authoritative: routes, JSON shapes, prop signatures, file ownership,
and the exact preprocessing maths. If code and this document disagree, the code is wrong.

---

## 1. What EcoSort does

A fully offline waste-sorting assistant.

1. The user gives it a picture of an item — from the **webcam** or by **uploading a file**.
2. A **TensorFlow.js** model running *in the browser* classifies it into a waste category.
3. The app shows the **top-3 categories** with confidence, the **bin colour** for the user's
   selected region, preparation steps, and what is/isn't accepted.
4. Every classification is posted to a **Node/Express** backend and stored in **SQLite**
   so the user gets history and statistics.

There are **no cloud API calls at runtime**. All inference is local. The only network
access in the whole project happens at *image build time* (`npm install`, and downloading
the pretrained MobileNetV2 weights, which are then baked into the backend image).

---

## 2. Two inference engines

| | **custom** | **fallback** |
|---|---|---|
| Served from | `/models/custom/model.json` | `/models/mobilenet_v2/model.json` |
| Origin | `ml/train.py` (MobileNetV2 transfer learning) | pretrained MobileNetV2, ImageNet-1k |
| Labels | waste categories directly | 1000 ImageNet classes → mapped to waste categories |
| Output | softmax probabilities, N classes | **1001 raw logits** |
| Present by default | **no** (user must train) | **yes** (baked into backend image) |

The frontend prefers `custom` and silently falls back to `fallback`. If neither is
present it shows an actionable error, never a blank screen.

### 2.1 MobileNetV2 fallback — empirically verified facts

These were verified by downloading the model and running inference. **Do not "fix" them.**

* Model URL:
  `https://storage.googleapis.com/tfjs-models/savedmodel/mobilenet_v2_1.0_224/model.json`
  plus weight shards `group1-shard1of4` … `group1-shard4of4` (total 13,984,940 bytes).
* It is a **TFJS graph model** (old TF-Hub converted format, no `format` field) →
  load with `tf.loadGraphModel`, **not** `loadLayersModel`.
* Input node `images`, shape `[-1, 224, 224, 3]`, dtype float32.
* **The model expects pixel values in `[0, 1]`.** The graph itself contains
  `hub_input/Mul (y = 2.0)` then `hub_input/Sub (y = 1.0)`, i.e. it rescales
  `[0,1] → [-1,1]` internally. So the correct preprocessing is **`pixels / 255`** and
  nothing else. Do *not* apply the Keras `x/127.5 - 1` transform.
* Output node `module_apply_default/MobilenetV2/Logits/output`, shape `[batch, 1001]`,
  **raw logits — there is no softmax in the graph.**
* Index 0 is the TF-Slim synthetic "background" class. Slice it off:
  `logits.slice([0, 1], [-1, 1000])`, then `softmax()`. The resulting index `i` matches
  `IMAGENET_CLASSES[i]` from `frontend/src/lib/imagenetClasses.js` (a generated file that
  already exists — 1000 entries, `[0] = "tench, Tinca tinca"`, `[999] = "toilet tissue, ..."`).
* Sanity check that must hold: a photo of a banana yields `banana` at ~97%.

### 2.2 Custom model contract

`ml/train.py` **must** build a Keras model whose *first layer* is a
`tf.keras.layers.Rescaling` that maps `[0,1]` onto the range its backbone was pretrained on,
so that the exported model, exactly like the fallback, **takes `[0,1]` input**. This keeps
one single preprocessing path in `classifier.js`. The final layer is a softmax over N waste
classes.

| backbone (`--backbone`) | first layer | maps `[0,1]` to |
|---|---|---|
| MobileNetV2 (default) | `Rescaling(scale=2.0, offset=-1.0)` | `[-1,1]` |
| EfficientNetV2-B0…B3 | `Rescaling(scale=255.0, offset=0.0)` | `[0,255]`; the Keras application's own preprocessing normalises from there |

`ml/export_tfjs.py` checks the first `Rescaling` against the backbone before exporting, and
a model trained with mixed precision is rebuilt in float32 first — TensorFlow.js has no
float16 tensors. `metadata.json.inputSize` carries the backbone's resolution (224–300), which
`classifier.js` already reads, so no frontend change follows from switching backbones.

### 2.3 `metadata.json` — uniform descriptor for both engines

Both `models/custom/` and `models/mobilenet_v2/` contain a `metadata.json` with the same
shape, so `classifier.js` has exactly one code path:

```jsonc
{
  "name": "ecosort-mobilenetv2",     // or "mobilenet_v2_1.0_224"
  "version": "1.0.0",
  "createdAt": "2026-09-18T00:00:00.000Z",
  "baseModel": "MobileNetV2",
  "inputSize": 224,                   // square side length
  "inputRange": [0, 1],               // ALWAYS [0,1]
  "outputActivation": "softmax",      // "softmax" (custom) | "logits" (fallback)
  "classOffset": 0,                   // 0 (custom) | 1 (fallback: drop background class)
  "labelKind": "waste",               // "waste" (custom) | "imagenet" (fallback)
  "classes": ["cardboard", "glass", "..."],  // custom only; omitted/null for imagenet
  "classCount": 1001,
  "metrics": { "valAccuracy": 0.0, "valLoss": 0.0 },  // custom only, optional
  "notes": "..."
}
```

A missing `metadata.json` is not fatal — `classifier.js` falls back to sane defaults
(`inputSize 224`, `inputRange [0,1]`, and `classOffset 1 / labelKind "imagenet"` for the
mobilenet_v2 path, `classOffset 0 / labelKind "waste"` for custom).

### 2.4 The model registry and model selection

EcoSort is not limited to the two original slots. **Any directory under a models root that
contains a `model.json` is a selectable engine.** The backend discovers them at request
time (with the same 5-second cache), so converting a new model makes it appear without a
restart.

```
models/
  custom/            <- from ml/train.py            (predicts waste categories directly)
  mobilenet_v2/      <- from scripts/fetch-mobilenet.mjs
  inceptionresnetv2/ <- from ml/convert_pretrained.py
  <anything else>/   <- discovered automatically
```

`metadata.json` gains four optional fields, all with sensible fallbacks so a hand-written
or older file still works:

| field | default | meaning |
|---|---|---|
| `displayName` | the id, prettified | what the picker shows |
| `description` | `""` | one line of help under the name |
| `quantization` | `"none"` | `none` / `float16` / `uint8` |
| `downloadBytes` | computed from the weight manifest | what the browser must fetch |

`GET /api/model/status` keeps every existing key (`custom`, `fallback`, `active`,
`searchedPaths`) unchanged and **adds**:

```jsonc
{
  "models": [
    {
      "id": "mobilenet_v2",
      "displayName": "MobileNetV2 (ImageNet)",
      "description": "...",
      "kind": "custom" | "imagenet",   // 'custom' = predicts waste ids directly
      "available": true,
      "source": "models-dir" | "bundled",
      "modelUrl": "/models/mobilenet_v2/model.json",
      "metadataUrl": "/models/mobilenet_v2/metadata.json",
      "metadata": { /* the merged descriptor, as before */ },
      "downloadBytes": 13984940,
      "recommended": true              // exactly one entry, the auto-pick
    }
  ],
  "defaultModelId": "mobilenet_v2"     // null when nothing is installed
}
```

Ordering is deterministic: the `custom` model first when present, then every ImageNet model
by `downloadBytes` ascending, so the cheapest download is offered first. Unreadable or
empty directories are skipped with a warning, never a 500.

**Frontend.** `loadEngine({ modelId, onProgress })` uses the requested model when it is
available and falls back to `defaultModelId` otherwise (a stored id for a model that has
since been deleted must not strand the user). The returned engine gains `modelId`,
`displayName` and `downloadBytes`. `useClassifier({ modelId })` disposes and reloads
whenever the id changes.

The selection lives in `localStorage` under `ecosort:modelId`; the empty string means
"auto — use whatever is recommended". Because a model can be ~100 MB, the picker **must**
show each option's download size, and anything over 50 MB is flagged before the user
switches.

Preprocessing is unchanged and stays uniform: every model declares `inputSize` and takes
`[0,1]`, so `classifier.js` needs no per-model branches. `inputSize` is read from metadata
(224 for the MobileNets, 299 for the Inception family).

---

## 3. The waste taxonomy — 10 categories

Canonical ids. Used by the backend rules, the frontend, the DB, and `ml/train.py`.

| id | label | colour (hex) | short meaning |
|---|---|---|---|
| `plastic`   | Plastic          | `#2563eb` | bottles, tubs, film, packaging |
| `paper`     | Paper            | `#0ea5e9` | newspaper, office paper, magazines |
| `cardboard` | Cardboard        | `#b45309` | corrugated boxes, cartons, egg boxes |
| `glass`     | Glass            | `#059669` | bottles, jars |
| `metal`     | Metal            | `#64748b` | cans, tins, foil, aerosols |
| `organic`   | Organic / Food   | `#65a30d` | food scraps, garden waste, compostables |
| `ewaste`    | Electronics      | `#7c3aed` | devices, cables, chargers, screens |
| `hazardous` | Hazardous        | `#dc2626` | batteries, paint, chemicals, bulbs, meds |
| `textile`   | Textiles         | `#db2777` | clothes, shoes, fabric |
| `trash`     | General Waste    | `#44403c` | non-recyclable landfill items |

`trash` doubles as the "we are not sure / not recyclable" bucket.

---

## 4. HTTP API (backend, port **4000**)

All responses are JSON unless stated. All errors use:

```json
{ "error": { "code": "not_found", "message": "human readable", "details": null } }
```

Codes: `bad_request` (400), `not_found` (404), `payload_too_large` (413),
`internal_error` (500).

### Static

* `GET /models/<path>` — serves the resolved models directory. **Both** the bind-mounted
  `MODELS_DIR` and the image-baked `BUNDLED_MODELS_DIR` are mounted at this prefix, with
  `MODELS_DIR` taking priority. Must send `Access-Control-Allow-Origin` (CORS) and
  `Cross-Origin-Resource-Policy: cross-origin` so the Vite dev server on :5173 can fetch it.
  Weight shards have no extension — serve them as `application/octet-stream`.

### Endpoints

```
GET  /api/health
  200 { status:"ok", version, uptimeSeconds, timestamp, db:{ ok:bool, path, classifications:int } }

GET  /api/model/status
  200 {
    custom:   { available:bool, modelUrl:"/models/custom/model.json"|null,
                metadataUrl:"/models/custom/metadata.json"|null, metadata:object|null },
    fallback: { available:bool, modelUrl:"/models/mobilenet_v2/model.json"|null,
                metadataUrl:"/models/mobilenet_v2/metadata.json"|null, metadata:object|null },
    active:   "custom"|"fallback"|"none",
    searchedPaths: string[]
  }

GET  /api/categories
  200 { categories:[ { id,label,shortLabel,description,icon,colorHex,textColorHex,examples:string[] } ] }

GET  /api/rules
  200 { defaultRegion:"us-generic", regions:[ { id,name,country,authority,updated,notes } ] }

GET  /api/rules/:regionId                 404 if unknown region
  200 { region:{...}, bins:[Bin], categories:{ [categoryId]: Guidance } }

GET  /api/rules/:regionId/:categoryId     404 if unknown region or category
  200 { region:{...}, category:Category, guidance:Guidance, bin:Bin }

POST /api/classifications
  body {
    predictions:[{category,label,confidence}]   // 1..10 entries, confidence 0..1, required
    source:"webcam"|"upload"                    // required
    modelKind:"custom"|"fallback"               // required
    regionId:string                             // required, must exist
    rawLabels?:[{label,confidence,index}]|null
    imageDataUrl?:string|null                   // "data:image/...;base64,..."  max 400_000 chars
    notes?:string|null                          // max 500
    durationMs?:number|null
  }
  201 { item: HistoryItem }
  400 on validation failure

GET  /api/classifications?limit=&offset=&category=&source=&modelKind=&regionId=&from=&to=&includeImage=
  limit default 25, max 100. offset default 0. includeImage default "true".
  from/to are ISO-8601 dates (inclusive) compared against created_at.
  200 { items:[HistoryItem], total:int, limit:int, offset:int }

GET    /api/classifications/:id      200 { item }   | 404
PATCH  /api/classifications/:id      body { correctedCategory?:string|null, notes?:string|null }
                                     200 { item }   | 400 | 404
DELETE /api/classifications/:id      204            | 404
DELETE /api/classifications?confirm=true            204 { } ; without confirm → 400

GET  /api/stats?regionId=&days=
  days default 30, max 365.
  200 {
    total:int, totalInWindow:int, windowDays:int,
    avgConfidence:number,          // 0..1, 0 when no rows
    recyclableRate:number,         // 0..1 share of rows whose effective category is recyclable in regionId
    byCategory:[{ category,label,colorHex,count,share }],       // desc by count, all 10 present
    bySource:[{ source,count }],
    byModelKind:[{ modelKind,count }],
    byDay:[{ day:"YYYY-MM-DD", count }],                        // dense, one entry per day in window
    topLabels:[{ label,count }],                                // max 10
    lastClassifiedAt:string|null
  }
```

### Shared shapes

```jsonc
Bin = { id, name, colorName, colorHex, textColorHex, accepts:[categoryId], description }

Guidance = {
  categoryId, binId, binName, colorName, colorHex, textColorHex,
  recyclable: bool,
  disposal: "Rinse and place in the blue recycling bin.",
  prepSteps: ["Empty any liquid", "Rinse", "Leave the cap on"],
  acceptedExamples: ["PET drink bottles", "..."],
  rejectedExamples: ["Plastic bags", "..."],
  notes: "string",
  dropOff: "string|null"        // where to take it when kerbside doesn't accept it
}

HistoryItem = {
  id:int, createdAt:ISO8601,
  topCategory, topLabel, topConfidence,
  predictions:[{category,label,confidence}],
  rawLabels:[{label,confidence,index}]|null,
  source, modelKind, regionId,
  imageDataUrl:string|null, correctedCategory:string|null, notes:string|null,
  durationMs:int|null,
  effectiveCategory: string       // correctedCategory ?? topCategory
}
```

### SQLite schema

Single table plus a migrations table. `better-sqlite3`, WAL mode, `PRAGMA foreign_keys=ON`.

```sql
CREATE TABLE IF NOT EXISTS schema_migrations (
  version     INTEGER PRIMARY KEY,
  name        TEXT NOT NULL,
  applied_at  TEXT NOT NULL
);

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
```

---

## 5. Frontend contract

React 18 + Vite 5 + Tailwind 3. Plain JavaScript with JSX (`.jsx`) — **no TypeScript**.
The Vite dev server proxies `/api` and `/models` to the backend, so the frontend only ever
uses same-origin relative URLs.

### Client-side shapes

```jsonc
Prediction     = { category, label, confidence, share? }
ClassifyResult = {
  predictions: Prediction[],        // top-3, sorted desc by confidence
  rawLabels: [{label,confidence,index}] | null,   // fallback only, top 10
  modelKind: "custom"|"fallback",
  durationMs: number,
  lowConfidence: boolean,           // true when top confidence < 0.20
  unmatchedMass: number             // fallback only: probability mass with no waste mapping
}
```

### `lib/classifier.js` — the public surface

```js
export async function loadEngine({ onProgress } = {}) -> Engine
// onProgress({ stage: 'probing'|'loading'|'warmup'|'ready', message, fraction })
// Resolution order: /api/model/status -> custom if available -> else fallback -> else throw EngineUnavailableError

Engine = {
  kind: 'custom' | 'fallback',
  classes: string[],                // waste ids (custom) or imagenet names (fallback)
  inputSize: number,
  metadata: object,
  classify(source, { topK = 3 }) -> Promise<ClassifyResult>,   // source: HTMLImageElement|HTMLVideoElement|HTMLCanvasElement
  // Live scan (5.1): many regions of ONE frame in one batched predict call.
  classifyRegions(source, regions, { topK = 3 }) -> Promise<{ results: ClassifyResult-like[], modelKind, durationMs }>,
  warmRegionBatches() -> Promise<void>,  // compiles batch sizes 1/2/4/6 once; idempotent
  dispose(): void
}

export class EngineUnavailableError extends Error {}
```

**Fallback aggregation algorithm (must be implemented exactly):**
1. probs = softmax(logits.slice([0,1],[-1,1000])) → 1000 absolute probabilities.
2. For each ImageNet index `i`: look up `IMAGENET_WASTE_MAP` by class name.
   A hit gives `{ category, weight }` (weight ∈ (0,1], default 1).
3. `scores[category] += probs[i] * weight`; misses accumulate into `unmatchedMass`.
4. Sort categories by score desc, take top-3, **keep the absolute score as `confidence`**
   (so confidences are honest probability mass and sum to ≤ 1, never normalised to 1).
5. `lowConfidence = topScore < 0.20`. If `scores` is empty, return a single
   `{ category:'trash', label:'Unrecognised item', confidence: 0 }` with `lowConfidence: true`.
6. `label` for a fallback prediction is the human category label plus the strongest
   contributing ImageNet name, e.g. `"Plastic — water bottle"`.

### `lib/imagenetWasteMap.js`

```js
export const IMAGENET_WASTE_MAP = {
  "water bottle": { category: "plastic", weight: 1 },
  "banana":       { category: "organic", weight: 1 },
  // ...
};
export function lookupImagenetClass(name) -> { category, weight } | null
```

Keys **must be exact strings from `IMAGENET_CLASSES`** (including the comma-separated
synonyms, e.g. `"pop bottle, soda bottle"`). A unit test asserts every key exists.

### Component tree & prop contracts

`App.jsx` owns all state and wires everything. Components are presentational unless noted.

```
<App>
  <AppHeader modelStatus regions regionId onRegionChange onOpenSettings />
  <Tabs value onChange items />          // "classify" | "scan" | "guide" | "history" | "stats"

  tab=classify:
    <CapturePanel mode onModeChange onImageReady busy disabled error onError />
      <WebcamCapture onCapture(canvas, dataUrl) active onError />
      <ImageDropzone onFile(file) busy />
      <PreviewCard dataUrl onClear onReclassify busy />
    <ResultsPanel result guidance rules categories busy error engineKind onSaved onCorrect />
      <PredictionList predictions categories selectedId onSelect />
        <PredictionRow prediction category rank selected onSelect />
          <ConfidenceBar value colorHex />
      <BinGuideCard guidance category bin region />
        <PrepStepList steps />
      <RawLabelsDisclosure rawLabels />

  tab=scan:                               // 5.1
    <LiveScanPanel detector engine engineReady engineProgress engineBlocked rules categories mirror topK onSaveItems saving />
      <DetectionBox track bin mirrored selected debug onSelect />   // one per tracked item
      <ScanItemList tracks rules categories selectedId onSelect mode debug />
      <BinGuideCard guidance category bin region />

  tab=guide:
    <BinColorGuide rules categories query onQueryChange />

  tab=history:
    <HistoryPanel items total loading filters onFiltersChange onDelete onCorrect onClearAll categories rules />
      <HistoryFilters filters onChange categories regions />
      <HistoryItemCard item category guidance onDelete onCorrect />

  tab=stats:
    <StatsPanel stats loading categories days onDaysChange />
      <CategoryBreakdown byCategory total />
      <DayBarChart byDay />

  <SettingsPanel open onClose settings onChange modelStatus onReloadEngine />
  <Toaster toasts onDismiss />
</App>
```

`categories` is always the array from `GET /api/categories`; helpers in `lib/categories.js`
(`indexById`, `categoryColor`, `categoryLabel`) are used rather than re-deriving.

### Styling rules

* Tailwind utility classes only — no CSS-in-JS, no component libraries.
* Dark mode via Tailwind's `class` strategy; `SettingsPanel` toggles `document.documentElement.classList`.
* Category colours come from data (`colorHex`), applied via inline `style`, **never** by
  building dynamic Tailwind class names (they would be purged).
* Must be responsive: single column under `md`, two columns at `lg`.
* Every interactive element needs a visible focus ring and an accessible name.

### 5.1 Live scan — detect, classify, track

The classifiers assume one item filling the frame. The **Live scan** tab handles a whole
scene: every item in view gets its own box, coloured by the bin it belongs in for the selected
region. Three stages run per frame, all in the browser:

```
fromPixels(frame) ─┬─> detector.detect ──> boxes (normalised, class-agnostic)
   (uploaded once) │        │ toPixelRegion: square, +10% padding, shifted into the frame
                   └─> engine.classifyRegions ──> waste predictions per box (one batch)
                                     │
                          tracker.update ──> stable, smoothed tracks ──> overlay + list
```

**Detector** (`lib/detector.js`, weights in `models/detectors/ssdlite_mobilenet_v2/`):
pretrained COCO SSDLite MobileNetV2, the model `@tensorflow-models/coco-ssd` calls
`lite_mobilenet_v2`. Graph model, run with `executeAsync`; int32 `[1,H,W,3]` input of raw
0–255 pixels; outputs `[1,1917,90]` sigmoid scores and `[1,1917,1,4]` `[ymin,xmin,ymax,xmax]`
boxes with no NMS applied; score column *j* is COCO id *j*+1. Two deliberate departures from
stock COCO-SSD, both measured on the waste test set:

* **The COCO label is not the answer.** It is usually wrong for waste (a phone scored as
  "bicycle") while the box is usually right, so a box's score is its best score over every
  class except people and furniture (`IGNORED_COCO_IDS`), and the waste classifier names it.
* **Threshold 0.2, not 0.5.** At 0.5 only 31% of test items got any box; at 0.2, 67%.

**Stills** (a frozen frame or an uploaded photo) also run the detector on a 2×2 grid of
overlapping tiles and merge the results (`mergeTiledDetections`: tile boxes cut by an inner
tile edge are dropped, the rest are suppressed by IoU *and* by containment). SSDLite sees a
300×300 thumbnail, so this is what finds small items; it costs five passes, so live frames
never use it.

**Classifier batching.** `classifyRegions` cuts every crop with one `tf.image.cropAndResize`
and pads the batch to 1, 2, 4 or 6. Both exist because WebGL compiles a shader per tensor
shape: per-crop `slice` + `resizeBilinear` recompiled on every frame (boxes move), and an
unpadded batch recompiled the whole network whenever the item count changed.
`warmRegionBatches()` compiles the four sizes before the first frame. The crop sampling grid
differs from the whole-image kernel by a sub-pixel shift, well inside the detector's box error.

**Tracker** (`lib/tracker.js`, pure): greedy IoU matching gives each item a stable id (also its
React key, so boxes animate instead of re-mounting); boxes and per-category confidences are
exponentially smoothed so a label only changes when the evidence does; a new item is shown
after two sightings and coasts for four frames after it disappears. Stills use *snap* mode:
exact boxes, immediate, unmatched tracks dropped.

**Saving** a frozen frame or photo writes one `POST /api/classifications` row per item, each
with its own crop as the thumbnail, so History and Stats count items exactly as for single
shots. No schema change.

Measured with headless Chrome on an Intel UHD 630 iGPU and a 1280×720 camera: ~10 fps with
two items in view (detect ≈ 60 ms, classify both ≈ 30 ms) with the MobileNetV2 custom model,
~5 fps (classify ≈ 113 ms) with the EfficientNetV2-B0 one; a still's tiled pass ≈ 400 ms.

---

## 6. File ownership (no two agents write the same file)

```
infra        docker-compose.yml, docker-compose.prod.yml, Makefile, .env.example,
             .gitignore, .dockerignore, backend/Dockerfile, backend/.dockerignore,
             frontend/Dockerfile, frontend/.dockerignore, frontend/nginx.conf,
             scripts/fetch-mobilenet.mjs, models/README.md, models/.gitkeep
backend-core backend/package.json, backend/src/{server,app,config,logger,db}.js,
             backend/src/middleware/*
backend-api  backend/src/routes/*, backend/src/services/*
backend-data backend/src/data/waste-categories.json, backend/src/data/recycling-rules.json
backend-test backend/tests/*
frontend-core frontend/package.json, vite.config.js, tailwind.config.js, postcss.config.js,
              index.html, .eslintrc / eslint.config.js, src/main.jsx, src/App.jsx, src/index.css,
              src/lib/{api,categories,format}.js, src/hooks/{useRules,useHistory,useLocalStorage,useToasts}.js,
              src/components/{AppHeader,ModelStatusBadge,RegionSelect,Tabs,Spinner,EmptyState,ErrorBanner,Toaster}.jsx
frontend-ml   src/lib/{classifier,imagenetWasteMap,imageUtils}.js, src/hooks/useClassifier.js,
              src/lib/__tests__/*
frontend-in   src/components/{CapturePanel,WebcamCapture,ImageDropzone,PreviewCard}.jsx,
              src/hooks/useWebcam.js
frontend-out  src/components/{ResultsPanel,PredictionList,PredictionRow,ConfidenceBar,
              BinGuideCard,PrepStepList,RawLabelsDisclosure,BinColorGuide}.jsx
frontend-hist src/components/{HistoryPanel,HistoryFilters,HistoryItemCard,StatsPanel,
              CategoryBreakdown,DayBarChart,SettingsPanel}.jsx
ml-train      ml/train.py, ml/prepare_dataset.py, ml/export_tfjs.py, ml/requirements.txt, ml/README.md
docs          README.md
generated     frontend/src/lib/imagenetClasses.js   (ALREADY WRITTEN — do not modify)
```

---

## 7. Runtime layout & environment

```
BACKEND  :4000   node:22-bookworm-slim
FRONTEND :5173   node:22-bookworm-slim (vite dev)  |  nginx:alpine (prod profile)
```

Environment variables (all have working defaults; see `.env.example`):

| var | default | meaning |
|---|---|---|
| `NODE_ENV` | `development` | |
| `BACKEND_PORT` | `4000` | host + container port for the API |
| `FRONTEND_PORT` | `5173` | host port for the Vite dev server |
| `DATA_DIR` | `/data` (container) / `./backend/data` (host) | SQLite lives here |
| `DB_FILE` | `${DATA_DIR}/ecosort.db` | |
| `MODELS_DIR` | `/models` (container) / `./models` (host) | bind-mounted, holds `custom/` |
| `BUNDLED_MODELS_DIR` | `/app/bundled-models` | baked into the backend image |
| `CORS_ORIGIN` | `http://localhost:5173` | comma-separated list, or `*` |
| `MAX_IMAGE_DATA_URL` | `400000` | characters |
| `LOG_LEVEL` | `info` | |
| `VITE_API_BASE` | *(empty)* | empty = same-origin + Vite proxy |
| `VITE_PROXY_TARGET` | `http://localhost:4000` | compose overrides to `http://backend:4000` |
| `DEFAULT_REGION` | `us-generic` | |

`docker compose up --build` must be sufficient — no manual steps. The backend Dockerfile
runs `scripts/fetch-mobilenet.mjs` at build time into `BUNDLED_MODELS_DIR`; if the download
fails the build still succeeds and the API reports `fallback.available: false` with the
`make fetch-models` remedy. `scripts/fetch-detector.mjs` does the same for the Live scan
detector, into `BUNDLED_MODELS_DIR/detectors/` — a level the classifier registry never lists,
while `/models/detectors/**` is still served. Both scripts share `scripts/lib/tfjs-model-fetch.mjs`.

---

## 8. Non-negotiables

* **No placeholders, no TODOs, no `...` elisions.** Every file is complete and runnable.
* **No cloud APIs at runtime.** No analytics, no CDN fonts, no remote model URLs in app code.
* Plain JavaScript (ESM, `"type": "module"`) on both sides. No TypeScript.
* Node 22. `better-sqlite3` for SQLite. `express@^4.21`.
* Tensors must be disposed — wrap inference in `tf.tidy()` and dispose anything that escapes.
* Every network call from the frontend goes through `lib/api.js`; it throws `ApiError`
  with `status`, `code`, `message`.
