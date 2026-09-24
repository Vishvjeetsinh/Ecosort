# EcoSort

**Point a camera at a piece of rubbish and EcoSort tells you which bin it goes in.**

It classifies the image with a model running in TensorFlow.js — MobileNetV2 out of the box,
or a bigger one you pick — shows the top‑3 waste categories with honest confidence numbers,
and turns the winner into local guidance: the bin colour, the preparation steps, what
belongs with it and what definitely does not — for whichever of seven recycling regions you
pick.

Everything runs on your machine. No cloud vision API, no image upload, no account, no
telemetry. The photo is classified **inside your browser tab** and never crosses the
network; the only thing the backend stores is the result you choose to save.

```bash
docker compose up --build
# → http://localhost:5173
```

That single command is the whole setup. The pretrained model is baked into the backend
image at build time, so the running stack needs no internet at all.

---

## Contents

- [What you get](#what-you-get)
- [Quick start](#quick-start)
- [How classification works](#how-classification-works)
- [Live scan: many items at once](#live-scan-many-items-at-once)
- [Choosing a model](#choosing-a-model)
- [The waste taxonomy](#the-waste-taxonomy)
- [Recycling regions](#recycling-regions)
- [Project layout](#project-layout)
- [HTTP API](#http-api)
- [Training your own model](#training-your-own-model)
- [Make targets](#make-targets)
- [Configuration](#configuration)
- [Running without Docker](#running-without-docker)
- [Production profile](#production-profile)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Design notes](#design-notes)
- [Licence and data provenance](#licence-and-data-provenance)

---

## What you get

| | |
|---|---|
| **Frontend** | React 18 + Vite 5 + Tailwind 3, plain JSX (no TypeScript) |
| **Inference** | `@tensorflow/tfjs` 4.22 in the browser — WebGL, falling back to CPU |
| **Backend** | Node 22 + Express 4, `better-sqlite3` (WAL), zod‑validated routes |
| **Database** | SQLite, one table plus a migration runner |
| **Training** | Python + Keras: MobileNetV2 transfer learning → TensorFlow.js |
| **Models** | bundled MobileNetV2, your trained classifier, or any converted `keras.applications` net |
| **Dev env** | Docker Compose: Vite dev server with HMR + API, one command |

Features:

- **Live scan**: point the camera at a table of mixed rubbish and every item gets its own
  box, coloured by the bin it goes in, tracked from frame to frame (~10 fps on an Intel iGPU
  with the MobileNetV2 model, ~5 fps with the more accurate EfficientNetV2-B0 one).
  Freeze a frame (or scan a photo) for a thorough tiled pass, then save every item at once.
- **Webcam capture** with a live square framing guide showing exactly the crop the model
  sees, a camera picker, a mirror toggle and an optional ~2 fps live mode.
- **Upload** by click, drag‑and‑drop or paste from the clipboard.
- **Top‑3 predictions** with confidence bars, and — on the ImageNet engines — a
  “what the model actually saw” disclosure listing the raw ImageNet labels.
- **Model picker**: your trained model, the bundled MobileNetV2, and any ImageNet network
  you convert with `make convert-model`, each listed with the download it costs.
- **Bin guidance** per region: colour, local bin name, disposal sentence, prep steps,
  accepted and rejected examples, drop‑off locations, and the issuing authority.
- **Bin colour guide** tab: searchable, with every swatch labelled by colour *name* as well
  as colour, so it works for colour‑blind users.
- **History** in SQLite with thumbnails, filters, pagination, per‑item category correction
  and date grouping.
- **Stats**: totals, average confidence, region‑aware recyclable rate, a hand‑rolled SVG
  day chart and a category breakdown. No charting library.
- **Dark mode** (light / dark / follow‑system), keyboard navigable throughout, responsive
  down to 360 px.

---

## Quick start

**Requirements:** Docker with Compose v2 (≥ 2.24 — the production override uses the
`!override` / `!reset` merge tags). Nothing else. Node and Python are only needed if you
want to run things outside Docker or train a model.

```bash
git clone <this repo> ecosort && cd ecosort
docker compose up --build
```

Then open **<http://localhost:5173>**.

- The API is on <http://localhost:4000> — check <http://localhost:4000/api/health>.
- The backend has a healthcheck and the frontend waits for it, so the first paint already
  has data.
- Camera access requires a **secure context**: `http://localhost` counts, a bare LAN IP
  like `http://192.168.1.5:5173` does **not**. Use the Upload tab, an SSH tunnel, or serve
  over HTTPS if you need the webcam from another device.

Stop with `Ctrl‑C`, or `docker compose down`. Your history survives in the `ecosort-data`
volume; `docker compose down -v` wipes it.

---

## How classification works

EcoSort has **three tiers of model**. It loads the best one that is installed, and you can
override that from the picker — see [Choosing a model](#choosing-a-model).

### 1. Custom model (preferred, optional)

If `models/custom/model.json` exists, it is used. That is what `ml/train.py` produces:
MobileNetV2 fine‑tuned on your own waste photos, predicting the waste categories directly.
Accurate, because it was trained for exactly this job. It covers whichever categories your
training data covers — all ten if you have images for all ten, fewer otherwise, and
`metadata.json` records exactly which.

### 2. Bundled MobileNetV2 (always present)

Otherwise EcoSort loads a pretrained **MobileNetV2 trained on ImageNet‑1k** and maps its
1000 object classes onto waste streams using `frontend/src/lib/imagenetWasteMap.js`
(**393 hand‑checked mappings**, each with a confidence weight).

So the model recognises a *water bottle* and EcoSort concludes *plastic*. It recognises a
*banana* and concludes *organic*. Classes that are not waste at all — animals, people,
landscapes, vehicles — are deliberately **left unmapped**, and their probability mass is
reported back as `unmatchedMass` rather than being laundered into a confident answer.

### 3. A converted `keras.applications` classifier (optional)

`make convert-model ARCH=InceptionResNetV2` turns any supported pretrained Keras ImageNet
network into a TensorFlow.js graph model under `models/<arch>/`, where it joins the
picker. It is an ImageNet classifier exactly like the bundled one, so it runs through the
same 393 mappings and the same arithmetic — it is simply much better at recognising the
object in the first place. `models/inceptionresnetv2/` is one of these: 299×299 input,
~80.3 % ImageNet top‑1, a 107 MB float16 download.

None of this branches. Every model declares its `inputSize` in `metadata.json` and takes
`[0, 1]` pixels, so adding one is a conversion plus a page reload, never a code change.

### The aggregation, precisely

This is what turns ImageNet classes into waste categories — identically for the bundled
MobileNetV2 and for every converted model:

1. Softmax the model's logits → 1000 absolute probabilities.
2. For each class, look up `{ category, weight }`; on a hit add `p × weight` to that
   category's score. On a miss, add `p` to `unmatchedMass`.
3. Sort categories by score and take the top 3.
4. **Confidences stay absolute.** They are real probability mass and sum to ≤ 1 — they are
   *not* renormalised to look like 100 %. A genuinely uncertain answer looks uncertain.
5. Below 0.20 the UI raises a low‑confidence advisory and explains what to do about it.

### The preprocessing contract

Every engine takes pixels in **`[0, 1]`** — `pixels / 255` and nothing else:

- The bundled pretrained graph rescales internally (it contains `hub_input/Mul (y=2.0)`
  then `hub_input/Sub (y=1.0)`, i.e. `[0,1] → [-1,1]`).
- The custom model's **first layer** is `Rescaling(scale=2.0, offset=-1.0)`, so the
  identical transform lives inside the exported graph.
- `ml/convert_pretrained.py` wraps every architecture it converts in that same
  `Rescaling` layer. It is precisely what `keras.applications.preprocess_input` does in
  “tf” mode, baked into the graph instead of left to the caller.

Applying the usual Keras `x/127.5 - 1` yourself would double‑apply it. One preprocessing
path, every engine — see `docs/ARCHITECTURE.md` §2 for the verified details.

The bundled MobileNetV2 outputs **1001** logits, not 1000: index 0 is the synthetic
“background” class from the TF‑Slim checkpoint. It is sliced off (`classOffset: 1`) before
the softmax, which is what lines the remaining indices up with `imagenetClasses.js`.
Converted Keras models have no background class — they end in a 1000‑way softmax with
`classOffset: 0`. The frontend reads that difference out of `metadata.json` rather than
hard‑coding it per model.

Each model directory carries a `metadata.json` describing `inputSize`, `inputRange`,
`outputActivation`, `classOffset` and `classes`, so the frontend has exactly one code path
for every engine and sniffs `model.json`'s `format` field to choose `loadGraphModel` vs
`loadLayersModel`.

---

## Live scan: many items at once

The Classify tab answers "what is this?" for one item filling the frame. The **Live scan**
tab answers it for a whole scene, in three stages that all run in the browser:

1. **Detect.** A pretrained COCO SSDLite MobileNetV2 (18 MB, `models/detectors/`) proposes
   boxes. Its COCO *label* is ignored: on waste it is usually wrong (a phone scored as a
   "bicycle") while the *box* is usually right, so boxes are scored class-agnostically at a
   0.2 threshold instead of COCO-SSD's 0.5, which took recall on the waste test set from 31%
   to 67%.
2. **Classify.** Each box becomes a padded square crop, and all crops go through the active
   waste classifier as **one batch** — so whichever model the picker shows names the items.
3. **Track.** An IoU tracker keeps each item's id, smooths its box and its label votes, and
   hides one-frame false positives, so labels do not flicker.

**Freeze frame** (or **Scan a photo**) re-scans the still with the detector run on the whole
picture *plus* four overlapping tiles, which finds small items the 300×300 live pass misses.
**Save** then writes one history row per item, each with its own crop as the thumbnail.

The **Debug** toggle shows what the detector itself thought each box was, and the per-frame
cost of each stage. The details, including why crops are batched and padded the way they
are, are in [ARCHITECTURE §5.1](docs/ARCHITECTURE.md#51-live-scan--detect-classify-track).

---

## Choosing a model

The settings panel lists every model the backend can see, with the download each one costs,
and remembers the choice in `localStorage`. *Auto* means “use whatever is recommended”:
your custom model when it exists, otherwise the cheapest ImageNet model installed. Anything
over 50 MB is flagged before you switch, because that download happens in the browser.

| model | input | download | ImageNet top‑1 | notes |
|---|---|---|---|---|
| **Custom** (`ml/train.py`) | 160–224 | **5.3 MB** measured | n/a | predicts the waste categories directly; most accurate for this job |
| **MobileNetV2** (bundled) | 224×224 | **13.4 MB** | ~71.8 % | the default; instant load |
| **InceptionResNetV2** | 299×299 | **107.1 MB** (float16) | ~80.3 % | best ImageNet accuracy here; slow first load |
| **InceptionV3** | 299×299 | **45.6 MB** (float16) | ~77.9 % | a good middle ground |
| **Xception** | 299×299 | ~44 MB (float16) | ~79.0 % | similar to InceptionV3 |
| **MobileNetV3Large** | 224×224 | ~11 MB (float16) | ~75.2 % | small and better than V2 |
| **NASNetMobile** | 224×224 | ~11 MB (float16) | ~74.4 % | compact |

Only the first three download figures are measured — the custom model at `alpha 1.0`, the
bundled MobileNetV2 at 14,097,648 bytes (13,984,940 of weights plus its 112,708-byte model.json), and the shipped
`models/inceptionresnetv2/`. **The rest are approximations until you convert them**;
`make convert-model` prints the real size when it finishes and writes the exact byte count
into the model's `metadata.json` as `downloadBytes`, which is the figure the picker shows.

### Bigger is not automatically better here

A larger ImageNet model recognises **the object** more reliably — it names the thing in
front of the camera correctly more often, which is a real gain and the reason
InceptionResNetV2 is offered at all.

But the **waste** answer still comes from the 393‑entry ImageNet → waste mapping, and that
is where the ceiling is. ImageNet's label set was never designed for this: 79 of the 393
mappings land in `textile` and 72 in `ewaste`, but only 9 in `paper` and 3 in `cardboard`,
because there simply are no ImageNet classes for “flattened cardboard box” or “rinsed
yoghurt pot”. A more accurate object classifier reads the same lopsided table, so it moves
the first half of the problem and leaves the second half exactly where it was.

**A custom model trained on real waste photos beats every one of them for this job**,
because it predicts the categories directly and never goes through the map at all. The
converted models are the best thing to run while you have no training data — not a
substitute for having some.

### Measured

The first trained model, for reference on what to expect:

| | |
|---|---|
| **Test accuracy** | **75.18%** on 560 held-out images (chance is 14.3%) |
| Validation accuracy | 77.14% |
| Classes | 7 of 10 — `ewaste, glass, hazardous, metal, organic, paper, plastic` |
| Data | 5,600 images, 800 per class, from the Kaggle `custom-waste-classification-dataset` |
| Training | MobileNetV2 `alpha 0.75` at 160×160, 10 + 5 epochs, ~30 min on a 12-core CPU |
| Download | 5.3 MB (5,559,881 bytes) |

Per-class F1 ranges from 0.68 (`glass`) to 0.85 (`organic`). The errors are the ones
inherent to the problem rather than signs of a broken model: `glass`↔`plastic` (transparent
containers), `plastic`↔`paper` (wrappers and labels), and `hazardous`→`ewaste` (a battery
is an electronic object). Test accuracy sitting *below* validation is the healthy
direction — it means the held-out split is genuinely held out.

`cardboard`, `textile` and `trash` are not in that dataset, so this model cannot predict
them. Fill those from TrashNet and the Kaggle 12-class set (see
[`ml/README.md`](ml/README.md)) and retrain to get all ten.

### Converting one

```bash
make list-models                                    # the supported architectures
make convert-model                                  # InceptionResNetV2 + float16 (the defaults)
make convert-model ARCH=InceptionV3                 # smaller, still much better than MobileNet
make convert-model ARCH=Xception QUANTIZE=uint8     # quarters the download, small accuracy cost
make convert-model ARCH=InceptionV3 CONVERT_ARGS=--force   # replace an existing export
```

`convert-model` depends on `venv`, so the Python toolchain is installed for you on first
use. A conversion takes a minute or two on a CPU, most of it the one‑off Keras weight
download (96 MB for InceptionV3, 225 MB for InceptionResNetV2) which is cached in
`~/.keras/models/` and never fetched again.

`QUANTIZE` is the download/accuracy dial:

| value | download | cost |
|---|---|---|
| `float16` *(default)* | **half** of float32 | no measurable accuracy loss |
| `uint8` | **a quarter** of float32 | a small, usually invisible, loss |
| `none` | full float32 | none — and 107 MB becomes 214 MB |

The output lands in `models/<arch lowercased>/` with its own `metadata.json`, and `./models`
is bind‑mounted into the backend, so **a converted model appears in the picker without a
restart or a rebuild** — within the 5‑second model‑status cache. While the conversion runs,
its output lives in a hidden `.<arch>.staging-<pid>` directory and is moved into place only
once it is complete and verified, so the picker never sees a half‑written model.

### Why ResNet, VGG and DenseNet are not offered

Because their preprocessing cannot be expressed as one `Rescaling` layer. The architectures
in the list all use Keras “tf” mode (`x/127.5 - 1` over `[0,255]`, i.e. exactly
`[0,1] → [-1,1]`), which the converter bakes into the graph — preserving EcoSort's single
`[0,1]` path. ResNet and VGG use “caffe” mode (BGR channel swap plus per‑channel mean
subtraction) and DenseNet uses “torch” mode (ImageNet mean/std). Getting either wrong
produces a model that loads, runs, and is confidently wrong — and a silently wrong
normalisation is far worse than an unsupported architecture.

---

## The waste taxonomy

Ten categories, used consistently by the UI, the API, the database and the trainer:

| id | label | colour | covers |
|---|---|---|---|
| `plastic` | Plastic | `#2563eb` | bottles, tubs, film, packaging |
| `paper` | Paper | `#0ea5e9` | newspaper, office paper, magazines |
| `cardboard` | Cardboard | `#b45309` | corrugated boxes, cartons, egg boxes |
| `glass` | Glass | `#059669` | bottles, jars |
| `metal` | Metal | `#64748b` | cans, tins, foil, aerosols |
| `organic` | Organic / Food | `#65a30d` | food scraps, garden waste, compostables |
| `ewaste` | Electronics | `#7c3aed` | devices, cables, chargers, screens |
| `hazardous` | Hazardous | `#dc2626` | batteries, paint, chemicals, bulbs, medicines |
| `textile` | Textiles | `#db2777` | clothes, shoes, fabric |
| `trash` | General Waste | `#44403c` | non‑recyclable landfill items |

`trash` doubles as the “not sure / not recyclable” bucket.

---

## Recycling regions

Rules genuinely differ by country, so `backend/src/data/recycling-rules.json` ships seven
real schemes rather than one generic one:

| id | scheme |
|---|---|
| `us-generic` | US single‑stream kerbside — blue recycling, green yard waste, black trash |
| `uk-london` | Typical London borough — blue mixed recycling, green food caddy, HWRC drop‑off |
| `de-berlin` | Duales System — Gelbe Tonne, Blaue Tonne, Braune Tonne, Glascontainer, **Pfand** |
| `jp-tokyo` | 23 wards — *moyaseru*/*moyasenai gomi*, *shigen*, separate PET, *sodai gomi* |
| `in-bengaluru` | BBMP — green (wet), blue/white (dry), red (sanitary), kabadiwala |
| `au-melbourne` | Victorian four‑bin — yellow recycling, **purple glass**, green FOGO, red general |
| `ca-toronto` | Blue Bin, Green Bin, Grey garbage, Community Environment Days |

These carry the differences that actually trip people up — Germany sorts by *packaging*,
so plastic and metal share the Gelbe Tonne; Melbourne's purple bin means glass must stay
out of the yellow one; Tokyo burns most non‑container plastics.

> Local rules change and vary street by street. Treat this as a well‑researched guide, not
> a legal authority, and check with your council when it matters. Each region records its
> `authority` and an `updated` date in the UI.

---

## Project layout

```
.
├── docker-compose.yml          dev stack: backend + Vite dev server
├── docker-compose.prod.yml     override: nginx-served production build
├── Makefile                    every routine task (`make help`)
├── .env.example                every setting, with its default
│
├── backend/                    Node 22 + Express 4 + SQLite
│   ├── Dockerfile              multi-stage; bakes the pretrained model in
│   └── src/
│       ├── app.js  server.js  config.js  db.js  logger.js
│       ├── middleware/         asyncHandler, validate (zod), errorHandler, …
│       ├── routes/             health, model, categories, rules, classifications, stats
│       ├── services/           all SQL and rules logic
│       └── data/               waste-categories.json, recycling-rules.json
│
├── frontend/                   React 18 + Vite 5 + Tailwind 3
│   ├── Dockerfile              targets: dev (Vite) and prod (nginx)
│   └── src/
│       ├── App.jsx             owns all state and wiring
│       ├── lib/
│       │   ├── classifier.js       the TFJS engine: load, preprocess, aggregate
│       │   ├── detector.js         Live scan: SSDLite boxes, class-agnostic, tiled stills
│       │   ├── tracker.js          Live scan: IoU tracking and label smoothing
│       │   ├── boxes.js  liveScan.js   box geometry; one detect → classify frame
│       │   ├── imagenetClasses.js  generated: the 1000 class names
│       │   ├── imagenetWasteMap.js 393 ImageNet → waste mappings
│       │   ├── imageUtils.js       centre-crop, downscale, decode
│       │   └── api.js              the only module that talks to the backend
│       ├── hooks/              useClassifier, useDetector, useLiveScan, useWebcam, …
│       └── components/         29 components
│
├── ml/                         host-side training (not in any image)
│   ├── train.py                two-stage MobileNetV2 transfer learning
│   ├── prepare_dataset.py      stratified split + dataset validation
│   ├── export_tfjs.py          Keras → TensorFlow.js + metadata.json
│   ├── convert_pretrained.py   keras.applications ImageNet net → TensorFlow.js
│   └── evaluate.py             report + single-image check for browser parity
│
├── models/                     bind-mounted into the backend; any dir with a model.json
│   ├── mobilenet_v2/           pretrained fallback (fetched)
│   ├── inceptionresnetv2/      converted (after `make convert-model`)
│   ├── custom/                 your trained model (after `make train`)
│   └── detectors/              Live scan object detector (fetched; never a classifier)
│
├── scripts/                    zero-dependency model downloaders
│   ├── fetch-mobilenet.mjs  fetch-detector.mjs
│   └── lib/tfjs-model-fetch.mjs    shared download, verify and atomic install
└── docs/ARCHITECTURE.md        the binding interface contract
```

---

## HTTP API

Base `http://localhost:4000`. Errors are always
`{ "error": { "code", "message", "details" } }`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | liveness + DB status + row count |
| `GET` | `/api/model/status` | which engines exist, their metadata, which is active |
| `GET` | `/api/categories` | the ten waste categories |
| `GET` | `/api/rules` | region list + default region |
| `GET` | `/api/rules/:region` | bins + guidance for all ten categories |
| `GET` | `/api/rules/:region/:category` | one category's guidance + its bin |
| `POST` | `/api/classifications` | save a result (thumbnail optional) |
| `GET` | `/api/classifications` | history: filter, paginate |
| `GET` | `/api/classifications/:id` | one row |
| `PATCH` | `/api/classifications/:id` | correct the category, edit notes |
| `DELETE` | `/api/classifications/:id` | delete one |
| `DELETE` | `/api/classifications?confirm=true` | clear all |
| `GET` | `/api/stats?days=&regionId=` | totals, per‑category, per‑day, recyclable rate |
| `GET` | `/models/**` | the model files themselves |

`POST /api/classifications` derives `topCategory`, `topLabel` and `topConfidence` from
`predictions[0]` server‑side rather than trusting the client. Full request and response
shapes are in `docs/ARCHITECTURE.md` §4.

---

## Training your own model

The bundled model is a decent demo and a converted one recognises objects better, but
neither was trained on waste: a model trained on real waste photos beats both, and it is
9 MB (see [Choosing a model](#choosing-a-model)). The trainer runs on the **host**, never in
a container.

```bash
make venv                       # creates ml/.venv and installs TensorFlow + tensorflowjs
make train-smoke                # ~1 min: proves the whole toolchain, needs no data
```

`train-smoke` trains on synthetic noise for one epoch of each stage and performs a real
TensorFlow.js export. If it succeeds, your environment is sound.

### With real data

```bash
ml/.venv/bin/python ml/prepare_dataset.py --scaffold   # creates ml/source/<category>/
# drop your photos into ml/source/<category>/
ml/.venv/bin/python ml/prepare_dataset.py             # ml/source → ml/dataset (train/val/test)
make train                                            # → models/custom/
```

Then click **Reload model** in EcoSort's settings (or refresh). The backend re‑checks the
models directory every few seconds, so no restart and no rebuild is needed — `./models` is
bind‑mounted into the container.

### The accurate model: all 10 classes, on an NVIDIA GPU

The first trained model knows 7 of the 10 categories (its data had no cardboard, textile or
trash) and scores 75% on its test split. The EfficientNetV2-B0 recipe below, even cut short
to fit a CPU, scored **87.4% on all 10 classes** — and 88.2% vs the old model's 40.6% on
photos from datasets neither had trained on. On a machine with an NVIDIA card (Linux, or
Windows through WSL2):

```bash
make venv venv-gpu      # CUDA-enabled TensorFlow from pip; only the NVIDIA driver is needed
make dataset            # downloads RealWaste + TrashNet, merges: 12,314 images, 10 classes
make train-gpu          # EfficientNetV2-B0, mixed precision, float16 export → models/custom/
```

Copy `models/custom/` back to the machine running EcoSort. The full walkthrough — WSL2,
backbone choices and their download sizes, the recipe — is in
[`ml/README.md`](ml/README.md#better-accuracy-on-an-nvidia-gpu).

`train.py` does the things that actually matter for transfer learning:

- **Stage 1** trains the head with the backbone frozen; **stage 2** unfreezes from
  `--fine-tune-at` with a much lower learning rate (AdamW + warmed-up cosine decay for
  EfficientNetV2).
- **BatchNorm layers stay frozen during fine‑tuning** — the classic transfer-learning pitfall.
- Augmentation runs on `[0,1]` images *before* the rescale, and is a no‑op at inference, so
  it never leaks into the exported graph.
- Class weights, label smoothing, optional MixUp/CutMix, early stopping, checkpoints.
- **Mixed precision without breaking the browser**: the model is rebuilt in float32 before
  export, because TensorFlow.js has no float16 tensors.
- Confusion matrices, per‑class precision/recall/F1 and training curves into `ml/artifacts/`.

`ml/README.md` covers dataset sources (TrashNet, TACO, the Kaggle garbage sets), how they
map onto the ten categories, expected training times and troubleshooting.

### Export format

The exporter converts **Keras → TF SavedModel → TFJS graph model**. This is deliberate:
Keras 3 writes `batch_shape` into its `InputLayer` config and `tf.loadLayersModel` cannot
read it, so a Keras‑3 *layers* export converts happily and then fails in the browser. The
SavedModel route avoids the Keras config entirely, loads on both Keras 2 and Keras 3
stacks, and runs faster. The exporter also refuses to ship a layers‑model it knows the
browser could not load. Verified: browser output matches Python to ~1e‑7.

---

## Make targets

`make help` lists them all. The ones you will actually use:

| target | does |
|---|---|
| `make up` / `make down` | build and start / stop the dev stack |
| `make logs` | follow both services |
| `make doctor` | print tool versions and which models are installed |
| `make fetch-models` | download the pretrained fallback and the Live scan detector into `./models` |
| `make test` | backend + frontend test suites |
| `make venv` | create `ml/.venv` and install the Python deps |
| `make train` / `make train-smoke` | train for real / prove the toolchain |
| `make dataset` | download RealWaste + TrashNet and build the 10-class dataset |
| `make venv-gpu` / `make gpu-check` | add CUDA to `ml/.venv` / explain whether the GPU is usable |
| `make train-gpu` | the recommended EfficientNetV2-B0 recipe on an NVIDIA GPU |
| `make list-models` | the pretrained architectures that can be converted |
| `make convert-model ARCH=… QUANTIZE=…` | convert one of them into `./models` |
| `make prod-up` | nginx production stack on :8080 |
| `make db-reset` | wipe the history table |
| `make clean` | stop, drop volumes, remove `node_modules` and build output |

---

## Configuration

Every variable has a working default, so **no `.env` file is required**. Copy
`.env.example` to `.env` to change anything. The ones worth knowing:

| variable | default | meaning |
|---|---|---|
| `BACKEND_PORT` | `4000` | host port for the API |
| `FRONTEND_PORT` | `5173` | host port for the Vite dev server |
| `MODELS_DIR` | `/models` | bind‑mounted models dir; wins over the baked‑in copy |
| `BUNDLED_MODELS_DIR` | `/app/bundled-models` | model baked into the backend image |
| `DB_FILE` | `/data/ecosort.db` | SQLite file inside the `ecosort-data` volume |
| `CORS_ORIGIN` | `localhost:5173,localhost:8080` | comma‑separated, or `*` |
| `MAX_IMAGE_DATA_URL` | `400000` | max thumbnail size in characters (else 413) |
| `DEFAULT_REGION` | `us-generic` | region used when the client picks none |
| `LOG_LEVEL` | `info` | `error` / `warn` / `info` / `debug` |

`DATA_DIR`, `MODELS_DIR` and `BUNDLED_MODELS_DIR` are pinned in `docker-compose.yml`
because they are wired to volume mounts — changing them via `.env` would orphan the mounts.

---

## Running without Docker

```bash
make install            # npm ci in both workspaces
make fetch-models       # populate ./models/mobilenet_v2
make dev-backend        # :4000
make dev-frontend       # :5173, proxying to :4000
```

Requires Node 22. On the host the backend defaults to `backend/data/ecosort.db` and
`./models`, so it behaves the same as in the container.

---

## Production profile

```bash
make prod-up            # → http://localhost:8080
```

This builds the frontend with Vite and serves the static bundle from nginx, which also
proxies `/api` and `/models` to the backend. It is a *local* production build — there is no
TLS termination, no auth and no rate limiting, so do not expose it to the internet as‑is.

---

## Tests

```bash
make test
```

- **Backend — 89 tests** (`node --test`, supertest): every endpoint's happy path and error
  path, pagination, filtering, validation limits, the 413 boundary, correction semantics,
  dense day‑series stats, a data‑integrity suite asserting every region covers all ten
  categories, every `binId` resolves, and every denormalised bin colour matches its bin,
  that a synthesised custom‑model descriptor never invents class names, and that the Live
  scan detector is served from `/models/detectors/` but never offered as a classifier.
- **Frontend — 153 tests** (vitest): the ImageNet→waste map (every key must exist in the
  real class list), the aggregation arithmetic, the formatters, the class‑order guard
  that refuses to name a custom model's outputs without a `classes` list to name them from,
  and Live scan's pure core — box geometry and crop regions, the detection threshold and
  class-agnostic NMS, the tiled-still merge, crop batching, and the tracker's matching,
  smoothing, coasting and snap behaviour.

The classifier's aggregation is exported as a pure function
(`aggregateImagenetPredictions`) precisely so it can be tested without a GPU or a DOM.

---

## Troubleshooting

**The camera never starts.** `getUserMedia` needs a secure context. `http://localhost`
qualifies; `http://<LAN-IP>` does not. Use Upload, tunnel, or serve over HTTPS.

**“No model available”.** The build‑time download was skipped or failed (offline build).
Run `make fetch-models`, then click *Reload model* — or `node scripts/fetch-mobilenet.mjs`
if you have no Make.

**A trained model does not show up.** It must be at `models/custom/model.json` with a
sibling `metadata.json`. Check `curl localhost:4000/api/model/status`. The model status is
cached for 5 seconds.

**The model takes ages to load the first time.** That is the weight download, not the
inference: InceptionResNetV2 is a 107 MB fetch before the first prediction. It happens
**once** — the browser caches the shards, so later loads are quick — but on a slow link
the first one is a long wait behind a progress bar. Pick a smaller model in settings, or
re‑convert with `make convert-model ARCH=InceptionV3 QUANTIZE=uint8 CONVERT_ARGS=--force`
to quarter the download. `MobileNetV2` and the custom model load effectively instantly.

**Predictions look silly on an ImageNet engine.** They will, sometimes — it is an ImageNet
classifier being asked a question it was not trained for. It has never seen a crushed can
on a kitchen counter. Fill the frame, use a plain background and good light; then train a
custom model.

**“Blocked request. This host … is not allowed.”** You are reaching the dev server through
a tunnel or a custom domain. Vite only answers to hostnames it recognises, because an
unrestricted dev server can be reached by any page that resolves its own domain to your
loopback (DNS rebinding). The common tunnel providers are allowed out of the box —
`.trycloudflare.com`, `.ngrok-free.app`, `.ngrok.io`, `.ngrok.app`, `.loca.lt`,
`.devtunnels.ms`, `.github.dev`, `.gitpod.io`, `.repl.co`, `.csb.app` — so
`cloudflared tunnel --url http://localhost:5173` just works. For any other hostname add it
to `.env`:

```bash
VITE_ALLOWED_HOSTS=ecosort.example.com,.my-tunnel.dev   # leading dot = all subdomains
```

`VITE_ALLOWED_HOSTS=*` accepts anything and switches the protection off — reasonable on a
trusted network, not on a shared one. Restart the frontend afterwards
(`docker compose restart frontend`).

**The console fills with HMR websocket errors behind a tunnel.** The page arrives on 443
but the HMR client still dials 5173 over plain `ws`. The app works; only hot reload is
broken. Point the client at the public port in `.env`:

```bash
VITE_HMR_CLIENT_PORT=443
VITE_HMR_PROTOCOL=wss
```

One upside of tunnelling: the tunnel serves over HTTPS, which is a secure context, so the
**webcam works from your phone or another machine** — something a bare `http://<LAN-IP>`
never allows.

**HMR stops firing.** Bind‑mount inotify events get lost on some hosts. Set
`CHOKIDAR_USEPOLLING=true` in `.env` and restart.

**`docker compose up` says the port is in use.** Set `BACKEND_PORT` / `FRONTEND_PORT` in
`.env`.

**`pip install -r ml/requirements.txt` fails.** Use the pins as written. Two of them are
load‑bearing and the comments explain why: `ydf<0.16` (newer builds ship protobuf 6
gencode, which TensorFlow's protobuf 5 runtime rejects) and `setuptools<81` (setuptools 81
removed `pkg_resources`, which `tensorflow_hub` still imports). Do not add your own
`tensorflow` bound on top — let `tensorflowjs` choose it.

---

## Design notes

**Why inference in the browser?** Privacy and honesty. The image genuinely never leaves the
machine, there is no GPU bill, and it scales to as many users as you have browsers. The
backend stays a small, boring JSON+SQLite service.

**Why absolute confidences?** Renormalising the top‑3 to sum to 100 % makes a coin flip
look like certainty. Reporting real probability mass means an uncertain answer *reads* as
uncertain, and `unmatchedMass` tells you when the model was mostly looking at something
that is not waste at all.

**Why denormalise bin colours into every guidance entry?** So the frontend never has to
join, and a rules file can never render a bin swatch that disagrees with its bin. A test
enforces that they match.

**Why is `imagenetClasses.js` generated?** Because getting the order wrong silently
mislabels everything. It is derived from the shipped model, and a test asserts the one
genuine upstream quirk — ImageNet reuses the label `"crane"` for both the bird (134) and
the machine (517), which is why that ambiguous name is deliberately left unmapped.

`docs/ARCHITECTURE.md` is the binding contract between the parts — read it before changing
an interface.

---

## Licence and data provenance

The application code in this repository is yours to use.

The pretrained fallback is Google's MobileNetV2 ImageNet module, converted for
TensorFlow.js and distributed under the **Apache License 2.0**. It is downloaded at build
time from `storage.googleapis.com/tfjs-models`; provenance, upstream URL and fetch date are
recorded in `models/mobilenet_v2/SOURCE.txt`.

The Live scan detector is the `ssdlite_mobilenet_v2_coco` checkpoint from the TensorFlow
Object Detection API model zoo, as converted and hosted by the tfjs-models project
(`@tensorflow-models/coco-ssd`), also **Apache License 2.0**, fetched the same way;
see `models/detectors/ssdlite_mobilenet_v2/SOURCE.txt`.

A custom model trained with `make dataset` learns from **RealWaste** (S. Single, S. Iranmanesh,
R. Raad, UCI Machine Learning Repository, doi:10.24432/C5SS4G, **CC BY 4.0** — attribution
required when you publish the model), **TrashNet** (G. Thung and M. Yang, **MIT**) and the
Kaggle *Custom Waste Classification* dataset you supply. The archives are fetched from their
publishers and verified by SHA-256 in `ml/build_dataset.py`.

Recycling rules were compiled from the published guidance of the named authorities. They
are a guide, not a legal reference — confirm locally when it matters.
