# `models/` — where EcoSort keeps its TensorFlow.js weights

This directory is bind-mounted into the backend container at `/models` (`MODELS_DIR`) and
served read-only over HTTP at `/models/...`. The browser loads the model from there and
runs every inference locally — nothing is ever sent to a cloud API.

Nothing in here is committed to git except this README and `.gitkeep`.

---

## The registry: one directory per model

EcoSort is not limited to a fixed set of slots. **Every directory under this one that
contains a `model.json` is a selectable engine**, and the frontend's model picker lists all
of them. Dropping a converted model in here is the whole installation procedure.

```
models/
  custom/                  from `make train`            -> waste categories directly
  mobilenet_v2/            from `make fetch-models`     -> ImageNet-1k
  inceptionresnetv2/       from `make convert-model`    -> ImageNet-1k
  <anything-else>/         discovered automatically
```

The rules the backend applies when it scans (`docs/ARCHITECTURE.md` §2.4 is authoritative):

* A directory counts only if it has a **`model.json`**. Weight shards without a manifest,
  stray downloads and empty directories are skipped.
* Directories whose name begins with a **dot are ignored**. That is what makes an
  in-flight conversion safe: `ml/convert_pretrained.py` writes into a sibling
  `.<arch>.staging-<pid>/` and moves the finished result into place in one step, so the
  picker never sees a half-written model.
* A directory that cannot be read, or whose `metadata.json` is corrupt, is **skipped with a
  warning** — never a 500. One bad model must not take the list down with it.
* The order is deterministic: `custom` first when it exists, then the ImageNet models by
  **download size ascending**, so the cheapest option is offered first. The first entry is
  flagged `recommended` and reported as `defaultModelId`.
* The scan is cached for **5 seconds**, so a new model shows up within a few seconds of
  being written. No restart, no rebuild — this directory is bind-mounted.

`GET /api/model/status` returns that registry as `models[]` plus `defaultModelId`, and
keeps its original `custom`, `fallback`, `active` and `searchedPaths` keys unchanged.
`curl localhost:4000/api/model/status` is the quickest way to see what the backend found
and where it looked.

---

## `models/custom/` — the trained classifier (preferred)

Produced by `ml/train.py` (MobileNetV2 transfer learning), exported to TFJS:

```
models/custom/
  model.json
  group1-shard1of*.bin …
  metadata.json
```

* Output: **softmax probabilities** over the 10 waste categories, in the order listed in
  `metadata.json.classes`.
* Not present until you train. Create it with `make train` (see `ml/README.md`).
* It is the only kind of model that answers the actual question — every ImageNet model
  below answers "what object is this?" and has its answer translated.

## `models/mobilenet_v2/` — the bundled fallback (always available)

Downloaded by `scripts/fetch-mobilenet.mjs` from the public TFJS model store:

```
models/mobilenet_v2/
  model.json            # ~112,708 bytes, TFJS *graph* model (no `format` field)
  group1-shard1of4      # 4,194,304 bytes
  group1-shard2of4      # 4,194,304 bytes
  group1-shard3of4      # 4,194,304 bytes
  group1-shard4of4      # 1,402,028 bytes   → 13,984,940 bytes total
  metadata.json
  SOURCE.txt            # upstream URL, Apache-2.0 licence, fetch date
```

* Output: **1001 raw logits**, no softmax in the graph. Index 0 is the TF-Slim synthetic
  *background* class, so it is sliced off (`classOffset: 1`) before softmax; the remaining
  1000 indices line up with `frontend/src/lib/imagenetClasses.js`.
* ImageNet class names are then mapped onto waste categories by
  `frontend/src/lib/imagenetWasteMap.js`.
* Load it with `tf.loadGraphModel`, **not** `tf.loadLayersModel`.

Get it with:

```bash
make fetch-models                  # → ./models/mobilenet_v2/
node scripts/fetch-mobilenet.mjs --force    # re-download over an existing copy
```

The script is idempotent (a complete model is left alone), atomic (it downloads into a
sibling temp directory and renames on success, so an interrupted run never leaves a
half-written model), and verifies the shard byte totals against the `weightsManifest`
before installing anything.

## `models/<arch>/` — converted `keras.applications` classifiers

Produced by `ml/convert_pretrained.py`, one directory per architecture, named after the
architecture in lower case:

```bash
make list-models                        # what can be converted
make convert-model                      # → ./models/inceptionresnetv2/ (float16)
make convert-model ARCH=InceptionV3     # → ./models/inceptionv3/
```

```
models/inceptionresnetv2/
  model.json               # TFJS *graph* model
  group1-shard1of27.bin …  # 27 shards; 112,262,070 bytes with the manifest (float16)
  metadata.json
```

* Output: a **1000-way softmax** — `keras.applications` models have no background class,
  so `classOffset` is `0` here where the bundled MobileNetV2 needs `1`. The frontend reads
  that from `metadata.json`; nothing is hard-coded per model.
* Input side length is whatever the architecture wants — **299** for the Inception family,
  224 for the MobileNets — and is likewise read from `metadata.json`.
* Load with `tf.loadGraphModel`: the converter goes Keras → SavedModel → `tfjs_graph_model`,
  the same route `ml/export_tfjs.py` uses.
* These are big. `inceptionresnetv2` is a ~107 MiB download even at float16, against
  13.3 MiB for the bundled MobileNetV2, which is exactly why the picker shows each model's
  size before you switch.

Delete one by deleting its directory; it disappears from the picker on the next scan.

---

## Preprocessing — identical for every model

**Every** model here takes pixel values in `[0, 1]`. Divide by 255 and do nothing else.

The bundled graph contains `hub_input/Mul (y=2.0)` followed by `hub_input/Sub (y=1.0)`,
which rescales `[0,1] → [-1,1]` internally. `ml/train.py` puts a matching
`tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)` first in the custom model, and
`ml/convert_pretrained.py` wraps every architecture it converts in the same layer — which is
why only architectures using Keras "tf"-mode preprocessing are offered at all. Applying the
usual Keras `x/127.5 - 1` transform yourself would double-apply it and wreck accuracy.

That contract is what keeps `frontend/src/lib/classifier.js` free of per-model branches:
resize to `inputSize`, divide by 255, run, then read `classOffset` and `labelKind` from the
metadata.

---

## `metadata.json` — the uniform descriptor

Every model directory carries the same shape, so the classifier has exactly one code path.
See `docs/ARCHITECTURE.md` §2.3 and §2.4 for the authoritative definition.

| field | custom | bundled fallback | converted |
|---|---|---|---|
| `name` | `ecosort-mobilenetv2` | `mobilenet_v2_1.0_224` | the arch, lower case |
| `inputSize` | `224` | `224` | `299` (Inception) / `224` (MobileNet) |
| `inputRange` | `[0, 1]` | `[0, 1]` | `[0, 1]` |
| `outputActivation` | `softmax` | `logits` | `softmax` |
| `classOffset` | `0` | `1` (drop the background class) | `0` |
| `labelKind` | `waste` | `imagenet` | `imagenet` |
| `classes` | the 10 waste ids | `null` | `null` |
| `classCount` | 10 | 1001 | 1000 |
| `metrics` | `{ valAccuracy, valLoss }` | omitted | omitted |

Four further fields describe the model to the picker. All four are optional — an older or
hand-written `metadata.json` still works, because each has a fallback:

| field | default when absent | meaning |
|---|---|---|
| `displayName` | the directory name, prettified | the name shown in the picker |
| `description` | `""` | one line of help under the name |
| `quantization` | `"none"` | `none` / `float16` / `uint8`, as converted |
| `downloadBytes` | summed from the `weightsManifest` | what the browser must fetch |

`downloadBytes` is what the "this is a 107 MB download" warning is computed from, so a
hand-edited value is a lie the UI will repeat. Let the converter write it.

A missing `metadata.json` altogether is not fatal — the classifier falls back to the
defaults for the directory it came from (`custom/` → waste labels, anything else → ImageNet
labels at 224×224).

---

## How the backend resolves the directories

`GET /models/<path>` is served from **two** roots, checked in this order:

1. **`MODELS_DIR`** — `/models` in the container, this directory on the host. Bind-mounted,
   so a newly trained `custom/` or a freshly converted model shows up without rebuilding
   the image.
2. **`BUNDLED_MODELS_DIR`** — `/app/bundled-models`, baked into the backend image at build
   time by `backend/Dockerfile`, which runs `scripts/fetch-mobilenet.mjs`.

The first root that has the requested file wins, so a host copy always shadows the bundled
one — the registry reports which root each model came from as `source: "models-dir"` or
`"bundled"`. That shadowing is what makes the stack work offline: even with an empty
`models/` directory on the host, the image already carries `mobilenet_v2/`.

If the image build happened without network access, the bake step prints a warning instead
of failing, and `GET /api/model/status` reports `fallback.available: false`. The fix is:

```bash
make fetch-models && docker compose restart backend
```

`GET /api/model/status` tells you exactly what was found, including the `searchedPaths` it
looked in.
