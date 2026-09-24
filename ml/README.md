# `ml/` — training your own EcoSort classifier

EcoSort ships with a **pretrained MobileNetV2 fallback** that maps ImageNet classes onto
waste categories. It works out of the box and needs nothing from this folder.

This folder is for the other engines. Mostly that means a **custom 10-class waste
classifier** you train on your own images: when `models/custom/model.json` exists, the
frontend prefers it and the model badge flips from `fallback` to `custom`. It also holds
`convert_pretrained.py`, which converts a pretrained `keras.applications` ImageNet network
(InceptionResNetV2 and friends) into a model EcoSort can load — no dataset required. See
[Converting a pretrained classifier](#converting-a-pretrained-classifier).

Everything here runs **on the host**, not in Docker. None of it is needed for
`docker compose up --build`.

**Want a more accurate model?** Jump to
[Better accuracy on an NVIDIA GPU](#better-accuracy-on-an-nvidia-gpu) — three commands on a
machine with an NVIDIA card.

---

## The one rule

> **The exported model takes pixels in `[0, 1]`.**

`frontend/src/lib/classifier.js` has exactly one preprocessing path, shared by every
engine: resize to the model's `inputSize` (224 here, 299 for the Inception family),
`toFloat()`, `div(255)`. Nothing else.

That is only correct because `train.py` puts a `Rescaling` layer first after the `Input`,
**inside the exported graph**, that maps `[0,1]` onto what the backbone was pretrained on:

| backbone | first layer | maps `[0,1]` to |
|---|---|---|
| MobileNetV2 | `Rescaling(scale=2.0, offset=-1.0)` | `[-1,1]` |
| EfficientNetV2-B0 … B3 | `Rescaling(scale=255.0, offset=0.0)` | `[0,255]`, which the Keras application then normalises itself |

MobileNetV2's mirrors the `hub_input/Mul(2.0)` + `hub_input/Sub(1.0)` pair baked into the
pretrained fallback graph (`docs/ARCHITECTURE.md` §2.1). `convert_pretrained.py` inserts
the equivalent layer in front of every architecture it converts, which is why the rule
holds for those models too.

So: **never** call any `tf.keras.applications.*.preprocess_input` in the data pipeline.
Doing that *and* the in-graph `Rescaling` double-applies the transform. The model still
trains happily and still reports a good validation score — it just predicts garbage in the
browser, because the browser feeds it `[0,1]` and Python fed it something else.
`export_tfjs.py` refuses to export a model whose first `Rescaling` is not the one its
backbone needs, and `evaluate.py` deliberately preprocesses with `/255` only, so a mismatch
shows up as Python and the browser disagreeing rather than as a silent accuracy hole.

---

## Install

Python **3.10 – 3.12**. Use a dedicated virtualenv — `tensorflowjs` pins a specific
TensorFlow minor and pulls in `tf-keras`, `tensorflow-decision-forests`, `jax` and
`jaxlib`. Mixing that into a general-purpose environment is how you lose an afternoon.

```bash
cd /path/to/EcoSort
python3 -m venv ml/.venv
. ml/.venv/bin/activate          # Windows: ml\.venv\Scripts\activate
pip install -U pip
pip install -r ml/requirements.txt
```

Budget ~2 GB of disk for the venv. On Apple Silicon you may need `tensorflow-macos`
instead of `tensorflow`; on an NVIDIA box use `tensorflow[and-cuda]`. Both are noted in
`ml/requirements.txt`.

### One note about the network

`pip install`, and the **first** `train.py` run, need internet: Keras downloads the
pretrained MobileNetV2 ImageNet weights (~14 MB) into `~/.keras/models/` and caches them
there forever. That is the only network access in this folder, it happens on your machine
before training, and it is unrelated to the app — EcoSort itself never calls out at
runtime. To train fully offline, copy `~/.keras/models/mobilenet_v2_weights_*.h5` from a
machine that has already run it.

### Verify the toolchain before collecting a single image

```bash
python ml/train.py --smoke-test
```

This ignores your dataset entirely. It synthesises 4 classes × 8 images of random noise at
224×224, trains one epoch of each stage at batch size 4, and then performs the **full**
TensorFlow.js export. It takes a couple of minutes on a CPU and proves that TensorFlow,
Keras and the `tensorflowjs` converter all work together before you invest in data.

The accuracy it reports is meaningless — it trained on noise. The throwaway export lands in
`ml/artifacts/smoke-export/`, deliberately **not** in `models/custom/`, so it can never be
picked up by the app.

---

## Better accuracy on an NVIDIA GPU

The first custom model scored **75.2% on its test split**, and it only knows **7 of the 10**
categories: its dataset had no cardboard, textile or trash, so it can never answer them. The
gains come from three places, in this order of importance:

1. **Data.** `make dataset` downloads two openly licensed sets — RealWaste (UCI, CC BY 4.0)
   and TrashNet (MIT) — and merges them with the Kaggle set you already have. Result:
   **12,314 images across all 10 classes** (was 5,600 across 7), de-duplicated across sources
   (184 duplicates dropped), capped at 1,500 per class, re-encoded to ≤512 px with EXIF
   rotation applied (the browser applies it; Keras does not).
2. **Backbone.** EfficientNetV2-B0 at 224 px instead of MobileNetV2 (α 0.75) at 160 px, and
   fine-tuned *entirely* rather than from layer 100.
3. **Recipe.** AdamW with a warmed-up cosine schedule for the fine-tune, MixUp/CutMix on
   30% of batches, square-root class weights for the thin classes (textile, trash), batches
   reshuffled every epoch, and mixed precision — all in `make train-gpu`.

### On the GPU machine

TensorFlow's CUDA build runs on **Linux or WSL2 only** (it dropped native Windows GPU support
after 2.10). The only system-level install is the **NVIDIA driver**; CUDA and cuDNN come from
pip.

| the GPU machine runs | do this first |
|---|---|
| Linux | install the NVIDIA driver, reboot, check `nvidia-smi` lists the card |
| Windows 10/11 | install the NVIDIA driver *on Windows*, then `wsl --install -d Ubuntu`, and do everything below inside Ubuntu (WSL2 shares the Windows driver; do **not** install a driver inside WSL) |

Then, in a copy of this project:

```bash
make venv venv-gpu      # training deps, then CUDA-enabled TensorFlow; ends with a GPU check
make dataset            # ~700 MB download once, then merge + split (≈10 min)
make train-gpu          # EfficientNetV2-B0, 8 + 30 epochs, exports to models/custom/
make evaluate           # the honest number: the test split
```

`make dataset` needs the Kaggle customwaste set in `custom-waste-classification-dataset/`,
exactly where it sits on the machine that trained the first model — copy that folder along
with the project (or see the command it prints). It is the only source of `ewaste` and
`hazardous`.

Bring the result home by copying **`models/custom/`** to the machine that runs EcoSort and
pressing *Reload model* in Settings. It holds `metadata.json` with the new class list, so the
app picks up all 10 categories with no code change.

`make gpu-check` is worth running first on its own: a GPU TensorFlow cannot see does not
raise an error — training silently falls back to the CPU and takes 20–50× longer. It names
the cause (no driver, native Windows, missing CUDA libraries) instead.

### Choosing a backbone

`make train-gpu TRAIN_ARGS="--backbone efficientnetv2b2"` swaps the network and keeps the rest
of the recipe. Every option exports to a TensorFlow.js graph model the app loads unchanged;
bigger ones cost download size and Live scan frame rate.

| `--backbone` | input | backbone params | float16 download | notes |
|---|---|---|---|---|
| `mobilenetv2` | 224 | 2.26 M | ≈ 4.3 MiB | the old default; fastest on CPU |
| `efficientnetv2b0` | 224 | 5.92 M | 11.5 MiB | **recommended**: the `make train-gpu` default |
| `efficientnetv2b1` | 240 | 6.93 M | ≈ 13.2 MiB | |
| `efficientnetv2b2` | 260 | 8.77 M | ≈ 16.7 MiB | more accurate, Live scan slower |
| `efficientnetv2b3` | 300 | 12.93 M | ≈ 24.7 MiB | largest that is still comfortable in a browser |

Sizes marked ≈ are two bytes per parameter; B0's is a real float16 export.

**Measured, same data, same test images (1,231 photos, all 10 classes).** A shortened run of
this recipe on a CPU (3 + 7 epochs, no GPU, no mixed precision) against the first model:

| | first model (MobileNetV2 α0.75, 160 px, 7 classes) | EfficientNetV2-B0, 224 px, 10 classes |
|---|---|---|
| all 10 classes | 57.9% (cannot answer 3 of them) | **87.4%** |
| the 7 classes both know | 67.9% | **87.6%** |
| photos from RealWaste/TrashNet, seen by neither in training | 40.6% | **88.2%** |
| Live scan on an Intel UHD 630 iGPU, two items in view | 9.9 fps | 5.3 fps |

The full `make train-gpu` schedule (8 + 30 epochs, MixUp/CutMix) should land above that CPU run.
Its weakest classes were `plastic` (70% recall — often called glass or metal) and `trash`
(75%), which is where more photos of your own will pay off first. The cost is the last row:
on integrated graphics the bigger network halves Live scan's frame rate. On a discrete GPU
that does not matter; on a laptop iGPU, `--image-size 192` claws some of it back.
ConvNeXt was tried and left out: its exact-GELU op (`Erfc`) is not supported by the
TensorFlow.js converter.

If the GPU runs out of memory, lower the batch: `make train-gpu TRAIN_ARGS="--batch-size 32"`.
If your card predates tensor cores (GTX 10-series and older), mixed precision does not help:
`make train-gpu GPU_PRECISION=`.

---

## The 10 categories

From `docs/ARCHITECTURE.md` §3. Folder names must match these ids exactly:

```
cardboard  ewaste  glass  hazardous  metal  organic  paper  plastic  textile  trash
```

Sorted alphabetically — and that sort order **is** the softmax output order. It is written
once into `ml/dataset/dataset.json` and copied into `models/custom/metadata.json`, so the
browser and the model always agree on which index means what. Never rename a class folder
after training without retraining.

---

## Workflow

### 1. Create the folders

```bash
python ml/prepare_dataset.py --scaffold
```

Creates `ml/source/<class>/` for all ten categories, each with a README describing what
belongs in it and where to find images.

### 2. Collect images

Put JPEG/PNG files straight into `ml/source/<class>/`. Aim for **at least 100 per class**;
below 20 the splitter warns and the model will memorise rather than learn. Vary the
background, lighting, angle and distance — a hundred photos of one bottle on one table
teaches the model about your table.

To pull in a downloaded dataset rather than your own photos, see the next step. Either way,
the target is the same ten folders.

### 2b. Merge downloaded datasets — `ingest_sources.py`

Every public dataset uses its own folder names: TrashNet says `cardboard`, the Kaggle
12-class set splits glass three ways by bottle colour, and nothing but your own photos says
`ewaste`. `ingest_sources.py` owns that mapping so `prepare_dataset.py` doesn't have to.

Start by looking at what you downloaded:

```bash
python ml/ingest_sources.py --inspect ~/Downloads/some-dataset
```

It prints every folder, its image count, and the EcoSort class it would map to — including
the ones it *cannot* map, which are the ones you have to decide about. Nothing is copied.

Then ingest, one `--source <name>=<dir>` per dataset:

```bash
python ml/ingest_sources.py \
    --source trashnet=~/Downloads/trashnet/data \
    --source garbage12=~/Downloads/garbage-classification \
    --source ewaste=~/Downloads/e-waste \
    --map 'Mobile Phones=ewaste' --map 'PCB=ewaste' \
    --cap 800 --dry-run
```

Drop `--dry-run` to write. `--link` hardlinks instead of copying, which matters when the
sources are several GB.

Three things it does that a few `cp` commands do not, all of which fail silently otherwise:

- **Renames every file `<source>__<original>`.** TrashNet and the Kaggle 12-class set both
  contain `cardboard/cardboard1.jpg`; copy both into one folder and you lose one without a
  word.
- **Drops byte-identical duplicates.** These datasets re-package each other. A duplicate
  that lands in train *and* test inflates your test accuracy for free, which defeats the
  point of measuring it.
- **Caps each class (`--cap 800`).** `clothes` + `shoes` is ~7,270 images against
  TrashNet's ~137 for `trash`. Uncapped, `train.py`'s inverse-frequency class weights turn
  that 29× ratio into a 29× weight on the noisiest class.

It only moves files. Decoding, validation and the split stay in `prepare_dataset.py`.

### 3. Split

```bash
python ml/prepare_dataset.py --dry-run      # see the plan, touch nothing
python ml/prepare_dataset.py                # copy into ml/dataset/{train,val,test}
```

The split is stratified per class, seeded (`--seed 42`) and therefore reproducible. Every
image is actually opened with Pillow, so corrupt files are reported and skipped rather than
crashing training two hours in. The per-class table and the imbalance ratio are printed,
and `ml/dataset/dataset.json` records the class list, the counts and the seed.

Useful flags: `--val-split`, `--test-split`, `--min-per-class`, `--move` (instead of copy),
`--overwrite`, `--allow-extra-classes`.

### 4. Train

```bash
python ml/train.py
```

Two stages:

| stage | backbone | lr | epochs | what it does |
|---|---|---|---|---|
| 1 | frozen | Adam `1e-3` | `--epochs` (20) | trains only the new classifier head |
| 2 | unfrozen from `--fine-tune-at` | see below | `--fine-tune-epochs` (10) | adapts the backbone |

Stage 2 depends on `--backbone`. MobileNetV2 (the default) keeps the original recipe:
layers 100+ unfrozen, Adam `1e-5`, halved on a val-loss plateau. EfficientNetV2 unfreezes
every layer and uses AdamW (weight decay `1e-4`) with a one-epoch linear warm-up to `1e-4`
and cosine decay to 1% of it. Each of these is a flag (`--fine-tune-at`, `--fine-tune-lr`,
`--schedule`, `--weight-decay`, `--warmup-epochs`).

Every `BatchNormalization` layer stays frozen in stage 2 and the backbone is always called
with `training=False`. That is not an oversight: with small batches, letting BN update its
moving averages is *the* classic MobileNetV2 fine-tuning failure — training accuracy keeps
climbing while validation accuracy collapses.

Class weights are on by default: inverse frequency, square-rooted (`--class-weight-power
0.5`), so a class with a quarter of the images weighs 2× rather than 4× — the thin classes
are also the noisiest. Disable with `--no-class-weights`. `--mixup` / `--cutmix` /
`--mix-prob` blend pairs of training images and their labels; they are off by default and
on in `make train-gpu`.
Augmentation (flip, rotate, zoom, translate, contrast, brightness) is applied in-graph
during training only, and is **excluded from the export** — `tfjs-layers` has no kernels for
`RandomFlip` and friends, so a `model.json` containing them fails to load in the browser
with "Unknown layer". `train.py` therefore builds two models that share every weight: an
`inference_model` that gets exported, wrapped by a `train_model` that adds augmentation.

`python ml/train.py --help` documents every flag. The ones you will actually reach for:

```bash
python ml/train.py --epochs 30 --fine-tune-epochs 15      # more training
python ml/train.py --batch-size 8                         # less memory
python ml/train.py --image-size 160 --alpha 0.75          # smaller, faster, less accurate
python ml/train.py --no-fine-tune                         # stage 1 only
python ml/train.py --mixed-precision                      # ~2x on a modern NVIDIA GPU
python ml/train.py --backbone efficientnetv2b0            # the stronger network (GPU advised)
python ml/train.py --quantize float16                     # half-size browser download
python ml/train.py --resume                               # continue from the best checkpoint
python ml/train.py --no-export                            # train now, convert later
```

### 5. Use it

The export goes to `models/custom/`, which is bind-mounted into the running backend at
`/models`. **Restart nothing.** Click *Reload model* in EcoSort's settings panel, or just
reload the page. The badge flips from `fallback` to `custom`.

### 6. Check it

```bash
make evaluate                                 # the test split - the honest number
python ml/evaluate.py                         # classification report + confusion matrix (val)
python ml/evaluate.py --image ~/Pictures/bottle.jpg
```

**Score the test split, not val.** `ModelCheckpoint` and `EarlyStopping` both select on
`val_accuracy`, so by the end of training val is no longer held out and its score is
optimistic. `make evaluate` defaults to `--split test` for that reason.

`--image` prints the top-3 with probabilities using exactly the browser's three steps —
centre-crop to the short edge, bilinear resize with half-pixel centres, then `/255`. If the
app disagrees with this output on the same photo, the bug is in the model plumbing, not in
preprocessing. That cross-check is worth running once after every training run: it is the
only thing that proves your measured accuracy describes what users actually get.

---

## What training writes

```
models/custom/
  model.json          # TFJS layers model - load with tf.loadLayersModel
  group1-shard*.bin   # weights, ~9 MB total at alpha 1.0
  metadata.json       # classes[], inputRange [0,1], outputActivation "softmax", classOffset 0

ml/artifacts/
  ecosort_mobilenetv2.keras          # the inference model, for re-export or evaluate.py
  checkpoints/best.keras             # best val_accuracy during the run
  training_log.csv                   # per-epoch metrics, both stages
  training_curves.png                # accuracy + loss, fine-tune boundary marked
  confusion_matrix_val.png           # counts
  confusion_matrix_val_normalized.png# row-normalised (read this one)
  confusion_matrix_val.csv
  classification_report_val.txt|json # per-class precision / recall / F1
  summary.json                       # everything above, machine-readable
  tensorboard/<run>/                 # tensorboard --logdir ml/artifacts/tensorboard
```

The same `*_test.*` files appear when `ml/dataset/test/` has images.

---

## Where to get images

None of these are downloaded for you — they have their own licences and several need a
Kaggle account. Fetch them by hand into `ml/source/<class>/`.

### RealWaste — the only open source of `textile`

<https://archive.ics.uci.edu/dataset/908/realwaste> (CC BY 4.0,
doi:10.24432/C5SS4G). 4,752 photos of real items on a landfill sorting line, 524×524, in
nine folders that map onto eight EcoSort classes (`Food Organics` and `Vegetation` both go to
`organic`). With TrashNet it fills everything the Kaggle customwaste set lacks: `cardboard`,
`textile` and `trash`. `make dataset` downloads, verifies and ingests it; the mapping is
`SOURCE_MAPS["realwaste"]` in `ingest_sources.py`.

### TrashNet — the obvious starting point

<https://github.com/garythung/trashnet> (MIT). ~2,527 photos of single items on a white
background, already organised into exactly the folder layout `prepare_dataset.py` wants:

| TrashNet class | images | EcoSort class |
|---|---|---|
| `cardboard` | ~403 | `cardboard` |
| `glass` | ~501 | `glass` |
| `metal` | ~410 | `metal` |
| `paper` | ~594 | `paper` |
| `plastic` | ~482 | `plastic` |
| `trash` | ~137 | `trash` |

**It covers 6 of the 10 categories.** There is nothing in TrashNet for `organic`,
`ewaste`, `hazardous` or `textile` — you must supply those yourself, or the model will
simply never predict them. Its white-background studio look also does not resemble a phone
photo of your kitchen bin, so mix in your own pictures before trusting the val score.

### Kaggle "Garbage Classification (12 classes)"

`mostafaabla/garbage-classification` — ~15k images across `battery`, `biological`,
`brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`,
`shoes`, `trash`, `white-glass`. This is the one that fills TrashNet's gaps:

* `battery` → `hazardous`
* `biological` → `organic`
* `clothes` + `shoes` → `textile` (merge both folders)
* `brown-glass` + `green-glass` + `white-glass` → `glass` (merge all three)

Still nothing for `ewaste`.

### Kaggle "Waste Classification data"

`techsash/waste-classification-data` — ~25k images labelled only `O` (organic) and `R`
(recyclable). Useful as bulk `organic` material; the `R` half is too coarse to map onto a
single EcoSort class, so ignore it.

### Kaggle "E-Waste Image Dataset"

Several exist (search "e-waste image dataset"); they cover phones, laptops, cables, PCBs
and small appliances. Any of them fills `ewaste`. Photographing your own drawer of old
chargers works just as well and matches the app's real input better.

### TACO — Trash Annotations in Context

<http://tacodataset.org> (CC BY 4.0). ~1,500 photos of litter in the wild with COCO-style
**segmentation masks** over 60 categories. Not drop-in: you have to crop objects out of the
annotations before it is usable as a classification set. Worth the effort for realism,
because unlike TrashNet these are real photographs of rubbish where it actually lies.

### Your own photos

The highest-value images by a wide margin. The app is used on a phone, pointed at a bin, in
kitchen lighting. Fifty photos taken that way are worth several hundred studio shots.

---

## How long does it take?

Measured on an Intel Core i5-10400 (6 cores, 12 threads), 9,235 training images, batch 32:
EfficientNetV2-B0 at 224 px took **~4.5 min per stage-1 epoch** and **~14 min per stage-2
epoch** (whole network unfrozen) — the 3 + 7 epoch run above took 1 h 45 min. That is why
the full recipe is `make train-gpu`.

Roughly, for MobileNetV2 on ~2,500 images at 224×224, batch 32, 20 + 10 epochs:

| hardware | stage 1 / epoch | stage 2 / epoch | total |
|---|---|---|---|
| modern 8-core CPU | 2 – 5 min | 4 – 10 min | **1.5 – 3.5 hours** |
| NVIDIA RTX 3060 / 4060 | 10 – 25 s | 20 – 45 s | **10 – 20 min** |
| Apple M-series (metal) | 30 – 60 s | 60 – 120 s | **30 – 60 min** |

Stage 2 is slower because gradients flow through the backbone. The first epoch of each
stage is always slower — that is the dataset being decoded and cached.

To get a usable model quickly on a CPU:

```bash
python ml/train.py --image-size 160 --alpha 0.75 --epochs 8 --fine-tune-epochs 4
```

That costs a few points of accuracy and roughly halves the time. The browser reads the
input size from `metadata.json`, so a 160×160 model works in the app with no code change.

---

## Converting a pretrained classifier

`ml/convert_pretrained.py` is the other way to get a model, and it needs no dataset at all.
It takes a pretrained ImageNet classifier out of `tf.keras.applications`, wraps it so it
honours the `[0, 1]` rule above, converts it to TensorFlow.js and writes it into
`models/<arch lowercased>/`, where EcoSort's model picker finds it by itself.

It does **not** make EcoSort better at waste: it makes it better at *objects*. The waste
answer still comes from the 393-entry ImageNet → waste map in
`frontend/src/lib/imagenetWasteMap.js`, and that map is as lopsided as ImageNet's label set
— 79 entries for `textile`, 72 for `ewaste`, 9 for `paper`, 3 for `cardboard`. Naming the
object correctly more often is a genuine improvement, but a model trained on the ten
categories skips the translation entirely and still beats all of them at a fraction of the
download. Convert while you are collecting images; train when you have them.

```bash
make list-models                                   # the supported architectures
make convert-model                                 # InceptionResNetV2 + float16 (the defaults)
make convert-model ARCH=InceptionV3                # smaller, still much better than MobileNet
make convert-model ARCH=Xception QUANTIZE=uint8    # quarters the download, at a small accuracy cost
```

The Make targets are thin wrappers, so the script works directly too — useful for the flags
Make does not surface:

```bash
python ml/convert_pretrained.py --list
python ml/convert_pretrained.py --arch InceptionV3 --quantize uint8 --force
python ml/convert_pretrained.py --arch MobileNetV3Large --out models/mnv3-uint8 --quantize uint8
```

| flag | default | what it does |
|---|---|---|
| `--arch` | `InceptionResNetV2` | which architecture to convert |
| `--out` | `models/<arch lowercased>` | where to write it |
| `--image-size` | the architecture's default (299 or 224) | rarely useful: with the classifier head attached, `keras.applications` refuses any size but its default when loading ImageNet weights |
| `--quantize` | `float16` | `none` / `float16` / `uint8` |
| `--force` | off | overwrite an existing export (otherwise it stops and says so) |
| `--list` | — | print the architectures and exit |

### Why the list is restricted

```
InceptionResNetV2  299  ~80.3 % top-1, and by far the largest
InceptionV3        299  noticeably better than MobileNet at a third of Inception-ResNet's size
Xception           299  depthwise-separable Inception; accuracy close to Inception v3
MobileNetV2        224  the bundled fallback's architecture, rebuilt from Keras weights
MobileNetV3Large   224  a little more accurate than V2 at a similar size
NASNetMobile       224  compact, competitive with MobileNetV3
```

ResNet, VGG and DenseNet are **deliberately** missing. `keras.applications` models do not
preprocess their own input — you are expected to call `preprocess_input` first — and the six
above all use Keras "tf" mode, which is `x/127.5 - 1` over `[0,255]`, i.e. exactly
`[0,1] → [-1,1]`. That is expressible as a single layer, so the converter bakes it in:

```python
inputs  = tf.keras.Input(shape=(size, size, 3))
x       = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(inputs)   # [0,1] -> [-1,1]
outputs = base(x, training=False)
```

The browser then divides by 255 and does nothing else — the same single preprocessing path
as `train.py`'s export and the bundled fallback. ResNet and VGG use "caffe" mode (BGR
channel swap plus per-channel mean subtraction) and DenseNet uses "torch" mode (ImageNet
mean/std); neither collapses into one `Rescaling`. Supporting them would mean a second
preprocessing path in `classifier.js` whose failure mode is silent: the model loads, runs,
and is confidently wrong. The script also refuses anything whose head is not 1000 (or 1001)
units, because EcoSort maps ImageNet-1k indices and nothing else.

### Quantization

| `--quantize` | download | cost |
|---|---|---|
| `float16` *(default)* | **half** of float32 | no measurable accuracy loss |
| `uint8` | **a quarter** of float32 | a small, usually invisible, loss |
| `none` | full float32 | none, and InceptionResNetV2 becomes a ~214 MB download |

The script prints the real size when it finishes and writes the exact byte count into
`metadata.json` as `downloadBytes`, which is the number the picker warns you about.
InceptionResNetV2 at float16 is 112,262,070 bytes across 27 shards plus the manifest.

### How long it takes, and the one network access

A conversion is a minute or two on a modern CPU — a SavedModel export followed by
`tensorflowjs_converter`; there is no training and no GPU work worth speaking of. Budget a
couple of GB of RAM for the Inception family.

The first conversion of a given architecture also downloads the Keras ImageNet weights:
**96 MB** for InceptionV3, **225 MB** for InceptionResNetV2, less for the MobileNets. They
land in `~/.keras/models/` and are cached there forever, so re-converting the same
architecture — to try a different `--quantize`, say — needs no network at all. As with
training, this happens on your machine, before the fact; EcoSort itself never calls out at
runtime.

### Where the output goes

```
models/inceptionresnetv2/
  model.json          # TFJS *graph* model - load with tf.loadGraphModel
  group1-shard*.bin   # 27 shards at float16
  metadata.json       # inputSize 299, inputRange [0,1], classOffset 0, labelKind "imagenet",
                      # plus displayName / description / quantization / downloadBytes
```

While it runs, the export lives in a hidden `.<arch>.staging-<pid>/` directory beside the
target and is moved into place in one step only after the shards have been verified against
the manifest — an interrupted or failed conversion therefore never leaves a half-written
model for the app to find. An existing export is left alone unless you pass `--force`.

`./models` is bind-mounted into the backend, so the new model appears in the picker within
the 5-second model-status cache. **Restart nothing**, exactly as with `models/custom/`. To
`classifier.js` the result is indistinguishable from the bundled engine: `[0,1]` in,
`inputSize` and `classOffset` read from `metadata.json`, and no per-model code anywhere.

---

## Troubleshooting

**`ResourceExhaustedError` / the process is OOM-killed**
Lower `--batch-size` (16, then 8). Add `--no-cache` so the decoded dataset streams from
disk instead of living in RAM — `train.py` prints its cache estimate at startup, and at
224×224 float32 each 1,000 images costs about 570 MiB. Fine-tuning needs more memory than
stage 1; raising `--fine-tune-at` unfreezes less of the backbone.

**`pip install` takes forever, or `tensorflowjs` and `tensorflow` fight**
They are version-coupled: `tensorflowjs` 4.x expects TensorFlow 2.15/2.16, which is why
`requirements.txt` bounds it at `<2.17`. Install them into a **separate venv** from the
rest of your Python work and let pip resolve them together. If pip starts backtracking
through dozens of versions, delete the venv and start again rather than waiting it out.

**The export fails but training succeeded**
Nothing is lost — the model is at `ml/artifacts/ecosort_mobilenetv2.keras`. Convert it on
its own (the error message prints the exact command):

```bash
python ml/export_tfjs.py --model ml/artifacts/ecosort_mobilenetv2.keras \
                         --out models/custom \
                         --classes cardboard,ewaste,glass,hazardous,metal,organic,paper,plastic,textile,trash
```

On TensorFlow 2.16 (Keras 3) the converter sometimes needs `TF_USE_LEGACY_KERAS=1` in the
environment. `export_tfjs.py` already falls back from the Python API to the
`tensorflowjs_converter` CLI by itself, via both HDF5 and `.keras`.

**Validation accuracy is poor**
Open `ml/artifacts/confusion_matrix_val_normalized.png` first — it tells you *which* classes
are being confused, and that usually names the fix:

* A whole row near zero → that class has too few images. The floor is ~100.
* Two classes swapping → they genuinely look alike in your photos (glass vs plastic
  bottles, paper vs cardboard). Add images that show the difference you care about.
* Great training accuracy, poor validation accuracy → overfitting. More data, or
  `--dropout 0.5`, or fewer epochs.
* Everything mediocre → try `--epochs 40 --fine-tune-epochs 20`, and check that
  `ml/dataset/train` actually contains what you think it does.

**Validation accuracy is suspiciously perfect**
Near-duplicate photos of the same object split across train and val leak the answer. The
splitter shuffles files, it cannot detect duplicates. Photograph different objects.

**The app still shows `fallback` after training**
Check `models/custom/model.json` exists on the **host** — that directory is bind-mounted,
so the container sees it immediately. Then `curl localhost:4000/api/model/status`, which
reports exactly what it found and every path it searched.

**Runs are not reproducible**
`--seed` seeds Python, NumPy and TensorFlow, which is enough on CPU. Bit-exact results on
GPU additionally need `TF_DETERMINISTIC_OPS=1` in the environment; `train.py` does not set
it for you because a handful of kernels have no deterministic implementation and would
raise instead of run.

---

## File map

| file | what it does |
|---|---|
| `build_dataset.py` | one command: download RealWaste + TrashNet (verified), merge with customwaste, split — `make dataset` |
| `ingest_sources.py` | `--inspect` a downloaded dataset; merge several into `source/` with de-collision, dedup and per-class caps |
| `prepare_dataset.py` | `--scaffold` the class folders; validate, stratify and split images into `dataset/` |
| `train.py` | two-stage transfer learning (MobileNetV2 or EfficientNetV2), artifacts, and the TFJS export |
| `gpu_check.py` | whether TensorFlow can train on an NVIDIA GPU here, and if not, why — `make gpu-check` |
| `export_tfjs.py` | Keras → TensorFlow.js conversion + `metadata.json`; also a standalone CLI |
| `convert_pretrained.py` | pretrained `keras.applications` ImageNet net → TensorFlow.js in `models/<arch>/` |
| `evaluate.py` | classification report, confusion matrix, and single-image top-3 with browser-identical preprocessing |
| `requirements.txt` | pinned-by-range dependency set, with the version-coupling notes |
