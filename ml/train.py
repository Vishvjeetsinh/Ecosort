#!/usr/bin/env python3
"""EcoSort - two-stage MobileNetV2 transfer-learning trainer, exported to TensorFlow.js.

    python ml/prepare_dataset.py --scaffold     # 1. make the class folders
    #  ... put images in ml/source/<class>/ ...
    python ml/prepare_dataset.py                # 2. split into ml/dataset/{train,val,test}
    python ml/train.py                          # 3. train + export to models/custom/
    python ml/train.py --smoke-test             # (no data needed: verifies the toolchain)

==============================================================================
THE ONE THING THAT MUST NOT BE GOT WRONG
==============================================================================
The exported model takes pixels in **[0, 1]**.

`frontend/src/lib/classifier.js` has exactly ONE preprocessing path shared by the custom
model and the pretrained MobileNetV2 fallback: resize to 224x224, `toFloat()`, `div(255)`.
Nothing else. That is only correct because this script puts

    tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)

*inside* the exported graph, which converts [0, 1] -> [-1, 1], the range MobileNetV2 was
pretrained on. It mirrors the `hub_input/Mul(2.0)` + `hub_input/Sub(1.0)` pair that the
pretrained graph model carries internally (docs/ARCHITECTURE.md section 2.1).

Consequences, all of them load-bearing:
  * NEVER call `tf.keras.applications.mobilenet_v2.preprocess_input` in the data pipeline.
    It does x/127.5 - 1 on [0,255] data; doing that *and* the in-graph Rescaling would
    double-apply the shift and wreck accuracy in a way that still trains happily.
  * The tf.data pipeline therefore does one single thing to pixels: `x / 255.0`.
  * `ml/evaluate.py` deliberately preprocesses the same way, so a mismatch shows up as a
    disagreement between Python and the browser rather than as mysterious bad predictions.

==============================================================================
WHY THERE ARE TWO KERAS MODELS
==============================================================================
`inference_model`  Input -> Rescaling -> MobileNetV2 -> GAP -> Dropout -> Dense(softmax)
`train_model`      Input -> augmentation -> inference_model

They share the same layer objects, so training either one trains both, but only
`inference_model` is exported. That is not cosmetic: tfjs-layers has no kernels for
`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomTranslation`, `RandomContrast` or
`RandomBrightness`, so a model.json containing them fails to load in the browser with
"Unknown layer". Keeping augmentation in a wrapper model is what keeps the export loadable.

The augmentation layers are also no-ops at inference time by design (they only perturb when
`training=True`), so the two models agree on every prediction anyway.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Must be set before TensorFlow is imported, or it has no effect.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from export_tfjs import ExportError, export_to_tfjs  # noqa: E402  (must follow sys.path setup)

# The scientific stack is imported lazily by load_stack() so that `--help` and argument
# errors still work in an environment where TensorFlow was never installed. Module-level
# functions read these as globals, which Python resolves at call time.
np = tf = keras = plt = None
classification_report = confusion_matrix = None


def load_stack() -> None:
    """Import numpy/TensorFlow/matplotlib/sklearn, or exit with an actionable message."""
    global np, tf, keras, plt, classification_report, confusion_matrix
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: we only ever write PNGs
        import matplotlib.pyplot as _plt
        import numpy as _np
        import tensorflow as _tf
        from sklearn.metrics import classification_report as _report
        from sklearn.metrics import confusion_matrix as _matrix
        from tensorflow import keras as _keras
    except ImportError as exc:
        sys.stderr.write(
            f"error: a training dependency is missing ({exc}).\n"
            "       python3 -m venv ml/.venv && . ml/.venv/bin/activate\n"
            "       pip install -U pip && pip install -r ml/requirements.txt\n"
        )
        raise SystemExit(2) from exc

    np, tf, keras, plt = _np, _tf, _keras, _plt
    classification_report, confusion_matrix = _report, _matrix

CANONICAL_CLASSES = sorted(
    [
        "plastic",
        "paper",
        "cardboard",
        "glass",
        "metal",
        "organic",
        "ewaste",
        "hazardous",
        "textile",
        "trash",
    ]
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# MobileNetV2 ImageNet weights only exist for these width multipliers.
VALID_ALPHAS = (0.35, 0.50, 0.75, 1.00, 1.30, 1.40)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ArgumentDefaultsHelpFormatter, minus the useless "(default: None)" noise."""

    def _get_help_string(self, action):
        # None and plain False add nothing: "--no-cache (default: False)" is noise.
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Two-stage MobileNetV2 transfer learning for the EcoSort waste classifier, "
            "exported to TensorFlow.js so the browser can run it offline."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "Determinism: python/numpy/TensorFlow are all seeded from --seed, which makes "
            "runs reproducible on CPU. Bit-exact reproducibility on GPU additionally needs "
            "TF_DETERMINISTIC_OPS=1 in the environment (it is not set here because a few "
            "kernels have no deterministic implementation and would raise instead of run)."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--data-dir",
        default=None,
        help="Dataset root holding train/ val/ [test/] and dataset.json (default: ml/dataset)",
    )
    paths.add_argument(
        "--out-dir",
        default=None,
        help="Where the TensorFlow.js export is written (default: models/custom)",
    )
    paths.add_argument(
        "--artifacts-dir",
        default=None,
        help="Checkpoints, plots, logs and reports (default: ml/artifacts)",
    )

    data = parser.add_argument_group("data")
    data.add_argument("--image-size", type=int, default=224, help="Square input side length in pixels")
    data.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for both stages")
    data.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not cache the decoded dataset in RAM (use this if the dataset does not fit)",
    )

    train = parser.add_argument_group("training")
    train.add_argument("--epochs", type=int, default=20, help="Stage 1 epochs (frozen backbone)")
    train.add_argument(
        "--fine-tune-epochs", type=int, default=10, help="Stage 2 epochs (partially unfrozen backbone)"
    )
    train.add_argument(
        "--fine-tune-at",
        type=int,
        default=100,
        help="Index of the first MobileNetV2 layer to unfreeze in stage 2 (of ~154)",
    )
    train.add_argument("--lr", type=float, default=1e-3, help="Stage 1 Adam learning rate")
    train.add_argument(
        "--fine-tune-lr",
        type=float,
        default=1e-5,
        help="Stage 2 Adam learning rate - must be tiny or fine-tuning destroys the features",
    )
    train.add_argument("--dropout", type=float, default=0.3, help="Dropout rate before the classifier head")
    train.add_argument(
        "--label-smoothing", type=float, default=0.05, help="CategoricalCrossentropy label smoothing"
    )
    train.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=f"MobileNetV2 width multiplier; ImageNet weights exist only for {VALID_ALPHAS}",
    )
    train.add_argument("--seed", type=int, default=42, help="Seed for python, numpy, TensorFlow and shuffling")
    train.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting (on by default; helps imbalanced sets)",
    )
    train.add_argument("--no-fine-tune", action="store_true", help="Stop after stage 1")
    train.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed_float16 - roughly 2x faster on a modern NVIDIA GPU, pointless on CPU",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Start from the weights in <artifacts-dir>/checkpoints/best.keras if it exists. "
            "Weights only - the epoch counter and optimiser state restart from zero."
        ),
    )

    output = parser.add_argument_group("output")
    output.add_argument("--no-export", action="store_true", help="Train but skip the TensorFlow.js conversion")
    output.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Ignore --data-dir entirely: synthesise 4 classes x 8 random-noise images, run one "
            "epoch per stage at batch 4 and still perform the full export. Verifies the whole "
            "toolchain - TensorFlow, Keras and the tensorflowjs converter - without any data."
        ),
    )
    return parser.parse_args(argv)


def resolve_paths(args):
    args.data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else REPO_ROOT / "ml" / "dataset"
    args.artifacts_dir = (
        Path(args.artifacts_dir).expanduser().resolve() if args.artifacts_dir else REPO_ROOT / "ml" / "artifacts"
    )
    explicit_out = args.out_dir is not None
    args.out_dir = Path(args.out_dir).expanduser().resolve() if explicit_out else REPO_ROOT / "models" / "custom"

    # A model trained on random noise must never land in the slot the app loads from.
    if args.smoke_test and not explicit_out:
        args.out_dir = args.artifacts_dir / "smoke-export"
    return args


def validate_args(args):
    problems = []
    if args.image_size < 32:
        problems.append("--image-size must be at least 32 (MobileNetV2 minimum)")
    if args.batch_size < 1:
        problems.append("--batch-size must be >= 1")
    if args.epochs < 1:
        problems.append("--epochs must be >= 1")
    if args.fine_tune_epochs < 0:
        problems.append("--fine-tune-epochs must be >= 0")
    if not 0.0 <= args.dropout < 1.0:
        problems.append("--dropout must be in [0, 1)")
    if not 0.0 <= args.label_smoothing < 1.0:
        problems.append("--label-smoothing must be in [0, 1)")
    if args.lr <= 0 or args.fine_tune_lr <= 0:
        problems.append("--lr and --fine-tune-lr must be > 0")
    if round(args.alpha, 2) not in VALID_ALPHAS:
        problems.append(
            f"--alpha {args.alpha} has no ImageNet weights; pick one of {VALID_ALPHAS}"
        )
    if args.fine_tune_at < 0:
        problems.append("--fine-tune-at must be >= 0")
    if problems:
        raise SystemExit("error: " + "\n       ".join(problems))


# --------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    setter = getattr(keras.utils, "set_random_seed", None)
    if callable(setter):
        setter(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
def read_class_names(data_dir: Path):
    """Label order comes from dataset.json when it exists - never re-derived twice."""
    manifest = data_dir / "dataset.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"error: {manifest} is unreadable: {exc}") from exc
        classes = payload.get("classes")
        if not isinstance(classes, list) or not classes:
            raise SystemExit(f"error: {manifest} has no non-empty `classes` array.")
        return [str(c) for c in classes], str(manifest)

    train_dir = data_dir / "train"
    if not train_dir.is_dir():
        raise SystemExit(
            f"error: no dataset at {data_dir}.\n"
            "       Build one with:  python ml/prepare_dataset.py --scaffold\n"
            "       then drop images in ml/source/<class>/ and run: python ml/prepare_dataset.py\n"
            "       Or verify the toolchain without data:  python ml/train.py --smoke-test"
        )
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not classes:
        raise SystemExit(f"error: {train_dir} has no class sub-directories.")
    return classes, f"{train_dir} (sorted sub-directory names; no dataset.json found)"


def check_split_dirs(data_dir: Path, class_names):
    """Every split that exists must contain every class folder, or Keras raises later."""
    for split in ("train", "val"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise SystemExit(
                f"error: {split_dir} does not exist. Run `python ml/prepare_dataset.py` first."
            )
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if not split_dir.is_dir():
            continue
        missing = [c for c in class_names if not (split_dir / c).is_dir()]
        if missing:
            raise SystemExit(
                f"error: {split_dir} is missing class folder(s): {', '.join(missing)}.\n"
                "       The class list and the directory tree must agree exactly. "
                "Re-run ml/prepare_dataset.py."
            )


def count_files_per_class(split_dir: Path, class_names):
    counts = {}
    for cls in class_names:
        class_dir = split_dir / cls
        if not class_dir.is_dir():
            counts[cls] = 0
            continue
        counts[cls] = sum(
            1 for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    return counts


def prepare_pipeline(ds, training: bool, cache: bool, seed: int, shuffle_buffer: int = 1000):
    """The ONLY pixel transform here is /255 - see the module docstring."""
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    if training:
        # After cache(), so the order is genuinely re-shuffled every epoch.
        ds = ds.shuffle(shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_real_datasets(args, class_names):
    common = dict(
        labels="inferred",
        label_mode="categorical",
        class_names=class_names,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        interpolation="bilinear",
        # The browser centre-crops to the short edge before it resizes
        # (frontend/src/lib/imageUtils.js canvasFromSource). The Keras default here is
        # False, which SQUASHES the full frame to a square instead - so without this the
        # model trains on stretched images and is served cropped ones. On a 16:9 webcam
        # frame that is ~44% of the width it never saw. Keep this in step with
        # ml/evaluate.py, which loads the same way.
        crop_to_aspect_ratio=True,
    )
    try:
        train_raw = keras.utils.image_dataset_from_directory(
            args.data_dir / "train", shuffle=True, seed=args.seed, **common
        )
        val_raw = keras.utils.image_dataset_from_directory(
            args.data_dir / "val", shuffle=False, **common
        )
    except Exception as exc:  # noqa: BLE001 - Keras raises several unrelated types here
        raise SystemExit(
            f"error: could not load the dataset from {args.data_dir}: {type(exc).__name__}: {exc}"
        ) from exc

    test_dir = args.data_dir / "test"
    test_raw = None
    if test_dir.is_dir():
        test_counts = count_files_per_class(test_dir, class_names)
        if sum(test_counts.values()) > 0:
            test_raw = keras.utils.image_dataset_from_directory(test_dir, shuffle=False, **common)

    cache = not args.no_cache
    train_counts = count_files_per_class(args.data_dir / "train", class_names)
    n_train = sum(train_counts.values())

    # The shuffle happens after batching, so its buffer counts BATCHES, not images -
    # a buffer of a few thousand here would hold gigabytes of decoded pixels.
    buffer = max(8, min(64, math.ceil(n_train / args.batch_size) or 1))

    if cache:
        cache_mib = n_train * args.image_size * args.image_size * 3 * 4 / (1024 ** 2)
        print(f"Cache estimate: {cache_mib:,.0f} MiB of decoded float32 pixels held in RAM")
        if cache_mib > 2048:
            print(
                "warning: that is a lot of RAM. Pass --no-cache to stream from disk instead "
                "(slower per epoch, but it will not be killed by the OOM reaper)."
            )

    datasets = {
        "train": prepare_pipeline(train_raw, True, cache, args.seed, buffer),
        "val": prepare_pipeline(val_raw, False, cache, args.seed),
        "test": prepare_pipeline(test_raw, False, cache, args.seed) if test_raw is not None else None,
    }
    return datasets, train_counts


def make_synthetic_datasets(args, class_names):
    """Random-noise stand-in used by --smoke-test. Same shapes, same dtypes, no disk."""
    n_classes = len(class_names)
    rng = np.random.default_rng(args.seed)

    def build(per_class):
        total = per_class * n_classes
        images = rng.integers(
            0, 256, size=(total, args.image_size, args.image_size, 3), dtype=np.int16
        ).astype("float32")
        labels = np.repeat(np.arange(n_classes), per_class)
        one_hot = np.eye(n_classes, dtype="float32")[labels]
        ds = tf.data.Dataset.from_tensor_slices((images, one_hot)).batch(args.batch_size)
        return ds

    datasets = {
        "train": prepare_pipeline(build(8), True, True, args.seed, 32),
        "val": prepare_pipeline(build(4), False, True, args.seed),
        "test": None,
    }
    counts = {cls: 8 for cls in class_names}
    return datasets, counts


def compute_class_weights(counts, class_names):
    """Inverse-frequency weights: total / (n_classes * count). Classes with 0 images get 0."""
    total = sum(counts.values())
    n_classes = len(class_names)
    if total == 0:
        raise SystemExit("error: the training split contains zero images.")
    weights = {}
    for index, cls in enumerate(class_names):
        count = counts.get(cls, 0)
        weights[index] = 0.0 if count == 0 else total / (n_classes * count)
    return weights


# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
def _accepts(cls, kwarg: str) -> bool:
    try:
        return kwarg in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def build_augmentation(seed: int):
    """In-graph augmentation, active only when the enclosing call has training=True.

    Order matters: this runs on [0, 1] pixels *before* the Rescaling layer, because
    RandomBrightness clips to its `value_range` and RandomContrast perturbs around the
    per-image mean. Both behave sensibly on [0, 1] and badly on [-1, 1].
    """
    L = keras.layers

    contrast_kwargs = {"factor": 0.15, "seed": seed, "name": "aug_contrast"}
    if _accepts(L.RandomContrast, "value_range"):
        contrast_kwargs["value_range"] = (0.0, 1.0)

    brightness_kwargs = {"factor": 0.15, "seed": seed, "name": "aug_brightness"}
    if _accepts(L.RandomBrightness, "value_range"):
        brightness_kwargs["value_range"] = (0.0, 1.0)

    return keras.Sequential(
        [
            L.RandomFlip("horizontal", seed=seed, name="aug_flip"),
            L.RandomRotation(0.15, fill_mode="reflect", seed=seed, name="aug_rotation"),
            L.RandomZoom(0.15, 0.15, fill_mode="reflect", seed=seed, name="aug_zoom"),
            L.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed, name="aug_translation"),
            L.RandomContrast(**contrast_kwargs),
            L.RandomBrightness(**brightness_kwargs),
        ],
        name="augmentation",
    )


def build_models(num_classes: int, image_size: int, alpha: float, dropout: float, seed: int):
    """Return (inference_model, train_model, base_model). They share every weight."""
    shape = (image_size, image_size, 3)

    base = keras.applications.MobileNetV2(
        input_shape=shape, include_top=False, weights="imagenet", alpha=alpha
    )
    base.trainable = False

    inputs = keras.Input(shape=shape, name="image")
    # [0,1] -> [-1,1]. This layer IS the browser contract; see the module docstring.
    x = keras.layers.Rescaling(scale=2.0, offset=-1.0, name="rescale_unit_to_signed")(inputs)
    # training=False pins the backbone's BatchNorm to inference statistics for the whole
    # run, including stage 2 - the documented way to fine-tune MobileNetV2 without the
    # moving averages drifting and destroying the pretrained features.
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="embedding")(x)
    x = keras.layers.Dropout(dropout, seed=seed, name="head_dropout")(x)
    # float32 output even under mixed_float16: softmax in fp16 overflows easily.
    outputs = keras.layers.Dense(
        num_classes, activation="softmax", dtype="float32", name="predictions"
    )(x)
    inference_model = keras.Model(inputs, outputs, name="ecosort_mobilenetv2")

    aug_inputs = keras.Input(shape=shape, name="image")
    augmentation = build_augmentation(seed)
    train_model = keras.Model(
        aug_inputs, inference_model(augmentation(aug_inputs)), name="ecosort_mobilenetv2_train"
    )
    return inference_model, train_model, base


def build_metrics(num_classes: int):
    metrics = ["accuracy"]
    # TopKCategoricalAccuracy(k=2) is meaningless - and always 1.0 - with two classes.
    if num_classes > 2:
        metrics.append(keras.metrics.TopKCategoricalAccuracy(k=2, name="top2_accuracy"))
    return metrics


def compile_model(model, lr: float, label_smoothing: float, num_classes: int):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
        metrics=build_metrics(num_classes),
    )


def unfreeze_for_fine_tuning(base, fine_tune_at: int) -> int:
    """Unfreeze base.layers[fine_tune_at:], but keep every BatchNormalization frozen.

    Leaving BN trainable is the classic MobileNetV2 fine-tuning trap: the tiny batches used
    here produce noisy statistics, the moving averages drift away from the ImageNet ones,
    and val accuracy collapses while train accuracy keeps climbing.
    """
    base.trainable = True
    cutoff = min(fine_tune_at, len(base.layers))
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    frozen_bn = 0
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn += 1
    trainable = sum(1 for layer in base.layers if layer.trainable)
    print(
        f"  unfroze {trainable}/{len(base.layers)} backbone layers from index {cutoff}; "
        f"{frozen_bn} BatchNormalization layers stay frozen"
    )
    return trainable


def build_callbacks(checkpoint_path: Path, log_path: Path, tensorboard_dir: Path, append: bool):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
        keras.callbacks.CSVLogger(str(log_path), append=append),
        keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir), histogram_freq=0),
    ]


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------
def merge_histories(*histories):
    merged = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)
    return merged


def plot_training_curves(history: dict, boundary: int, path: Path) -> None:
    pairs = [("accuracy", "val_accuracy", "Accuracy"), ("loss", "val_loss", "Loss")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(12, 4.5))
    for ax, (train_key, val_key, title) in zip(np.atleast_1d(axes), pairs):
        epochs = range(1, len(history.get(train_key, [])) + 1)
        if history.get(train_key):
            ax.plot(epochs, history[train_key], label=f"train {title.lower()}", linewidth=1.8)
        if history.get(val_key):
            ax.plot(
                range(1, len(history[val_key]) + 1),
                history[val_key],
                label=f"val {title.lower()}",
                linewidth=1.8,
            )
        if 0 < boundary < len(history.get(train_key, [])):
            ax.axvline(boundary + 0.5, color="#dc2626", linestyle="--", linewidth=1, label="fine-tune starts")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("EcoSort - MobileNetV2 transfer learning")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, title: str, path: Path, normalize: bool) -> None:
    data = matrix.astype("float64")
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)

    size = max(5.0, 0.7 * len(class_names) + 2.5)
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(data, interpolation="nearest", cmap="Blues", vmin=0, vmax=data.max() or 1)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    threshold = (data.max() or 1) / 2.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = f"{data[i, j]:.2f}" if normalize else f"{int(matrix[i, j])}"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[i, j] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def collect_predictions(model, dataset):
    """Return (y_true, y_pred, y_prob) as plain numpy, in dataset order."""
    probabilities = model.predict(dataset, verbose=0)
    true_batches = [batch.numpy() for _, batch in dataset.unbatch().batch(1024)]
    y_true = np.concatenate(true_batches, axis=0).argmax(axis=1)
    y_pred = probabilities.argmax(axis=1)
    return y_true, y_pred, probabilities


def write_reports(model, dataset, class_names, artifacts_dir: Path, split: str):
    y_true, y_pred, _ = collect_predictions(model, dataset)
    if y_true.size == 0:
        print(f"  {split}: empty split, skipping report")
        return None

    labels = list(range(len(class_names)))
    text_report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
    )
    dict_report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\nPer-class metrics ({split}, {y_true.size} images)")
    print(text_report)

    (artifacts_dir / f"classification_report_{split}.txt").write_text(text_report, encoding="utf-8")
    (artifacts_dir / f"classification_report_{split}.json").write_text(
        json.dumps(dict_report, indent=2) + "\n", encoding="utf-8"
    )
    np.savetxt(artifacts_dir / f"confusion_matrix_{split}.csv", matrix, fmt="%d", delimiter=",")
    plot_confusion_matrix(
        matrix, class_names, f"Confusion matrix ({split}, counts)",
        artifacts_dir / f"confusion_matrix_{split}.png", normalize=False,
    )
    plot_confusion_matrix(
        matrix, class_names, f"Confusion matrix ({split}, row-normalised)",
        artifacts_dir / f"confusion_matrix_{split}_normalized.png", normalize=True,
    )
    # sklearn only emits an "accuracy" key when set(labels) == the labels actually present,
    # so a class with zero images in this split would turn that lookup into a KeyError.
    accuracy = dict_report.get("accuracy")
    if not isinstance(accuracy, float):
        accuracy = float((y_true == y_pred).mean())
    return {"accuracy": accuracy, "report": dict_report, "matrix": matrix.tolist()}


def metric_value(results, name: str, default: float = 0.0) -> float:
    """Read a metric out of evaluate(return_dict=True), tolerating Keras' name prefixes."""
    value = results.get(name)
    if isinstance(value, (int, float)):
        return float(value)
    for key, candidate in results.items():
        if key.endswith(name) and isinstance(candidate, (int, float)):
            return float(candidate)
    return default


def format_results(results) -> str:
    return ", ".join(
        f"{k}={v:.4f}" for k, v in results.items() if isinstance(v, (int, float))
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(argv=None) -> int:
    args = resolve_paths(parse_args(argv))
    validate_args(args)
    load_stack()
    return run(args)


def run(args) -> int:
    seed_everything(args.seed)

    if args.mixed_precision:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            print("warning: --mixed-precision requested but no GPU is visible; it will not help on CPU.")
        keras.mixed_precision.set_global_policy("mixed_float16")

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.artifacts_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.keras"
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tensorboard_dir = args.artifacts_dir / "tensorboard" / run_stamp
    csv_log = args.artifacts_dir / "training_log.csv"

    print("=" * 78)
    print("EcoSort - MobileNetV2 transfer learning")
    print("=" * 78)
    print(f"TensorFlow {tf.__version__} / Keras {keras.__version__}")
    devices = tf.config.list_physical_devices("GPU")
    print(f"GPU(s): {[d.name for d in devices] if devices else 'none - training on CPU'}")

    if args.smoke_test:
        class_names = ["glass", "metal", "paper", "plastic"]
        args.batch_size = 4
        args.epochs = 1
        args.fine_tune_epochs = 1 if not args.no_fine_tune else 0
        class_source = "synthetic (--smoke-test)"
        print("\n*** SMOKE TEST: random-noise data, 1 epoch per stage. Accuracy is meaningless. ***")
        datasets, train_counts = make_synthetic_datasets(args, class_names)
    else:
        class_names, class_source = read_class_names(args.data_dir)
        check_split_dirs(args.data_dir, class_names)
        datasets, train_counts = load_real_datasets(args, class_names)

    num_classes = len(class_names)
    if num_classes < 2:
        raise SystemExit(f"error: need at least 2 classes, found {num_classes}.")

    non_canonical = [c for c in class_names if c not in CANONICAL_CLASSES]
    if non_canonical:
        print(
            f"\nwarning: {len(non_canonical)} class id(s) are outside the canonical taxonomy "
            f"of docs/ARCHITECTURE.md section 3: {', '.join(non_canonical)}\n"
            "         The EcoSort UI has no bin guidance, colour or label for these."
        )

    print(f"\nData dir      : {args.data_dir if not args.smoke_test else '(none - synthetic)'}")
    print(f"Artifacts dir : {args.artifacts_dir}")
    print(f"Export dir    : {args.out_dir}")
    print(f"Label order   : {class_source}")
    print(f"Classes ({num_classes}) : {', '.join(class_names)}")
    print(f"Train images  : {sum(train_counts.values())}  " + ", ".join(f"{k}={v}" for k, v in train_counts.items()))

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_counts, class_names)
        formatted = ", ".join(f"{class_names[i]}={w:.2f}" for i, w in sorted(class_weights.items()))
        print(f"Class weights : {formatted}")

    print("\nBuilding model ...")
    inference_model, train_model, base = build_models(
        num_classes, args.image_size, args.alpha, args.dropout, args.seed
    )
    print(
        f"  MobileNetV2(alpha={args.alpha}, include_top=False) - {len(base.layers)} backbone layers, "
        f"{base.count_params():,} backbone params"
    )
    print(f"  head: GlobalAveragePooling2D -> Dropout({args.dropout}) -> Dense({num_classes}, softmax)")
    print("  first layer after Input is Rescaling(scale=2.0, offset=-1.0): the model takes [0,1] input")

    if args.resume:
        if checkpoint_path.is_file():
            print(f"\nResuming from {checkpoint_path}")
            try:
                restored = keras.models.load_model(checkpoint_path, compile=False)
                train_model.set_weights(restored.get_weights())
                del restored
            except Exception as exc:  # noqa: BLE001 - shape/architecture mismatch, corrupt file...
                raise SystemExit(
                    f"error: --resume could not restore {checkpoint_path}: {type(exc).__name__}: {exc}\n"
                    "       The checkpoint must come from an identical architecture "
                    "(same --image-size, --alpha, --dropout and class count)."
                ) from exc
        else:
            print(f"\nwarning: --resume given but {checkpoint_path} does not exist; starting from ImageNet weights.")

    # ---- Stage 1: frozen backbone -------------------------------------------------
    print("\n" + "-" * 78)
    print(f"STAGE 1/2 - frozen backbone, {args.epochs} epoch(s), lr={args.lr}")
    print("-" * 78)
    compile_model(train_model, args.lr, args.label_smoothing, num_classes)
    try:
        history1 = train_model.fit(
            datasets["train"],
            validation_data=datasets["val"],
            epochs=args.epochs,
            class_weight=class_weights,
            callbacks=build_callbacks(checkpoint_path, csv_log, tensorboard_dir / "stage1", append=False),
            verbose=1,
        )
    except tf.errors.ResourceExhaustedError as exc:
        raise SystemExit(
            f"error: out of memory during stage 1 ({exc.message if hasattr(exc, 'message') else exc}).\n"
            "       Retry with a smaller --batch-size (e.g. 16 or 8), or --no-cache, "
            "or a smaller --image-size / --alpha."
        ) from exc
    completed = (history1.epoch[-1] + 1) if history1.epoch else 0

    # ---- Stage 2: fine-tune -------------------------------------------------------
    history2 = None
    boundary = completed
    if args.no_fine_tune or args.fine_tune_epochs < 1:
        print("\nStage 2 skipped (--no-fine-tune or --fine-tune-epochs 0).")
    else:
        print("\n" + "-" * 78)
        print(f"STAGE 2/2 - fine-tuning from layer {args.fine_tune_at}, "
              f"{args.fine_tune_epochs} epoch(s), lr={args.fine_tune_lr}")
        print("-" * 78)
        unfreeze_for_fine_tuning(base, args.fine_tune_at)
        # Recompiling is mandatory: Keras caches the trainable-variable list at compile time.
        compile_model(train_model, args.fine_tune_lr, args.label_smoothing, num_classes)
        try:
            history2 = train_model.fit(
                datasets["train"],
                validation_data=datasets["val"],
                epochs=completed + args.fine_tune_epochs,
                initial_epoch=completed,
                class_weight=class_weights,
                callbacks=build_callbacks(
                    checkpoint_path, csv_log, tensorboard_dir / "stage2", append=True
                ),
                verbose=1,
            )
        except tf.errors.ResourceExhaustedError as exc:
            raise SystemExit(
                f"error: out of memory during fine-tuning ({exc}).\n"
                "       Fine-tuning needs more memory than stage 1 because gradients flow through "
                "the backbone. Retry with a smaller --batch-size, or raise --fine-tune-at."
            ) from exc
        completed = (history2.epoch[-1] + 1) if history2.epoch else completed

    # ---- Evaluation ---------------------------------------------------------------
    print("\n" + "-" * 78)
    print("EVALUATION")
    print("-" * 78)
    # Evaluate through the *exported* graph, so the numbers describe what ships.
    compile_model(inference_model, args.fine_tune_lr, args.label_smoothing, num_classes)
    val_results = inference_model.evaluate(datasets["val"], verbose=0, return_dict=True)
    print("val : " + format_results(val_results))

    test_results = None
    if datasets["test"] is not None:
        test_results = inference_model.evaluate(datasets["test"], verbose=0, return_dict=True)
        print("test: " + format_results(test_results))

    history = merge_histories(history1, history2)
    curves_path = args.artifacts_dir / "training_curves.png"
    plot_training_curves(history, boundary, curves_path)
    print(f"\nWrote {curves_path}")

    write_reports(inference_model, datasets["val"], class_names, args.artifacts_dir, "val")
    if datasets["test"] is not None:
        write_reports(inference_model, datasets["test"], class_names, args.artifacts_dir, "test")

    final_model_path = args.artifacts_dir / "ecosort_mobilenetv2.keras"
    try:
        inference_model.save(final_model_path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"error: could not save {final_model_path}: {type(exc).__name__}: {exc}") from exc
    print(f"Wrote {final_model_path}")

    metrics = {
        "valAccuracy": metric_value(val_results, "accuracy"),
        "valLoss": metric_value(val_results, "loss"),
    }
    if test_results:
        metrics["testAccuracy"] = metric_value(test_results, "accuracy")
        metrics["testLoss"] = metric_value(test_results, "loss")

    # ---- Export -------------------------------------------------------------------
    export_info = None
    if args.no_export:
        print("\nExport skipped (--no-export).")
    else:
        try:
            export_info = export_to_tfjs(
                inference_model,
                args.out_dir,
                class_names,
                metrics=metrics,
                base_model="MobileNetV2",
                image_size=args.image_size,
                epochs=completed,
            )
        except ExportError as exc:
            raise SystemExit(
                f"error: {exc}\n"
                f"       Training itself succeeded - the model is at {final_model_path}.\n"
                f"       Re-run just the conversion with:\n"
                f"         python ml/export_tfjs.py --model {final_model_path} "
                f"--out {args.out_dir} --classes {','.join(class_names)}"
            ) from exc

    summary = {
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "smokeTest": bool(args.smoke_test),
        "classes": class_names,
        "epochsCompleted": completed,
        "imageSize": args.image_size,
        "alpha": args.alpha,
        "seed": args.seed,
        "trainCounts": train_counts,
        "metrics": metrics,
        "exportDir": str(args.out_dir) if export_info else None,
        "kerasModel": str(final_model_path),
    }
    (args.artifacts_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # ---- Final summary ------------------------------------------------------------
    print("\n" + "=" * 78)
    print("DONE")
    print("=" * 78)
    print(f"Epochs run       : {completed}")
    print(f"Val accuracy     : {metrics['valAccuracy'] * 100:.2f}%   (loss {metrics['valLoss']:.4f})")
    if test_results:
        print(f"Test accuracy    : {metrics['testAccuracy'] * 100:.2f}%   (loss {metrics['testLoss']:.4f})")
    print(f"Keras model      : {final_model_path}")
    print(f"Best checkpoint  : {checkpoint_path}")
    print(f"Artifacts        : {args.artifacts_dir}")
    if export_info:
        print(f"TFJS model       : {args.out_dir}/model.json  ({len(export_info['shards'])} shard(s))")
        print(f"Metadata         : {args.out_dir}/metadata.json")
    print(f"Class order      : {', '.join(f'{i}={c}' for i, c in enumerate(class_names))}")
    print("                   (softmax index -> waste id; identical to metadata.json.classes)")

    if args.smoke_test:
        print(
            "\nSmoke test complete: TensorFlow, Keras and the TensorFlow.js converter all work.\n"
            f"The throwaway export is in {args.out_dir} - it was trained on random noise, so do\n"
            "NOT copy it into models/custom/. Train on real data next:\n"
            "  python ml/prepare_dataset.py --scaffold"
        )
    elif export_info:
        print(
            "\nNEXT STEP: restart nothing. models/ is bind-mounted into the running backend, so\n"
            "just click \"Reload model\" in EcoSort's settings panel, or reload the page.\n"
            "The model status badge should flip from 'fallback' to 'custom'."
        )
    else:
        print(
            "\nNEXT STEP: convert the saved model when you are ready:\n"
            f"  python ml/export_tfjs.py --model {final_model_path} --out {REPO_ROOT / 'models' / 'custom'} "
            f"--classes {','.join(class_names)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
