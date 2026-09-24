#!/usr/bin/env python3
"""EcoSort - evaluate a trained model, or classify one image the way the browser does.

    python ml/evaluate.py                                   # report on the val split
    python ml/evaluate.py --split test
    python ml/evaluate.py --image ~/Pictures/bottle.jpg      # top-3, like the app

WHY THE PREPROCESSING LOOKS SO PLAIN
------------------------------------
This script deliberately does exactly what the browser does and nothing more. The whole
contract is three steps:

    pixels (uint8, [0,255])
      ->  centre-crop to the short edge      (canvasFromSource, imageUtils.js)
      ->  resize bilinear to NxN, half-pixel centres, align_corners=False
                                             (resizeBilinear(x, s, false, true), classifier.js)
      ->  / 255.0                            (classifier.js runInference)

All three matter. Skip the crop and a 16:9 photo scores against pixels the app would have
thrown away; get the resize kernel wrong and every sample lands between the pixels the
model was trained on. Both failures are quiet - the accuracy number still prints, it just
stops describing production.

No `mobilenet_v2.preprocess_input`, no mean subtraction, no channel swap. The [0,1] -> [-1,1]
step lives inside the exported graph as `Rescaling(scale=2.0, offset=-1.0)`
(docs/ARCHITECTURE.md section 2.2). So if Python and the browser ever disagree about an
image, the bug is in the model - not in a preprocessing step one side forgot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# Imported lazily by load_stack() so `--help` works without the training venv.
np = tf = keras = Image = None
classification_report = confusion_matrix = None


def load_stack() -> None:
    """Import numpy/TensorFlow/Pillow/sklearn, or exit with an actionable message."""
    global np, tf, keras, Image, classification_report, confusion_matrix
    try:
        import numpy as _np
        import tensorflow as _tf
        from PIL import Image as _Image
        from sklearn.metrics import classification_report as _report
        from sklearn.metrics import confusion_matrix as _matrix
        from tensorflow import keras as _keras
    except ImportError as exc:
        sys.stderr.write(
            f"error: a dependency is missing ({exc}).\n"
            "       . ml/.venv/bin/activate && pip install -r ml/requirements.txt\n"
        )
        raise SystemExit(2) from exc

    np, tf, keras, Image = _np, _tf, _keras, _Image
    classification_report, confusion_matrix = _report, _matrix

DEFAULT_MODEL_CANDIDATES = (
    REPO_ROOT / "ml" / "artifacts" / "ecosort_mobilenetv2.keras",
    REPO_ROOT / "ml" / "artifacts" / "checkpoints" / "best.keras",
)
# train.py names the saved model after its backbone and records the path here.
LAST_RUN_SUMMARY = REPO_ROOT / "ml" / "artifacts" / "summary.json"
CLASS_SOURCE_CANDIDATES = (
    REPO_ROOT / "models" / "custom" / "metadata.json",
    REPO_ROOT / "ml" / "dataset" / "dataset.json",
)


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ArgumentDefaultsHelpFormatter, minus the useless "(default: None)" noise."""

    def _get_help_string(self, action):
        # None and plain False add nothing: "--no-cache (default: False)" is noise.
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description=(
            "Score a trained EcoSort model on a dataset split, or classify a single image "
            "using the exact preprocessing the browser uses."
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Trained .keras/.h5 model (default: the one ml/artifacts/summary.json names, then "
        "ml/artifacts/ecosort_mobilenetv2.keras, then ml/artifacts/checkpoints/best.keras)",
    )
    parser.add_argument("--data-dir", default=None, help="Dataset root (default: ml/dataset)")
    parser.add_argument(
        "--split", default="val", choices=("train", "val", "test"), help="Which split to score"
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Classify this single image and print the top-K instead of scoring a split",
    )
    parser.add_argument("--top-k", type=int, default=3, help="How many predictions to print for --image")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for split evaluation")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Override the input side length (default: read from the model's input shape)",
    )
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names in softmax order (overrides every other source)",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help="JSON file with a `classes` array (default: models/custom/metadata.json, "
        "then ml/dataset/dataset.json)",
    )
    parser.add_argument(
        "--save-confusion",
        default=None,
        help="Also write the confusion matrix to this PNG path",
    )
    return parser.parse_args(argv)


def resolve_model_path(value):
    if value:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"error: model not found: {path}")
        return path
    candidates = list(DEFAULT_MODEL_CANDIDATES)
    try:
        summary = json.loads(LAST_RUN_SUMMARY.read_text(encoding="utf-8"))
        recorded = summary.get("kerasModel")
        # A --smoke-test run was trained on random noise; never evaluate that by default.
        if isinstance(recorded, str) and recorded and not summary.get("smokeTest"):
            candidates.insert(0, Path(recorded))
    except (OSError, ValueError):
        pass  # no previous run, or an unreadable summary: fall back to the fixed names
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(
        "error: no trained model found. Looked for:\n"
        + "\n".join(f"       {c}" for c in DEFAULT_MODEL_CANDIDATES)
        + "\n       Train one with:  python ml/train.py\n"
        "       Or point at a file with:  --model path/to/model.keras"
    )


def load_class_names(args, num_outputs: int, data_dir: Path):
    if args.classes:
        names = [c.strip() for c in args.classes.split(",") if c.strip()]
        source = "--classes"
    else:
        names, source = None, None
        candidates = (
            [Path(args.classes_file).expanduser().resolve()] if args.classes_file else list(CLASS_SOURCE_CANDIDATES)
        )
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"warning: ignoring {candidate}: {exc}")
                continue
            found = payload.get("classes") if isinstance(payload, dict) else payload
            if isinstance(found, list) and found:
                names, source = [str(c) for c in found], str(candidate)
                break
        if names is None:
            split_dir = data_dir / args.split
            if split_dir.is_dir():
                names = sorted(p.name for p in split_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
                source = f"{split_dir} (sorted sub-directory names)"
        if not names:
            names = [f"class_{i}" for i in range(num_outputs)]
            source = "generated placeholders - no class list found"

    if len(names) != num_outputs:
        raise SystemExit(
            f"error: the model has {num_outputs} output(s) but the class list from {source} has "
            f"{len(names)}: {', '.join(names)}.\n"
            "       Pass the right list with --classes or --classes-file."
        )
    return names, source


def infer_image_size(model, override):
    if override:
        return int(override)
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) == 4 and shape[1] and shape[2]:
        return int(shape[1])
    raise SystemExit(
        f"error: could not read the input size from the model (input_shape={shape}). Pass --image-size."
    )


def preprocess_image(path: Path, image_size: int):
    """uint8 pixels -> centre-crop -> bilinear resize -> /255. Exactly what the browser does."""
    try:
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            array = np.asarray(rgb, dtype=np.float32)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the user's file, not a bug
        raise SystemExit(f"error: could not read image {path}: {type(exc).__name__}: {exc}") from exc

    # Centre-crop to the short edge first, mirroring canvasFromSource in
    # frontend/src/lib/imageUtils.js. Resizing the whole frame instead would score this
    # image against pixels the app crops away - silently, and only on non-square photos.
    height, width = array.shape[0], array.shape[1]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    square = array[top : top + side, left : left + side, :]

    # tf.image.resize bilinear is align_corners=False + half_pixel_centers=True, which is
    # what classifier.js asks tfjs for explicitly. PIL's kernel is a third thing again, so
    # the resize stays here in TF even though the decode happened in Pillow.
    resized = tf.image.resize(square[None, ...], (image_size, image_size), method="bilinear")
    return (resized / 255.0).numpy()


def classify_single(model, class_names, image_path: Path, image_size: int, top_k: int) -> int:
    batch = preprocess_image(image_path, image_size)
    probabilities = model.predict(batch, verbose=0)[0]

    total = float(probabilities.sum())
    print(f"\nImage        : {image_path}")
    print(f"Input        : 1x{image_size}x{image_size}x3, values in "
          f"[{batch.min():.3f}, {batch.max():.3f}]  (contract: [0, 1])")
    print(f"Output       : {probabilities.size} values summing to {total:.4f}  (contract: softmax, ~1.0)")
    if abs(total - 1.0) > 0.05:
        print("warning: the outputs do not sum to 1 - this model does not end in a softmax, "
              "so the browser's confidences will be wrong.")

    k = max(1, min(top_k, probabilities.size))
    order = np.argsort(probabilities)[::-1][:k]
    width = max(len(class_names[i]) for i in order)
    print(f"\nTop-{k}:")
    for rank, index in enumerate(order, start=1):
        value = float(probabilities[index])
        bar = "#" * int(round(value * 40))
        print(f"  {rank}. {class_names[index].ljust(width)}  {value * 100:6.2f}%  {bar}")
    return 0


def print_confusion(matrix, class_names) -> None:
    width = max(max(len(c) for c in class_names), 9)
    cell = max(5, max(len(str(int(v))) for v in matrix.flatten()) + 1)
    header = "true \\ pred".ljust(width) + "".join(c[:cell - 1].rjust(cell) for c in class_names)
    print("\nConfusion matrix (rows = true, columns = predicted)")
    print(header)
    for i, name in enumerate(class_names):
        row = name.ljust(width) + "".join(str(int(v)).rjust(cell) for v in matrix[i])
        print(row)


def evaluate_split(model, class_names, data_dir: Path, split: str, image_size: int, batch_size: int,
                   save_confusion) -> int:
    split_dir = data_dir / split
    if not split_dir.is_dir():
        raise SystemExit(
            f"error: {split_dir} does not exist. Build a dataset with `python ml/prepare_dataset.py`, "
            "or point at another one with --data-dir."
        )
    missing = [c for c in class_names if not (split_dir / c).is_dir()]
    if missing:
        raise SystemExit(f"error: {split_dir} is missing class folder(s): {', '.join(missing)}")

    try:
        raw = keras.utils.image_dataset_from_directory(
            split_dir,
            labels="inferred",
            label_mode="categorical",
            class_names=class_names,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            interpolation="bilinear",
            # Matches ml/train.py and the browser's centre-crop; see the module docstring.
            # The Keras default is False, which would squash instead of crop.
            crop_to_aspect_ratio=True,
            shuffle=False,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"error: could not load {split_dir}: {type(exc).__name__}: {exc}") from exc

    # The one and only pixel transform - see the module docstring.
    dataset = raw.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)).prefetch(tf.data.AUTOTUNE)

    probabilities = model.predict(dataset, verbose=1)
    y_true = np.concatenate([b.numpy() for _, b in dataset.unbatch().batch(1024)], axis=0).argmax(axis=1)
    y_pred = probabilities.argmax(axis=1)

    if y_true.size == 0:
        raise SystemExit(f"error: {split_dir} contains no images.")

    labels = list(range(len(class_names)))
    accuracy = float((y_true == y_pred).mean())
    mean_confidence = float(probabilities.max(axis=1).mean())

    print(f"\nSplit        : {split}  ({y_true.size} images, {len(class_names)} classes)")
    print(f"Accuracy     : {accuracy * 100:.2f}%")
    print(f"Mean top-1 confidence: {mean_confidence * 100:.2f}%")
    print("\nPer-class metrics")
    print(
        classification_report(
            y_true, y_pred, labels=labels, target_names=class_names, digits=4, zero_division=0
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    print_confusion(matrix, class_names)

    worst = []
    for i, name in enumerate(class_names):
        total = int(matrix[i].sum())
        if total == 0:
            continue
        correct = int(matrix[i, i])
        confusions = [(class_names[j], int(matrix[i, j])) for j in range(len(class_names)) if j != i and matrix[i, j]]
        confusions.sort(key=lambda pair: pair[1], reverse=True)
        worst.append((correct / total, name, confusions[:2]))
    worst.sort()
    if worst:
        print("\nWeakest classes (recall, most-confused-with)")
        for recall, name, confusions in worst[:5]:
            detail = ", ".join(f"{other} x{count}" for other, count in confusions) or "-"
            print(f"  {name:<12} recall {recall * 100:6.2f}%   confused with: {detail}")

    if save_confusion:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        path = Path(save_confusion).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        size = max(5.0, 0.7 * len(class_names) + 2.5)
        fig, ax = plt.subplots(figsize=(size, size))
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(labels)
        ax.set_yticks(labels)
        ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"EcoSort confusion matrix ({split}, acc {accuracy * 100:.1f}%)")
        threshold = matrix.max() / 2 if matrix.max() else 0.5
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=7,
                        color="white" if matrix[i, j] > threshold else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        print(f"\nWrote {path}")

    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    load_stack()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else REPO_ROOT / "ml" / "dataset"
    model_path = resolve_model_path(args.model)

    print(f"TensorFlow {tf.__version__} / Keras {keras.__version__}")
    print(f"Model        : {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"error: could not load {model_path}: {type(exc).__name__}: {exc}") from exc

    num_outputs = int(model.outputs[0].shape[-1])
    class_names, source = load_class_names(args, num_outputs, data_dir)
    image_size = infer_image_size(model, args.image_size)
    print(f"Classes ({len(class_names)}) : {', '.join(class_names)}")
    print(f"Class source : {source}")
    print(f"Input size   : {image_size}x{image_size}, preprocessing = pixels / 255 only")

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise SystemExit(f"error: image not found: {image_path}")
        return classify_single(model, class_names, image_path, image_size, args.top_k)

    return evaluate_split(
        model, class_names, data_dir, args.split, image_size, args.batch_size, args.save_confusion
    )


if __name__ == "__main__":
    raise SystemExit(main())
