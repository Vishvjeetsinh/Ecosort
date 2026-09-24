#!/usr/bin/env python3
"""Convert a pretrained `tf.keras.applications` ImageNet classifier to TensorFlow.js.

EcoSort can run any of these as its fallback engine: the model predicts ImageNet-1k
classes and `frontend/src/lib/imagenetWasteMap.js` maps those onto waste streams.

    python ml/convert_pretrained.py --arch InceptionResNetV2
    python ml/convert_pretrained.py --arch InceptionResNetV2 --quantize uint8
    python ml/convert_pretrained.py --list

WHY THIS SCRIPT EXISTS
    MobileNetV2 is the only ImageNet classifier Google publishes in a ready-made
    TensorFlow.js build. Everything else has to be converted from Keras, which is what
    this does: download the Keras weights, wrap the model so it honours EcoSort's
    [0,1] input contract, and run it through the same SavedModel -> tfjs_graph_model
    route that ml/export_tfjs.py uses for custom models.

THE [0,1] INPUT CONTRACT
    `keras.applications` models do NOT preprocess their own input; you are expected to
    call `preprocess_input` first. Every architecture below uses "tf" mode, i.e.
    x/127.5 - 1 over [0,255], which is exactly [0,1] -> [-1,1]. Putting a
    `Rescaling(scale=2.0, offset=-1.0)` layer at the front bakes that into the exported
    graph, so the browser divides by 255 and nothing else -- the same single code path
    `classifier.js` already uses for MobileNetV2 and for custom models.

    Architectures with "caffe" preprocessing (ResNet50, VGG*: BGR channel swap plus
    per-channel mean subtraction) or "torch" preprocessing (DenseNet: ImageNet mean/std)
    are deliberately NOT offered. They cannot be expressed as one Rescaling layer, and a
    silently wrong normalisation is far worse than an unsupported architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# arch -> (keras attribute, default input size, human description)
# Every entry MUST use "tf"-mode preprocessing (x/127.5 - 1). See the module docstring.
ARCHITECTURES: dict[str, tuple[str, int, str]] = {
    "InceptionResNetV2": (
        "InceptionResNetV2",
        299,
        "Inception-ResNet v2 - the most accurate option here (~80.3% ImageNet top-1), "
        "and by far the largest.",
    ),
    "InceptionV3": (
        "InceptionV3",
        299,
        "Inception v3 - noticeably better than MobileNet at a third of "
        "Inception-ResNet's size.",
    ),
    "Xception": (
        "Xception",
        299,
        "Xception - depthwise-separable Inception; accuracy close to Inception v3.",
    ),
    "MobileNetV2": (
        "MobileNetV2",
        224,
        "MobileNetV2 - the same architecture as the bundled fallback, rebuilt from "
        "Keras weights.",
    ),
    "MobileNetV3Large": (
        "MobileNetV3Large",
        224,
        "MobileNetV3 Large - a little more accurate than V2 at a similar size.",
    ),
    "NASNetMobile": (
        "NASNetMobile",
        224,
        "NASNet Mobile - compact, competitive with MobileNetV3.",
    ),
}

QUANTIZATION_FLAGS = {
    "none": [],
    "float16": ["--quantize_float16", "*"],
    "uint8": ["--quantize_uint8", "*"],
}


class ConvertError(RuntimeError):
    """Raised with an actionable message when conversion cannot continue."""


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def build_model(arch: str, image_size: int):
    """The Keras classifier, wrapped so it takes [0,1] input."""
    import tensorflow as tf

    attr, _, _ = ARCHITECTURES[arch]
    factory = getattr(tf.keras.applications, attr, None)
    if factory is None:
        raise ConvertError(
            f"tf.keras.applications has no {attr} in TensorFlow {tf.__version__}."
        )

    print(f"  loading {arch} ImageNet weights (first run downloads ~100-250 MB) ...")
    base = factory(weights="imagenet", include_top=True, input_shape=(image_size, image_size, 3))

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image", dtype="float32")
    # [0,1] -> [-1,1]; identical to keras preprocess_input("tf" mode) over [0,255].
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="rescale_unit_to_signed")(inputs)
    outputs = base(x, training=False)
    model = tf.keras.Model(inputs, outputs, name=f"ecosort_{arch.lower()}")

    units = int(model.outputs[0].shape[-1])
    if units not in (1000, 1001):
        raise ConvertError(
            f"{arch} produced {units} output units; expected 1000 (or 1001 with a "
            "background class). EcoSort maps ImageNet-1k indices, so this model cannot "
            "be used as a fallback engine."
        )
    return model, units


def run_converter(args: list[str]) -> tuple[bool, str]:
    for cmd in ([sys.executable, "-m", "tensorflowjs.converters.converter", *args],
                ["tensorflowjs_converter", *args]):
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except (OSError, ValueError) as exc:
            continue
        if proc.returncode == 0:
            return True, " ".join(cmd[:3])
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, tail[-1] if tail else f"exit {proc.returncode}"
    return False, "tensorflowjs_converter is not installed in this environment"


def convert(model, out_dir: Path, quantize: str) -> str:
    """Keras -> SavedModel -> tfjs graph model, written into out_dir."""
    with tempfile.TemporaryDirectory(prefix="ecosort-convert-") as tmp:
        sm_dir = Path(tmp) / "saved_model"
        print("  exporting a SavedModel ...")
        exporter = getattr(model, "export", None)
        if callable(exporter):
            exporter(str(sm_dir))
        else:  # Keras 2
            import tensorflow as tf

            tf.saved_model.save(model, str(sm_dir))
        if not (sm_dir / "saved_model.pb").is_file():
            raise ConvertError("Keras produced no SavedModel (saved_model.pb missing)")

        print(f"  converting to TensorFlow.js (quantization: {quantize}) ...")
        ok, detail = run_converter(
            [
                "--input_format=tf_saved_model",
                "--output_format=tfjs_graph_model",
                "--signature_name=serving_default",
                "--saved_model_tags=serve",
                *QUANTIZATION_FLAGS[quantize],
                str(sm_dir),
                str(out_dir),
            ]
        )
        if not ok:
            raise ConvertError(f"tensorflowjs_converter failed: {detail}")
        return detail


def measure(out_dir: Path) -> tuple[int, list[str]]:
    spec = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
    shards = [p for group in spec.get("weightsManifest", []) for p in group["paths"]]
    missing = [p for p in shards if not (out_dir / p).is_file()]
    if missing:
        raise ConvertError(f"model.json references missing shard(s): {', '.join(missing[:5])}")
    total = sum((out_dir / p).stat().st_size for p in shards)
    total += (out_dir / "model.json").stat().st_size
    return total, shards


def write_metadata(out_dir: Path, arch: str, image_size: int, units: int,
                   quantize: str, total_bytes: int) -> None:
    _, _, description = ARCHITECTURES[arch]
    offset = 1 if units == 1001 else 0
    metadata = {
        "name": arch.lower(),
        "displayName": f"{arch} (ImageNet)",
        "description": description,
        "version": "1.0.0",
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "baseModel": arch,
        "inputSize": image_size,
        "inputRange": [0, 1],
        # include_top=True in keras.applications always ends in a softmax.
        "outputActivation": "softmax",
        "classOffset": offset,
        "labelKind": "imagenet",
        "classes": None,
        "classCount": units,
        "quantization": quantize,
        "downloadBytes": total_bytes,
        "notes": (
            f"{arch} converted from tf.keras.applications by ml/convert_pretrained.py. "
            "TFJS *graph* model - load with tf.loadGraphModel. Feed pixels scaled to "
            "[0,1] and nothing else: the exported graph begins with "
            "Rescaling(scale=2.0, offset=-1.0), which applies the [-1,1] normalisation "
            "keras.applications.preprocess_input would otherwise do, so doing it "
            "yourself would double-apply it. Outputs softmax probabilities over "
            "ImageNet-1k in the standard class order used by "
            "frontend/src/lib/imagenetClasses.js."
        ),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a pretrained keras.applications ImageNet classifier to TensorFlow.js.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arch", default="InceptionResNetV2",
                        help="Architecture to convert (default: InceptionResNetV2). --list shows them all.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: models/<arch lowercased>)")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Override the input side length (defaults per architecture)")
    parser.add_argument("--quantize", choices=sorted(QUANTIZATION_FLAGS), default="float16",
                        help="Weight quantization. float16 halves the download for ~no accuracy "
                             "loss and is the default; uint8 quarters it with a small loss; "
                             "none keeps full float32.")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing export")
    parser.add_argument("--list", action="store_true", help="List the supported architectures and exit")
    args = parser.parse_args()

    if args.list:
        print("Supported architectures (all use [0,1] -> [-1,1] preprocessing):\n")
        for name, (_, size, desc) in ARCHITECTURES.items():
            print(f"  {name:<20} {size}x{size}  {desc}")
        print("\nResNet/VGG/DenseNet are intentionally absent: their preprocessing cannot be")
        print("expressed as a single Rescaling layer, and getting it wrong is silent.")
        return 0

    if args.arch not in ARCHITECTURES:
        print(f"error: unknown architecture {args.arch!r}. Run --list to see the options.", file=sys.stderr)
        return 2

    _, default_size, _ = ARCHITECTURES[args.arch]
    image_size = args.image_size or default_size
    out_dir = Path(args.out) if args.out else REPO_ROOT / "models" / args.arch.lower()
    out_dir = out_dir.expanduser().resolve()

    if (out_dir / "model.json").is_file() and not args.force:
        print(f"{out_dir}/model.json already exists. Re-run with --force to replace it.")
        return 0

    print(f"\nConverting {args.arch} -> {out_dir}")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    staging = out_dir.parent / f".{out_dir.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)
        model, units = build_model(args.arch, image_size)
        route = convert(model, staging, args.quantize)
        total, shards = measure(staging)
        write_metadata(staging, args.arch, image_size, units, args.quantize, total)

        print(f"  converted via {route}")
        print(f"  {len(shards)} shard(s), {human_bytes(total)} total download")
        if total > 120 * 1024 * 1024:
            print(f"  NOTE: {human_bytes(total)} is a large first-load for a browser. "
                  "Consider --quantize uint8.")

        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        for item in sorted(staging.iterdir()):
            shutil.move(str(item), str(out_dir / item.name))
    except ConvertError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone. Select \"{args.arch}\" in EcoSort's model picker (reload the page if it is open).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
