#!/usr/bin/env python3
"""EcoSort - convert a trained Keras model to TensorFlow.js and write its metadata.json.

Used two ways:

    # from ml/train.py
    from export_tfjs import export_to_tfjs
    export_to_tfjs(model, out_dir, class_names, metrics=..., image_size=224)

    # standalone, e.g. to re-export an old checkpoint
    python ml/export_tfjs.py --model ml/artifacts/checkpoints/best.keras \\
                            --out models/custom \\
                            --classes cardboard,ewaste,glass,...

THE PREPROCESSING CONTRACT (docs/ARCHITECTURE.md sections 2.2 / 2.3)
--------------------------------------------------------------------
The exported graph MUST accept pixels in [0, 1], because
`frontend/src/lib/classifier.js` has exactly one preprocessing path for both engines:
`tf.browser.fromPixels(...).resizeBilinear(...).toFloat().div(255)` and nothing else.

That works only because the model itself starts with
`tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)`, which maps [0, 1] -> [-1, 1] inside
the graph - the range MobileNetV2 was trained on. This module refuses to export a model
whose input range looks wrong, so a regression in train.py is caught here rather than
three weeks later when the browser quietly misclassifies everything.
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

METADATA_VERSION = "1.0.0"
MODEL_NAME = "ecosort-mobilenetv2"

# Only these may be deleted when cleaning the target directory. Anything else in there is
# the user's and we refuse to touch it - `--out` pointed at the wrong place is a very
# cheap mistake to make and a very expensive one to make silently.
DELETABLE_EXACT = {"model.json", "metadata.json"}
DELETABLE_PREFIXES = ("group",)
DELETABLE_SUFFIXES = (".bin",)

_NOTES_TEMPLATE = (
    "Custom EcoSort classifier: MobileNetV2 transfer learning exported from ml/train.py. "
    "{loader} Feed pixels scaled to [0,1] and nothing else: the graph's first layer is "
    "Rescaling(scale=2.0, offset=-1.0), which converts [0,1] to the [-1,1] range "
    "MobileNetV2 expects, so applying the Keras x/127.5-1 transform yourself would "
    "double-apply it. The output is a softmax over classes[] in that exact order; there "
    "is no background class, hence classOffset 0."
)

_LOADERS = {
    "graph-model": "TFJS *graph* model - load with tf.loadGraphModel, not tf.loadLayersModel.",
    "layers-model": "TFJS *layers* model - load with tf.loadLayersModel, not tf.loadGraphModel.",
}


def metadata_notes(model_format: str) -> str:
    """The notes string, matched to whichever TFJS format the converter actually produced."""
    loader = _LOADERS.get(
        model_format,
        "Load with tf.loadGraphModel unless model.json says format 'layers-model'.",
    )
    return _NOTES_TEMPLATE.format(loader=loader)


def read_model_format(out_dir: Path) -> str:
    """The `format` field of a converted model.json ('graph-model' / 'layers-model')."""
    try:
        return json.loads((out_dir / "model.json").read_text(encoding="utf-8")).get(
            "format", "graph-model"
        )
    except (OSError, ValueError):
        return "graph-model"


class ExportError(RuntimeError):
    """Raised when the model could not be converted or the result failed verification."""


def _is_deletable(path: Path) -> bool:
    name = path.name
    if name in DELETABLE_EXACT:
        return True
    if name.endswith(DELETABLE_SUFFIXES):
        return True
    return name.startswith(DELETABLE_PREFIXES) and "shard" in name


def inspect_output_dir(out_dir: Path) -> list:
    """Create out_dir and list its removable contents, refusing to touch unrelated files.

    Called *before* conversion so a wrongly-pointed --out fails in a second rather than
    after a twenty-minute training run.
    """
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ExportError(f"could not create {out_dir}: {exc}") from exc
    entries = sorted(out_dir.iterdir())
    strangers = [p for p in entries if not _is_deletable(p)]
    if strangers:
        raise ExportError(
            f"refusing to clean {out_dir}: it contains file(s) that are not part of a TFJS "
            f"export: {', '.join(p.name for p in strangers[:10])}. "
            "Point --out at a dedicated directory, or remove those files yourself."
        )
    return entries


def clean_output_dir(out_dir: Path) -> list[str]:
    """Remove a previous export from out_dir, refusing to delete unrelated files."""
    removed = []
    for path in inspect_output_dir(out_dir):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise ExportError(f"could not remove {path}: {exc}") from exc
        removed.append(path.name)
    return removed


def _reset_dir(path: Path) -> None:
    """Empty a staging directory between conversion attempts, so a failed route's
    half-written output cannot be mistaken for the next route's result."""
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _convert_via_saved_model(model, out_dir: Path) -> str:
    """Primary route: Keras model -> TF SavedModel -> TFJS *graph* model.

    Keras 3 (the default from TensorFlow 2.16 on) writes `batch_shape` into its
    InputLayer config, and tfjs-layers cannot read that: `tf.loadLayersModel` dies with
    "An InputLayer should be passed either a `batchInputShape` or an `inputShape`".
    So a Keras-3 layers export converts cleanly and is then unloadable in the browser --
    the failure only shows up at runtime, which is the worst place for it.

    Round-tripping through a SavedModel drops the Keras config entirely and yields a
    graph model, which tf.loadGraphModel reads on both the Keras 2 and Keras 3 stacks
    and which also runs faster in the browser. Verified to match the Python model's
    softmax to ~1e-7.
    """
    with tempfile.TemporaryDirectory(prefix="ecosort-savedmodel-") as tmp:
        sm_dir = Path(tmp) / "saved_model"
        exporter = getattr(model, "export", None)
        if callable(exporter):
            exporter(str(sm_dir))            # Keras 3
        else:
            import tensorflow as tf          # Keras 2 fallback

            tf.saved_model.save(model, str(sm_dir))

        if not (sm_dir / "saved_model.pb").is_file():
            raise ExportError("Keras produced no SavedModel (saved_model.pb is missing)")

        ok, detail = _run_converter_cli(
            [
                "--input_format=tf_saved_model",
                "--output_format=tfjs_graph_model",
                "--signature_name=serving_default",
                "--saved_model_tags=serve",
                str(sm_dir),
                str(out_dir),
            ]
        )
        if not ok:
            raise ExportError(detail)
        return f"SavedModel -> tfjs_graph_model via {detail}"


def _convert_with_python_api(model, out_dir: Path) -> str:
    import tensorflowjs as tfjs  # imported lazily: train.py --no-export must work without it

    tfjs.converters.save_keras_model(model, str(out_dir))
    return f"tensorflowjs {getattr(tfjs, '__version__', 'unknown')} (Python API)"


def _run_converter_cli(args: list[str]) -> tuple[bool, str]:
    """Try `python -m tensorflowjs.converters.converter`, then the console script."""
    attempts = [
        [sys.executable, "-m", "tensorflowjs.converters.converter", *args],
        ["tensorflowjs_converter", *args],
    ]
    problems = []
    for cmd in attempts:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        except (OSError, ValueError) as exc:
            problems.append(f"{cmd[0]}: {exc}")
            continue
        if proc.returncode == 0:
            return True, " ".join(cmd[:3])
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        problems.append(f"{' '.join(cmd[:3])} exited {proc.returncode}: {tail[-1] if tail else 'no output'}")
    return False, "; ".join(problems)


def _convert_with_cli(model, out_dir: Path) -> str:
    """Fallback path: dump the model to a temp file and shell out to the converter.

    Worth the extra step because the Python API and the CLI disagree about Keras 3 in
    several tensorflowjs/tensorflow combinations; whichever one works, works.
    """
    with tempfile.TemporaryDirectory(prefix="ecosort-export-") as tmp:
        tmp_dir = Path(tmp)
        failures = []

        h5_path = tmp_dir / "model.h5"
        try:
            model.save(h5_path)
            ok, detail = _run_converter_cli(
                ["--input_format=keras", str(h5_path), str(out_dir)]
            )
            if ok:
                return f"tensorflowjs_converter --input_format=keras via {detail}"
            failures.append(f"HDF5 route: {detail}")
        except Exception as exc:  # noqa: BLE001 - report and try the next route
            failures.append(f"HDF5 route: could not save .h5 ({type(exc).__name__}: {exc})")

        keras_path = tmp_dir / "model.keras"
        try:
            model.save(keras_path)
            ok, detail = _run_converter_cli(
                ["--input_format=keras_keras", str(keras_path), str(out_dir)]
            )
            if ok:
                return f"tensorflowjs_converter --input_format=keras_keras via {detail}"
            failures.append(f".keras route: {detail}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f".keras route: could not save .keras ({type(exc).__name__}: {exc})")

        raise ExportError("; ".join(failures))


def convert_model(model, out_dir: Path) -> str:
    """Convert a Keras model into out_dir. Returns a description of the route taken.

    Route order matters. The SavedModel -> graph-model route goes first because it is the
    only one that reliably produces something the browser can actually load on a Keras 3
    stack; the Keras layers routes stay as fallbacks for older Keras 2 setups where the
    converter's Python API is the better-tested path.
    """
    problems: list[str] = []

    _reset_dir(out_dir)
    try:
        return _convert_via_saved_model(model, out_dir)
    except ExportError as exc:
        problems.append(f"SavedModel -> graph-model: {exc}")
    except ImportError as exc:
        problems.append(
            f"SavedModel -> graph-model: tensorflowjs is not importable ({exc}). "
            "Install it with:  pip install -r ml/requirements.txt"
        )
    except Exception as exc:  # noqa: BLE001 - any failure falls through to the next route
        problems.append(f"SavedModel -> graph-model: {type(exc).__name__}: {exc}")

    print(f"  graph-model route unavailable: {problems[-1]}")
    print("  falling back to the Keras layers routes ...")

    _reset_dir(out_dir)
    try:
        return _convert_with_python_api(model, out_dir)
    except ImportError as exc:
        problems.append(
            f"tfjs Python API: not importable ({exc}). "
            "Install it with:  pip install -r ml/requirements.txt"
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"tfjs Python API: save_keras_model failed ({type(exc).__name__}: {exc})")

    _reset_dir(out_dir)
    try:
        return _convert_with_cli(model, out_dir)
    except ExportError as exc:
        problems.append(f"converter CLI: {exc}")

    raise ExportError(
        "TensorFlow.js conversion failed on every route.\n  "
        + "\n  ".join(problems)
        + "\nFix: install the converter into this same virtualenv with\n"
        "     pip install -r ml/requirements.txt\n"
        "and keep tensorflowjs and tensorflow at the versions pinned there - adding your\n"
        "own tensorflow bound on top is what usually breaks this."
    )


def assert_browser_loadable(out_dir: Path, model_format: str) -> None:
    """Refuse to ship a layers-model that @tensorflow/tfjs would fail to load.

    A Keras 3 layers export converts without complaint and then throws in the browser,
    so catch it here where the message can actually say what to do about it.
    """
    if model_format != "layers-model":
        return
    try:
        spec = json.loads((out_dir / "model.json").read_text(encoding="utf-8"))
        layers = (
            spec.get("modelTopology", {})
            .get("model_config", {})
            .get("config", {})
            .get("layers", [])
        )
    except (OSError, ValueError, AttributeError):
        return

    for layer in layers:
        if layer.get("class_name") != "InputLayer":
            continue
        config = layer.get("config", {})
        if "batch_shape" in config and "batch_input_shape" not in config:
            raise ExportError(
                "the converter produced a Keras 3 layers-model, whose InputLayer uses\n"
                "`batch_shape`. tfjs-layers only understands `batch_input_shape`, so\n"
                "tf.loadLayersModel would fail in the browser with\n"
                '  "An InputLayer should be passed either a `batchInputShape` or an `inputShape`".\n'
                "This build should have taken the SavedModel -> graph-model route instead;\n"
                "check the route message printed above for why it was skipped."
            )


def check_input_range(model) -> None:
    """Assert the [0,1]-input contract by finding the Rescaling layer inside the model."""

    def walk(layers):
        for layer in layers:
            yield layer
            inner = getattr(layer, "layers", None)
            if inner:
                yield from walk(inner)

    rescalers = [l for l in walk(model.layers) if type(l).__name__ == "Rescaling"]
    if not rescalers:
        raise ExportError(
            "the model has no Rescaling layer, so it cannot be taking [0,1] input. "
            "docs/ARCHITECTURE.md section 2.2 requires Rescaling(scale=2.0, offset=-1.0) "
            "as the first layer after the Input."
        )
    first = rescalers[0]
    scale = float(getattr(first, "scale", 0.0))
    offset = float(getattr(first, "offset", 0.0))
    if abs(scale - 2.0) > 1e-6 or abs(offset + 1.0) > 1e-6:
        raise ExportError(
            f"Rescaling layer '{first.name}' has scale={scale}, offset={offset}; the browser "
            "contract requires scale=2.0, offset=-1.0 so that [0,1] input maps to [-1,1]. "
            f"scale={1/127.5:.6f} would mean the model expects [0,255] - see "
            "docs/ARCHITECTURE.md section 2.2."
        )


def write_metadata(
    out_dir: Path,
    class_names,
    metrics=None,
    base_model: str = "MobileNetV2",
    image_size: int = 224,
    epochs=None,
    model_format: str = "graph-model",
) -> Path:
    """Write out_dir/metadata.json in the exact shape of ARCHITECTURE section 2.3."""
    classes = list(class_names)
    metadata = {
        "name": MODEL_NAME,
        "version": METADATA_VERSION,
        "createdAt": datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "baseModel": base_model,
        "inputSize": int(image_size),
        "inputRange": [0, 1],
        "outputActivation": "softmax",
        "classOffset": 0,
        "labelKind": "waste",
        "classes": classes,
        "classCount": len(classes),
        "metrics": dict(metrics) if metrics else None,
        "notes": metadata_notes(model_format),
    }
    if epochs is not None:
        metadata["epochs"] = int(epochs)

    path = out_dir / "metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def verify_export(out_dir: Path) -> dict:
    """Parse model.json, confirm every weight shard exists, and total the bytes on disk."""
    model_json_path = out_dir / "model.json"
    if not model_json_path.is_file():
        raise ExportError(f"conversion reported success but {model_json_path} does not exist.")

    try:
        model_json = json.loads(model_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"{model_json_path} is not readable JSON: {exc}") from exc

    manifest = model_json.get("weightsManifest")
    if not isinstance(manifest, list) or not manifest:
        raise ExportError(f"{model_json_path} has no weightsManifest - the export is incomplete.")

    shards = []
    missing = []
    total_bytes = model_json_path.stat().st_size
    for group in manifest:
        for rel in group.get("paths", []):
            shard = out_dir / rel
            if shard.is_file():
                shards.append(rel)
                total_bytes += shard.stat().st_size
            else:
                missing.append(rel)
    if missing:
        raise ExportError(
            f"weightsManifest references {len(missing)} missing shard(s): {', '.join(missing[:5])}"
        )
    if not shards:
        raise ExportError(f"{model_json_path} lists no weight shards - the export is empty.")

    return {
        "modelJson": str(model_json_path),
        "shards": shards,
        "totalBytes": total_bytes,
        "format": model_json.get("format", "graph-model (no format field)"),
    }


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GiB"


def export_to_tfjs(
    model,
    out_dir,
    class_names,
    metrics=None,
    base_model: str = "MobileNetV2",
    image_size: int = 224,
    epochs=None,
) -> dict:
    """Convert `model` to TFJS in `out_dir` and write metadata.json next to it.

    Returns the verification dict from verify_export(), extended with `metadataPath`.
    Raises ExportError with an actionable message on every failure path.
    """
    out_dir = Path(out_dir).expanduser().resolve()
    classes = list(class_names)
    if not classes:
        raise ExportError("class_names is empty - metadata.json.classes must list every output class.")

    out_units = int(model.outputs[0].shape[-1])
    if out_units != len(classes):
        raise ExportError(
            f"the model has {out_units} output unit(s) but {len(classes)} class name(s) were "
            "given. metadata.json.classes must line up index-for-index with the softmax."
        )

    check_input_range(model)

    print(f"\nExporting to TensorFlow.js -> {out_dir}")
    existing = inspect_output_dir(out_dir)

    # Build into a sibling staging directory and swap on success, so a converter failure
    # cannot leave the user with neither the old model nor a new one.
    staging = out_dir.parent / f".{out_dir.name}.staging-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        staging.mkdir(parents=True)

        route = convert_model(model, staging)
        print(f"  converted via {route}")

        # The format the converter actually chose drives both the loadability check and
        # the loader sentence in metadata.json, so read it back rather than assuming.
        model_format = read_model_format(staging)
        assert_browser_loadable(staging, model_format)

        write_metadata(
            staging,
            classes,
            metrics=metrics,
            base_model=base_model,
            image_size=image_size,
            epochs=epochs,
            model_format=model_format,
        )
        info = verify_export(staging)
        print(
            f"  verified: {len(info['shards'])} weight shard(s), "
            f"{human_bytes(info['totalBytes'])} total, format={info['format']}"
        )

        if existing:
            removed = clean_output_dir(out_dir)
            print(f"  replaced {len(removed)} file(s) from the previous export")
        for item in sorted(staging.iterdir()):
            shutil.move(str(item), str(out_dir / item.name))
    finally:
        # A no-op once every file has been moved out; the safety net on every failure path.
        shutil.rmtree(staging, ignore_errors=True)

    info = verify_export(out_dir)
    info["metadataPath"] = str(out_dir / "metadata.json")
    print(f"  installed {len(info['shards']) + 2} file(s) in {out_dir}")
    return info


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ArgumentDefaultsHelpFormatter, minus the useless "(default: None)" noise."""

    def _get_help_string(self, action):
        # None and plain False add nothing: "--no-cache (default: False)" is noise.
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def _load_classes(args) -> list[str]:
    if args.classes:
        names = [c.strip() for c in args.classes.split(",") if c.strip()]
        if not names:
            raise SystemExit("error: --classes was empty after parsing.")
        return names

    if args.classes_file:
        path = Path(args.classes_file).expanduser().resolve()
    else:
        path = REPO_ROOT / "ml" / "dataset" / "dataset.json"
        if not path.is_file():
            raise SystemExit(
                "error: no class list. Pass --classes a,b,c or --classes-file "
                "path/to/dataset.json (none found at ml/dataset/dataset.json)."
            )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: could not read class list from {path}: {exc}") from exc

    if isinstance(payload, list):
        names = [str(c) for c in payload]
    elif isinstance(payload, dict) and isinstance(payload.get("classes"), list):
        names = [str(c) for c in payload["classes"]]
    else:
        raise SystemExit(f"error: {path} has no `classes` array.")
    print(f"Class list read from {path}")
    return names


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="export_tfjs.py",
        description="Convert a trained EcoSort .keras model to TensorFlow.js + metadata.json.",
        formatter_class=_HelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to the trained .keras (or .h5) model")
    parser.add_argument("--out", default=None, help="Output directory (default: models/custom)")
    parser.add_argument(
        "--classes",
        default=None,
        help="Comma-separated class names in softmax order (overrides --classes-file)",
    )
    parser.add_argument(
        "--classes-file",
        default=None,
        help="JSON file with a `classes` array (default: ml/dataset/dataset.json)",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Square input side length")
    parser.add_argument("--base-model", default="MobileNetV2", help="Value for metadata.baseModel")
    parser.add_argument(
        "--val-accuracy", type=float, default=None, help="Value for metadata.metrics.valAccuracy"
    )
    parser.add_argument(
        "--val-loss", type=float, default=None, help="Value for metadata.metrics.valLoss"
    )
    args = parser.parse_args(argv)

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise SystemExit(f"error: model file not found: {model_path}")

    out_dir = (
        Path(args.out).expanduser().resolve() if args.out else REPO_ROOT / "models" / "custom"
    )
    class_names = _load_classes(args)

    # Quieten TF's C++ logger before it is imported, purely for readable output.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise SystemExit(
            f"error: TensorFlow is not installed ({exc}).\n"
            "       pip install -r ml/requirements.txt"
        ) from exc

    print(f"Loading {model_path}")
    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:  # noqa: BLE001 - surface the real loader error
        raise SystemExit(f"error: could not load {model_path}: {type(exc).__name__}: {exc}") from exc

    metrics = None
    if args.val_accuracy is not None or args.val_loss is not None:
        metrics = {}
        if args.val_accuracy is not None:
            metrics["valAccuracy"] = float(args.val_accuracy)
        if args.val_loss is not None:
            metrics["valLoss"] = float(args.val_loss)

    try:
        info = export_to_tfjs(
            model,
            out_dir,
            class_names,
            metrics=metrics,
            base_model=args.base_model,
            image_size=args.image_size,
        )
    except ExportError as exc:
        raise SystemExit(f"error: {exc}") from exc

    print(f"\nDone. {info['modelJson']}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print("Reload the EcoSort page (or press 'Reload model' in Settings) to pick it up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
