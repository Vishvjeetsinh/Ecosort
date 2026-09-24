#!/usr/bin/env python3
"""EcoSort - turn a flat folder of labelled images into a train/val/test split.

Input layout (whatever you collected):

    <source>/plastic/0001.jpg
    <source>/plastic/0002.jpg
    <source>/glass/...

Output layout (what ml/train.py consumes):

    <out>/train/plastic/...
    <out>/val/plastic/...
    <out>/test/plastic/...
    <out>/dataset.json

`dataset.json` records the *sorted* class list. That sort order is the label order used by
`tf.keras.utils.image_dataset_from_directory`, by the softmax output of the exported model
and by `metadata.json.classes` in the browser - so it has to be written down once and
reused, never re-derived independently in two places.

Typical use:

    python ml/prepare_dataset.py --scaffold                 # make empty class folders
    # ... drop your images in ml/source/<class>/ ...
    python ml/prepare_dataset.py --dry-run                  # see the split first
    python ml/prepare_dataset.py                            # actually copy the files
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from PIL import Image, ImageFile
except ImportError as exc:  # pragma: no cover - environment problem, not a code path
    sys.stderr.write(
        "error: Pillow is not installed.\n"
        "       python3 -m venv ml/.venv && . ml/.venv/bin/activate\n"
        "       pip install -r ml/requirements.txt\n"
    )
    raise SystemExit(2) from exc

# Fail loudly on truncated JPEGs instead of silently padding them with grey, which is what
# we want here: a corrupt file should be reported and skipped, not quietly trained on.
ImageFile.LOAD_TRUNCATED_IMAGES = False

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

# docs/ARCHITECTURE.md section 3. Sorted, because sorted() is the label order.
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

SCAFFOLD_NOTES = {
    "cardboard": (
        "Corrugated boxes, shipping cartons, egg boxes, cereal boxes, kitchen-roll tubes.",
        "TrashNet has a `cardboard` class (~400 images) that drops straight in here.",
    ),
    "ewaste": (
        "Phones, laptops, cables, chargers, remote controls, keyboards, circuit boards, "
        "printer cartridges, small appliances.",
        "Not in TrashNet. The Kaggle 'E-Waste Image Dataset' and 'Garbage Classification "
        "(12 classes)' sets both cover it; own photos of your drawer of old cables work well.",
    ),
    "glass": (
        "Bottles, jars, drinking glasses, broken glass (photograph it safely).",
        "TrashNet `glass` (~500 images).",
    ),
    "hazardous": (
        "Batteries, paint tins, aerosols with hazard symbols, solvents, light bulbs, "
        "fluorescent tubes, expired medicine, syringes.",
        "Not in TrashNet. Kaggle 'Garbage Classification (12 classes)' has `battery`; "
        "photograph household chemicals for the rest.",
    ),
    "metal": (
        "Drink cans, food tins, aluminium foil, bottle caps, cutlery, empty aerosols.",
        "TrashNet `metal` (~400 images).",
    ),
    "organic": (
        "Food scraps, fruit and vegetable peel, coffee grounds, tea bags, eggshells, "
        "garden trimmings.",
        "Kaggle 'Waste Classification data' (organic vs recyclable) and the `biological` "
        "class of the 12-class Garbage Classification set.",
    ),
    "paper": (
        "Newspaper, magazines, office paper, envelopes, receipts, paper bags.",
        "TrashNet `paper` (~590 images).",
    ),
    "plastic": (
        "PET bottles, tubs, yoghurt pots, bottle caps, film, bags, packaging trays.",
        "TrashNet `plastic` (~480 images); TACO also has plastic-heavy annotations.",
    ),
    "textile": (
        "Clothes, shoes, towels, bedding, offcuts of fabric, belts, bags.",
        "Not in TrashNet. The Kaggle 'Garbage Classification (12 classes)' set has "
        "`clothes` and `shoes` - merge both into this folder.",
    ),
    "trash": (
        "Anything genuinely non-recyclable: crisp packets, dirty nappies, cigarette butts, "
        "polystyrene, mixed-material packaging, sweet wrappers.",
        "TrashNet `trash` (~130 images - the smallest class, top it up yourself).",
    ),
}

SCAFFOLD_README = """# `{cls}` training images

Put images of **{cls}** items in this folder (no sub-folders).

**What belongs here:** {what}

**Where to find images:** {where}

## Rules of thumb

* Accepted file types: {suffixes}.
* Aim for **at least 100 images per class**; 20 is the absolute floor and will overfit.
* Vary background, lighting, angle and distance. A hundred photos of the same bottle on
  the same table teaches the model about your table, not about plastic.
* One dominant object per photo. If two categories are visible, pick the one that fills
  the frame or drop the image.
* Roughly balance the classes. `ml/prepare_dataset.py` prints the imbalance ratio and
  `ml/train.py` applies class weights, but neither can invent data.
* Keep the split honest: do not put near-duplicate shots of the same object in here and
  expect the val score to mean anything - the splitter is random, so duplicates leak.

Then run, from the repository root:

    python ml/prepare_dataset.py --dry-run
    python ml/prepare_dataset.py
    python ml/train.py
"""


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """ArgumentDefaultsHelpFormatter, minus the useless "(default: None)" noise."""

    def _get_help_string(self, action):
        # None and plain False add nothing: "--no-cache (default: False)" is noise.
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="prepare_dataset.py",
        description=(
            "Split a flat <source>/<class>/*.jpg collection into a deterministic, "
            "stratified train/val/test dataset for ml/train.py."
        ),
        formatter_class=_HelpFormatter,
        epilog=(
            "The class list written to <out>/dataset.json is sorted alphabetically, and "
            "that order is the softmax output order of the exported model. Do not "
            "rename class folders after training without retraining."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Directory containing one sub-directory per class (default: ml/source)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory to write train/val/test into (default: ml/dataset)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.15, help="Fraction of each class held out for validation"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.10, help="Fraction of each class held out for test"
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed; identical seed = identical split")
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=20,
        help="Warn loudly about any class with fewer usable images than this",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--copy",
        dest="move",
        action="store_false",
        help="Copy files into the output tree, leaving the source untouched (default)",
    )
    mode.add_argument(
        "--move",
        dest="move",
        action="store_true",
        help="Move files instead of copying (saves disk, destroys the source layout)",
    )
    parser.set_defaults(move=False)

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without creating or touching a single file",
    )
    parser.add_argument(
        "--allow-extra-classes",
        action="store_true",
        help=(
            "Permit class folders outside the canonical 10 of docs/ARCHITECTURE.md "
            "section 3. The app only knows the canonical ids, so extra classes will "
            "render with a fallback label in the UI."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing <out>/{train,val,test} before writing (otherwise they are merged into)",
    )
    parser.add_argument(
        "--scaffold",
        action="store_true",
        help="Create empty <source>/<class>/ folders with a README for all 10 categories, then exit",
    )
    return parser.parse_args(argv)


def resolve_dir(value, default: Path) -> Path:
    """User-supplied paths resolve against the CWD; defaults against the repo root."""
    if value is None:
        return default
    return Path(value).expanduser().resolve()


def scaffold(source: Path, dry_run: bool) -> int:
    print(f"Scaffolding {len(CANONICAL_CLASSES)} class folders under {source}")
    created = 0
    for cls in CANONICAL_CLASSES:
        target = source / cls
        readme = target / "README.md"
        what, where = SCAFFOLD_NOTES[cls]
        body = SCAFFOLD_README.format(
            cls=cls,
            what=what,
            where=where,
            suffixes=", ".join(sorted(IMAGE_SUFFIXES)),
        )
        exists = target.is_dir()
        if dry_run:
            print(f"  [dry-run] {'keep  ' if exists else 'create'} {target}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        if not exists:
            created += 1
        # Never clobber a README the user has edited.
        if not readme.exists():
            readme.write_text(body, encoding="utf-8")
        print(f"  {'kept  ' if exists else 'created'} {target}")

    if dry_run:
        print("\nDry run - nothing was written.")
        return 0

    print(
        f"\n{created} new folder(s). Drop images into each, then run:\n"
        f"  python ml/prepare_dataset.py --dry-run"
    )
    return 0


def is_image_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def validate_image(path: Path):
    """Return (ok, reason). Actually decodes the file - a .jpg extension proves nothing."""
    try:
        with Image.open(path) as img:
            img.verify()  # cheap structural check; leaves the file object unusable
    except Exception as exc:  # noqa: BLE001 - any decoder failure means "skip this file"
        return False, f"{type(exc).__name__}: {exc}"

    try:
        # verify() does not decode pixels, so open a second time and force a full decode.
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            img.load()
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"

    if width < 16 or height < 16:
        return False, f"too small ({width}x{height})"
    return True, None


def split_counts(n: int, val_split: float, test_split: float):
    """Stratified per-class counts that never starve the training split."""
    n_val = int(round(n * val_split))
    n_test = int(round(n * test_split))

    # With 3+ images and a non-zero split, always keep at least one in each held-out set,
    # otherwise small classes silently vanish from val and the metric becomes a lie.
    if val_split > 0 and n >= 3:
        n_val = max(1, n_val)
    if test_split > 0 and n >= 3:
        n_test = max(1, n_test)

    while n - n_val - n_test < 1 and (n_val + n_test) > 0:
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
    return n - n_val - n_test, n_val, n_test


def scan_classes(source: Path, allow_extra: bool):
    """Return (class_names, files_by_class, skipped) or raise SystemExit on a fatal problem."""
    if not source.is_dir():
        raise SystemExit(
            f"error: source directory {source} does not exist.\n"
            f"       Create it with:  python ml/prepare_dataset.py --scaffold"
        )

    class_dirs = sorted(p for p in source.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not class_dirs:
        raise SystemExit(
            f"error: {source} has no class sub-directories.\n"
            f"       Create them with:  python ml/prepare_dataset.py --scaffold"
        )

    names = [p.name for p in class_dirs]
    unknown = [n for n in names if n not in CANONICAL_CLASSES]
    if unknown and not allow_extra:
        raise SystemExit(
            "error: class folder(s) outside the canonical taxonomy: "
            + ", ".join(unknown)
            + "\n       Canonical ids (docs/ARCHITECTURE.md section 3): "
            + ", ".join(CANONICAL_CLASSES)
            + "\n       Rename them, or pass --allow-extra-classes to keep them anyway."
        )
    if unknown:
        print(f"warning: keeping {len(unknown)} non-canonical class(es): {', '.join(unknown)}")
        print("         The EcoSort UI has no bin guidance for these ids.")

    files_by_class = {}
    skipped = []
    for class_dir in class_dirs:
        candidates = sorted(
            (p for p in class_dir.rglob("*") if is_image_path(p)),
            key=lambda p: str(p.relative_to(class_dir)),
        )
        usable = []
        for path in candidates:
            ok, reason = validate_image(path)
            if ok:
                usable.append(path)
            else:
                skipped.append({"path": str(path), "class": class_dir.name, "reason": reason})
        files_by_class[class_dir.name] = usable

    empty = [c for c, f in files_by_class.items() if not f]
    if empty:
        print(f"warning: {len(empty)} class folder(s) contain no usable images: {', '.join(sorted(empty))}")
        for name in empty:
            files_by_class.pop(name)

    class_names = sorted(files_by_class)
    if not class_names:
        raise SystemExit(f"error: no usable images found anywhere under {source}.")
    if len(class_names) < 2:
        raise SystemExit(
            f"error: only one class ({class_names[0]}) has images - a classifier needs at least two."
        )
    return class_names, files_by_class, skipped


def print_table(class_names, plan, min_per_class):
    width = max(len(c) for c in class_names + ["class"])
    header = f"{'class'.ljust(width)}  {'total':>7}  {'train':>7}  {'val':>7}  {'test':>7}"
    print("\n" + header)
    print("-" * len(header))
    thin = []
    for cls in class_names:
        n_train, n_val, n_test = plan[cls]["counts"]
        total = n_train + n_val + n_test
        flag = ""
        if total < min_per_class:
            flag = "  <-- below --min-per-class"
            thin.append(cls)
        print(
            f"{cls.ljust(width)}  {total:>7}  {n_train:>7}  {n_val:>7}  {n_test:>7}{flag}"
        )
    print("-" * len(header))
    totals = [0, 0, 0]
    for cls in class_names:
        counts = plan[cls]["counts"]
        totals = [a + b for a, b in zip(totals, counts)]
    print(
        f"{'TOTAL'.ljust(width)}  {sum(totals):>7}  {totals[0]:>7}  {totals[1]:>7}  {totals[2]:>7}"
    )

    sizes = [sum(plan[c]["counts"]) for c in class_names]
    ratio = max(sizes) / max(1, min(sizes))
    print(f"\nImbalance ratio (largest class / smallest class): {ratio:.2f}x")
    if ratio > 3:
        print(
            "warning: strong class imbalance. ml/train.py applies class weights by default,\n"
            "         but collecting more images for the small classes works far better."
        )
    if thin:
        print("")
        print("!" * 72)
        print(f"WARNING: {len(thin)} class(es) below --min-per-class={min_per_class}: {', '.join(thin)}")
        print("         Expect the model to overfit and to confuse these with everything else.")
        print("!" * 72)
    return ratio, totals


def main(argv=None) -> int:
    args = parse_args(argv)

    source = resolve_dir(args.source, REPO_ROOT / "ml" / "source")
    out = resolve_dir(args.out, REPO_ROOT / "ml" / "dataset")

    if args.scaffold:
        return scaffold(source, args.dry_run)

    if not 0 <= args.val_split < 1 or not 0 <= args.test_split < 1:
        raise SystemExit("error: --val-split and --test-split must each be in [0, 1).")
    if args.val_split + args.test_split >= 0.9:
        raise SystemExit(
            f"error: --val-split + --test-split = {args.val_split + args.test_split:.2f} "
            "leaves almost nothing to train on (max 0.9)."
        )
    if out == source:
        raise SystemExit("error: --out must differ from --source.")
    try:
        out.relative_to(source)
        raise SystemExit("error: --out must not live inside --source (it would re-ingest its own output).")
    except ValueError:
        pass  # good: out is not under source

    print(f"source : {source}")
    print(f"out    : {out}")
    print(f"mode   : {'move' if args.move else 'copy'}{'  (dry run)' if args.dry_run else ''}")
    print(f"split  : train {1 - args.val_split - args.test_split:.2f} / val {args.val_split:.2f} / test {args.test_split:.2f}   seed {args.seed}")

    class_names, files_by_class, skipped = scan_classes(source, args.allow_extra_classes)

    if skipped:
        print(f"\nSkipped {len(skipped)} unreadable/corrupt file(s):")
        for entry in skipped[:20]:
            print(f"  - {entry['path']}: {entry['reason']}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more (full list in dataset.json)")

    # Deterministic: the file list is sorted before shuffling, and each class gets its own
    # derived seed so adding a class does not reshuffle the others.
    plan = {}
    for index, cls in enumerate(class_names):
        files = list(files_by_class[cls])
        rng = random.Random(args.seed * 1000003 + index)
        rng.shuffle(files)
        n_train, n_val, n_test = split_counts(len(files), args.val_split, args.test_split)
        plan[cls] = {
            "counts": (n_train, n_val, n_test),
            "train": files[:n_train],
            "val": files[n_train : n_train + n_val],
            "test": files[n_train + n_val : n_train + n_val + n_test],
        }

    ratio, totals = print_table(class_names, plan, args.min_per_class)

    if args.dry_run:
        print("\nDry run - no files were copied, moved or deleted.")
        print("Re-run without --dry-run to build the dataset.")
        return 0

    if args.overwrite:
        for split in ("train", "val", "test"):
            target = out / split
            if target.exists():
                print(f"removing {target}")
                shutil.rmtree(target)

    written = 0
    collisions = 0
    for cls in class_names:
        for split in ("train", "val", "test"):
            target_dir = out / split / cls
            target_dir.mkdir(parents=True, exist_ok=True)
            for src_path in plan[cls][split]:
                dest = target_dir / src_path.name
                # Sub-folders inside a class dir can produce duplicate basenames; de-duplicate
                # deterministically rather than silently overwriting one with the other.
                if dest.exists():
                    stem, suffix = src_path.stem, src_path.suffix
                    counter = 1
                    while dest.exists():
                        dest = target_dir / f"{stem}__{counter}{suffix}"
                        counter += 1
                    collisions += 1
                try:
                    if args.move:
                        shutil.move(str(src_path), str(dest))
                    else:
                        shutil.copy2(src_path, dest)
                except OSError as exc:
                    raise SystemExit(f"error: failed to write {dest}: {exc}") from exc
                written += 1

    if collisions:
        print(f"\nnote: renamed {collisions} file(s) whose basename already existed in the target folder.")

    dataset_json = {
        "createdAt": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "seed": args.seed,
        "source": str(source),
        "mode": "move" if args.move else "copy",
        "valSplit": args.val_split,
        "testSplit": args.test_split,
        # The single source of truth for label order. train.py reads this back.
        "classes": class_names,
        "classCount": len(class_names),
        "canonical": all(c in CANONICAL_CLASSES for c in class_names),
        "counts": {
            split: {cls: plan[cls]["counts"][i] for cls in class_names}
            for i, split in enumerate(("train", "val", "test"))
        },
        "totals": {
            "train": totals[0],
            "val": totals[1],
            "test": totals[2],
            "all": sum(totals),
        },
        "imbalanceRatio": round(ratio, 4),
        "skipped": skipped,
        "notes": (
            "classes[] is sorted and IS the label index order used by "
            "tf.keras.utils.image_dataset_from_directory, by the exported softmax and by "
            "models/custom/metadata.json. Do not reorder it."
        ),
    }
    manifest_path = out / "dataset.json"
    manifest_path.write_text(json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote {written} file(s) into {out}")
    print(f"Wrote {manifest_path}")
    print(f"Classes (label order): {', '.join(class_names)}")
    print("\nNext:\n  python ml/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
