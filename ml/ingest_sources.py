#!/usr/bin/env python3
"""EcoSort - merge downloaded datasets into ml/source/<class>/.

    python ml/ingest_sources.py --inspect ~/Downloads/some-kaggle-dataset
    python ml/ingest_sources.py --source trashnet=~/Downloads/trashnet/data
    python ml/ingest_sources.py --source garbage12=~/Downloads/garbage --cap 800 --link

`prepare_dataset.py` scaffolds `ml/source/` and splits it into train/val/test. It does not
know how to get images INTO `ml/source/`, because every public dataset uses its own folder
names: TrashNet calls it `cardboard`, the Kaggle 12-class set splits glass three ways by
bottle colour, and nothing but your own photos calls anything `ewaste`. That mapping is
what this script owns.

WHY THIS IS A SCRIPT AND NOT A FEW `cp` COMMANDS
------------------------------------------------
Three things go wrong when you merge datasets by hand, and all three are silent:

1. **Filename collisions.** TrashNet and the Kaggle 12-class set both contain
   `cardboard/cardboard1.jpg`. Copy both into one folder and you keep one image and lose
   the other without a word. Every file here is renamed `<source>__<original>` on the way
   in, so two sources can never overwrite each other.

2. **Duplicates across the split boundary.** These datasets re-package each other, so the
   same photo turns up in two of them. `prepare_dataset.py` shuffles files; it cannot see
   that two paths are the same image. A duplicate that lands in train AND test inflates
   your test accuracy for free - which defeats the entire point of measuring it. Every
   file is hashed and repeats are dropped.

3. **Imbalance.** `clothes` + `shoes` is ~7,270 images while TrashNet's `trash` is ~137.
   Left alone, `compute_class_weights` in train.py turns that 29x ratio into a 29x weight
   on the noisiest class and destabilises fine-tuning. `--cap` fixes it at the source,
   deterministically, so a later "why is textile weak?" has an auditable answer.

This script only moves files. Decoding, validation, the train/val/test split and the
imbalance warnings all live in `prepare_dataset.py` and run next.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Single source of truth for the label set, so this file can never drift from the splitter.
from prepare_dataset import CANONICAL_CLASSES, IMAGE_SUFFIXES, is_image_path  # noqa: E402

# Mirrors CATEGORY_SYNONYMS in frontend/src/lib/classifier.js. Keep the two in step: this
# one names training folders, that one names a trained model's output classes, and a
# disagreement between them is the kind of bug that only shows up as bad predictions.
SYNONYMS = {
    "cardboard": "cardboard",
    "carton": "cardboard",
    "cartons": "cardboard",
    "ewaste": "ewaste",
    "e-waste": "ewaste",
    "electronic": "ewaste",
    "electronics": "ewaste",
    "electronicwaste": "ewaste",
    "glass": "glass",
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
    "battery": "hazardous",
    "batteries": "hazardous",
    "chemical": "hazardous",
    "chemicals": "hazardous",
    "hazard": "hazardous",
    "hazardous": "hazardous",
    # The scaffold notes for `hazardous` in prepare_dataset.py list light bulbs and
    # fluorescent tubes explicitly, so this is the taxonomy's own answer, not a guess.
    "bulb": "hazardous",
    "bulbs": "hazardous",
    "lightbulb": "hazardous",
    "lightbulbs": "hazardous",
    "metal": "metal",
    "aluminium": "metal",
    "aluminum": "metal",
    "can": "metal",
    "cans": "metal",
    "tin": "metal",
    "organic": "organic",
    "biological": "organic",
    "compost": "organic",
    "food": "organic",
    "foodwaste": "organic",
    "paper": "paper",
    "plastic": "plastic",
    "clothes": "textile",
    "clothing": "textile",
    "fabric": "textile",
    "shoes": "textile",
    "textile": "textile",
    "textiles": "textile",
    "trash": "trash",
    "general": "trash",
    "landfill": "trash",
    "other": "trash",
    "rubbish": "trash",
}

# Datasets documented in ml/README.md, keyed by the short name you pass to --source.
# A value of None means "deliberately ignored" and is reported as such rather than
# silently skipped, so an ignored folder is always a decision someone can find.
SOURCE_MAPS = {
    "trashnet": {
        "cardboard": "cardboard",
        "glass": "glass",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "trash",
    },
    # Kaggle mostafaabla/garbage-classification. Fills TrashNet's gaps except ewaste.
    "garbage12": {
        "battery": "hazardous",
        "biological": "organic",
        "brown-glass": "glass",
        "green-glass": "glass",
        "white-glass": "glass",
        "cardboard": "cardboard",
        "clothes": "textile",
        "shoes": "textile",
        "metal": "metal",
        "paper": "paper",
        "plastic": "plastic",
        "trash": "trash",
    },
    # Kaggle techsash/waste-classification-data. Only the organic half is usable: "R"
    # (recyclable) spans five EcoSort classes at once and cannot be mapped to one.
    "techsash": {
        "O": "organic",
        "R": None,
    },
    # Kaggle wasifmahmood01/custom-waste-classification-dataset. ~11.5k images, and by far
    # the best `ewaste` source available - that is the class nothing else covers. Ships its
    # own train/test split, which prepare_dataset.py re-splits; see ml/README.md.
    # Covers 7 of the 10 classes: cardboard, textile and trash are not in it.
    "customwaste": {
        "E-waste": "ewaste",
        "battery waste": "hazardous",
        "light bulbs": "hazardous",
        "glass waste": "glass",
        "metal waste": "metal",
        "organic waste": "organic",
        "paper waste": "paper",
        "plastic waste": "plastic",
        # Tyres, bumpers, oil filters, brake discs. None of the ten categories fits: it is
        # not household waste at all, and EcoSort is a kitchen-bin app. Filing it under
        # `trash` would teach "trash = car parts", which is worse than not having it.
        # Override with --map if your deployment sees automotive waste.
        "automobile wastes": None,
    },
}


def normalise(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def guess_class(folder_name: str):
    """Best-effort folder -> EcoSort class. Returns (class_or_None, how)."""
    key = normalise(folder_name)
    if key in SYNONYMS:
        return SYNONYMS[key], "exact"

    # Kaggle folders are often "plastic water bottles" or "01_cardboard". Match whole
    # tokens, last first, because the head noun tends to come last ("battery waste").
    tokens = [t for t in key.replace("-", " ").split() if t]
    for token in reversed(tokens):
        if token in SYNONYMS:
            return SYNONYMS[token], f"token '{token}'"
        if token.endswith("s") and token[:-1] in SYNONYMS:
            return SYNONYMS[token[:-1]], f"token '{token[:-1]}'"

    # Last resort, for multi-word synonyms like "e-waste" that tokenising splits apart.
    # Anchored on word boundaries: a plain substring test matches "e-waste" inside
    # "automobilE-WASTEs" and would file car parts under electronics.
    for synonym, target in SYNONYMS.items():
        if re.search(rf"(?<![a-z0-9]){re.escape(synonym)}(?![a-z0-9])", key):
            return target, f"phrase '{synonym}'"
    return None, "no match"


def image_dirs(root: Path):
    """Map every directory that directly contains images to its image files."""
    found = defaultdict(list)
    for dirpath, _dirnames, filenames in os.walk(root):
        directory = Path(dirpath)
        files = [directory / f for f in filenames if Path(f).suffix.lower() in IMAGE_SUFFIXES]
        if files:
            found[directory] = sorted(files)
    return found


def inspect(root: Path) -> int:
    """Print an unknown dataset's structure and the mapping this script would infer."""
    if not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")

    found = image_dirs(root)
    if not found:
        raise SystemExit(
            f"error: no image files under {root}\n"
            f"       (looked for {', '.join(sorted(IMAGE_SUFFIXES))})"
        )

    print(f"Inspecting {root}\n")
    rows = []
    for directory in sorted(found):
        rel = directory.relative_to(root)
        label = rel.name if rel.name else root.name
        target, how = guess_class(label)
        rows.append((str(rel) or ".", len(found[directory]), target, how))

    width = max(len(r[0]) for r in rows)
    print(f"{'folder'.ljust(width)}  {'images':>7}  {'-> EcoSort class':<18} how")
    print("-" * (width + 50))
    total = 0
    unmapped = []
    for rel, count, target, how in rows:
        total += count
        shown = target if target else "?"
        print(f"{rel.ljust(width)}  {count:>7}  {shown:<18} {how}")
        if target is None:
            unmapped.append(rel)

    print("-" * (width + 50))
    print(f"{'total'.ljust(width)}  {total:>7}")

    per_class = Counter()
    for rel, count, target, _how in rows:
        if target:
            per_class[target] += count
    print("\nWould contribute:")
    for cls in CANONICAL_CLASSES:
        n = per_class.get(cls, 0)
        flag = "" if n else "   <- nothing for this class"
        print(f"  {cls:<10} {n:>7}{flag}")

    if unmapped:
        print(f"\n{len(unmapped)} folder(s) could not be mapped automatically:")
        for rel in unmapped:
            print(f"  {rel}")
        # Quoted, because Kaggle folder names routinely contain spaces.
        print(
            "\nMap them explicitly, e.g.:\n"
            f"  python ml/ingest_sources.py --source 'mydata={root}' \\\n"
            f"      --map '{unmapped[0]}=ewaste'"
        )
    print(
        "\nNothing was copied. Re-run without --inspect, and with --source <name>=<dir>,\n"
        "to ingest. Add --dry-run to see the plan first."
    )
    return 0


def file_digest(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_mapping(source_name: str, root: Path, overrides: dict):
    """Decide the EcoSort class for every image folder under `root`."""
    table = SOURCE_MAPS.get(source_name, {})
    plan = {}
    ignored = []
    unmapped = []

    for directory, files in image_dirs(root).items():
        rel = directory.relative_to(root)
        label = rel.name if rel.name else root.name

        # Explicit --map wins over everything, by relative path then by folder name.
        target = overrides.get(str(rel), overrides.get(normalise(label)))
        how = "--map"

        if target is None and str(rel) not in overrides and normalise(label) not in overrides:
            if label in table or normalise(label) in table:
                target = table.get(label, table.get(normalise(label)))
                how = f"SOURCE_MAPS[{source_name}]"
                if target is None:
                    ignored.append((str(rel), len(files)))
                    continue
            else:
                target, how = guess_class(label)

        if target is None:
            unmapped.append((str(rel), len(files)))
            continue
        if target not in CANONICAL_CLASSES:
            raise SystemExit(
                f"error: '{target}' is not an EcoSort class.\n"
                f"       valid: {', '.join(CANONICAL_CLASSES)}"
            )
        plan.setdefault(target, []).extend(files)

    return plan, ignored, unmapped


def ingest(args) -> int:
    out = Path(args.out).expanduser().resolve()
    rng = random.Random(args.seed)

    overrides = {}
    for item in args.map or []:
        if "=" not in item:
            raise SystemExit(f"error: --map expects folder=class, got '{item}'")
        folder, _, cls = item.partition("=")
        overrides[normalise(folder)] = cls.strip().lower()
        overrides[folder.strip()] = cls.strip().lower()

    sources = []
    for item in args.source:
        if "=" not in item:
            raise SystemExit(
                f"error: --source expects <name>=<dir>, got '{item}'\n"
                f"       known names: {', '.join(sorted(SOURCE_MAPS))} (any other name works too)"
            )
        name, _, path = item.partition("=")
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"error: --source {name}: {root} is not a directory")
        sources.append((name.strip(), root))

    print(f"out    : {out}")
    print(f"mode   : {'hardlink' if args.link else 'copy'}{'  (dry run)' if args.dry_run else ''}")
    print(f"cap    : {args.cap if args.cap else 'none'} per class     seed {args.seed}")
    print()

    # Collect first, cap second: capping per source would bias toward whichever source
    # happened to be listed first rather than sampling the merged pool.
    pooled = defaultdict(list)
    for name, root in sources:
        plan, ignored, unmapped = resolve_mapping(name, root, overrides)
        contributed = sum(len(v) for v in plan.values())
        print(f"{name}: {root}")
        print(f"  {contributed} image(s) across {len(plan)} class(es)")
        for cls in sorted(plan):
            print(f"    {cls:<10} {len(plan[cls]):>7}")
        for rel, count in ignored:
            print(f"    (ignored {rel}: {count} image(s) - mapped to None in SOURCE_MAPS)")
        for rel, count in unmapped:
            print(f"    WARNING unmapped {rel}: {count} image(s) skipped - use --map")
        print()
        for cls, files in plan.items():
            pooled[cls].extend((name, f) for f in files)

    if not pooled:
        raise SystemExit("error: nothing to ingest. Run with --inspect <dir> to see why.")

    seen = set()
    if not args.no_dedup:
        print("Hashing for duplicates ...")

    tally = Counter()
    per_source = defaultdict(Counter)
    duplicates = 0
    written = 0

    for cls in sorted(pooled):
        entries = pooled[cls][:]
        rng.shuffle(entries)

        kept = []
        for name, path in entries:
            if args.cap and len(kept) >= args.cap:
                break
            if not args.no_dedup:
                try:
                    digest = file_digest(path)
                except OSError as exc:
                    print(f"  WARNING could not read {path}: {exc}")
                    continue
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
            kept.append((name, path))

        target_dir = out / cls
        if not args.dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)

        for name, path in kept:
            # Source prefix is what makes two datasets' identical filenames survive
            # landing in the same folder.
            dest = target_dir / f"{name}__{path.name}"
            suffix = 1
            while dest.exists():
                dest = target_dir / f"{name}__{path.stem}__{suffix}{path.suffix}"
                suffix += 1
            if not args.dry_run:
                try:
                    if args.link:
                        os.link(path, dest)
                    else:
                        shutil.copy2(path, dest)
                except OSError as exc:
                    # Hardlinks fail across filesystems; a copy is always correct.
                    if args.link:
                        shutil.copy2(path, dest)
                    else:
                        print(f"  WARNING could not place {path}: {exc}")
                        continue
            tally[cls] += 1
            per_source[cls][name] += 1
            written += 1

    print()
    print("=" * 72)
    print("INGESTED" + ("  (dry run - nothing was written)" if args.dry_run else ""))
    print("=" * 72)
    for cls in CANONICAL_CLASSES:
        n = tally.get(cls, 0)
        breakdown = ", ".join(f"{s} {c}" for s, c in sorted(per_source[cls].items()))
        flag = "   <- EMPTY, the model can never predict this" if n == 0 else ""
        print(f"  {cls:<10} {n:>7}  {breakdown}{flag}")
    print(f"\n  {'total':<10} {written:>7}")
    if duplicates:
        print(f"\n  {duplicates} duplicate image(s) dropped (identical bytes).")

    counts = [tally.get(c, 0) for c in CANONICAL_CLASSES if tally.get(c, 0) > 0]
    if counts:
        ratio = max(counts) / min(counts)
        print(f"  imbalance ratio {ratio:.1f}x")
        if ratio > 3:
            print("  WARNING: above 3x. Lower --cap, or collect more of the thin classes.")

    missing = [c for c in CANONICAL_CLASSES if tally.get(c, 0) == 0]
    if missing:
        print(f"\n  {len(missing)} class(es) have no images: {', '.join(missing)}")
        print("  A class with no training images is a class the model will never predict.")

    print("\nNext:\n  python ml/prepare_dataset.py        # validate, split, write dataset.json")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Merge downloaded waste datasets into ml/source/<class>/.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "known source names (others work too, folders are matched by name):\n  "
            + "\n  ".join(sorted(SOURCE_MAPS))
        ),
    )
    parser.add_argument(
        "--inspect",
        metavar="DIR",
        help="print a dataset's folder structure and the mapping that would be inferred, then exit",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="a dataset to ingest, e.g. trashnet=~/Downloads/trashnet/data (repeatable)",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="FOLDER=CLASS",
        help="force a folder onto an EcoSort class, overriding the inferred mapping (repeatable)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "ml" / "source"),
        help="destination root (default: ml/source)",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=800,
        help="maximum images per class; 0 disables (default: 800)",
    )
    parser.add_argument("--seed", type=int, default=42, help="sampling seed (default: 42)")
    parser.add_argument("--link", action="store_true", help="hardlink instead of copying")
    parser.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="skip the md5 duplicate check (faster, but duplicates can leak across the split)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.inspect:
        return inspect(Path(args.inspect).expanduser().resolve())
    if not args.source:
        raise SystemExit(
            "error: nothing to do. Pass --inspect <dir> to explore a dataset, or\n"
            "       --source <name>=<dir> to ingest one. See --help."
        )
    return ingest(args)


if __name__ == "__main__":
    raise SystemExit(main())
