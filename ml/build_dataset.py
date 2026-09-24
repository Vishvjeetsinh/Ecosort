#!/usr/bin/env python3
"""EcoSort - from nothing to a 10-class training set in one command.

    python ml/build_dataset.py                        # download, merge, split
    python ml/build_dataset.py --cap 2000             # allow more images per class
    python ml/build_dataset.py --garbage12 ~/Downloads/garbage_classification

The app has ten categories, but the dataset the first model was trained on covers seven:
it has no cardboard, textile or trash, so that model can never answer them. This script
fills the gap with two openly licensed datasets it can download itself, and merges them
with the one it cannot:

    source        licence     downloaded   fills
    ------------  ----------  -----------  -------------------------------------------------
    RealWaste     CC BY 4.0   yes (657 MB) cardboard, textile, trash + 5 shared classes
    TrashNet      MIT         yes (41 MB)  cardboard, trash + 4 shared classes
    customwaste   Kaggle      no           ewaste, hazardous + 5 shared classes (the only
                                           source of those two; see --customwaste)
    garbage12     Kaggle      no           optional: ~7,000 clothes/shoes for thin `textile`

Three steps, each of them an existing script, so nothing here re-implements their rules:
  1. download + verify (size and SHA-256) + extract into ml/downloads/   (once; cached)
  2. ml/ingest_sources.py --replace --link --cap N  -> ml/source/<class>/  (deduplicated)
  3. ml/prepare_dataset.py --overwrite --max-side 512  -> ml/dataset/{train,val,test}

Re-running is safe: downloads are reused, --replace swaps out exactly the files earlier
runs ingested (your own photos in ml/source/ are never touched), and the split is
deterministic for a given --seed.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import ingest_sources  # noqa: E402  (must follow sys.path setup)
import prepare_dataset  # noqa: E402

DOWNLOADS_DIR = HERE / "downloads"

# Byte counts and hashes measured on the archives as published; a mismatch means a
# truncated download or a changed upstream, and training on either would be a silent lie.
ARCHIVES = {
    "realwaste": {
        "title": "RealWaste (UCI Machine Learning Repository, CC BY 4.0, doi:10.24432/C5SS4G)",
        "url": "https://archive.ics.uci.edu/static/public/908/realwaste.zip",
        "file": "realwaste.zip",
        "bytes": 688_545_323,
        "sha256": "1ede08b32358ee62065bcc1c8cb47a2ece04e8dff5b1fb53342bb8750895b2f3",
        "root": "realwaste/realwaste-main/RealWaste",
    },
    "trashnet": {
        "title": "TrashNet (Thung & Yang, MIT licence)",
        "url": "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip",
        "file": "trashnet-dataset-resized.zip",
        "bytes": 42_834_870,
        "sha256": "c060e8abfe5d6de0578ca15be1ed8ad0794a865d333c3473d53d1d9ad6e38b8c",
        "root": "trashnet/dataset-resized",
    },
}

CUSTOMWASTE_DEFAULT = REPO_ROOT / "custom-waste-classification-dataset" / "wastes"
CUSTOMWASTE_KAGGLE = "wasifmahmood01/custom-waste-classification-dataset"
GARBAGE12_KAGGLE = "mostafaabla/garbage-classification"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, spec: dict) -> str | None:
    """None when `path` is the archive `spec` describes, else the reason it is not."""
    if not path.is_file():
        return "missing"
    size = path.stat().st_size
    if size != spec["bytes"]:
        return f"{size:,} bytes, expected {spec['bytes']:,}"
    if sha256_of(path) != spec["sha256"]:
        return "SHA-256 mismatch"
    return None


def download(spec: dict, dest: Path) -> None:
    """Stream to a .part file and rename on success, so an interrupted run never leaves an
    archive that merely looks complete."""
    partial = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {spec['url']}")
    request = urllib.request.Request(spec["url"], headers={"User-Agent": "EcoSort-build-dataset"})
    received = 0
    last_report = 0
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)
            received += len(chunk)
            if received - last_report >= 50 * (1 << 20):
                print(f"    {received / (1 << 20):,.0f} / {spec['bytes'] / (1 << 20):,.0f} MiB")
                last_report = received
    partial.replace(dest)


def fetch(name: str, downloads: Path, offline: bool) -> Path:
    """Make sure archive `name` is downloaded, verified and extracted; return its class root."""
    spec = ARCHIVES[name]
    archive = downloads / spec["file"]
    root = downloads / spec["root"]
    marker = downloads / f".{name}.extracted"

    if marker.is_file() and root.is_dir():
        print(f"{name}: already extracted at {root}")
        return root

    print(f"{name}: {spec['title']}")
    problem = verify_archive(archive, spec)
    if problem:
        if offline:
            raise SystemExit(
                f"error: {archive} is {problem} and --offline forbids downloading it.\n"
                f"       Fetch it yourself from {spec['url']}"
            )
        downloads.mkdir(parents=True, exist_ok=True)
        download(spec, archive)
        problem = verify_archive(archive, spec)
        if problem:
            raise SystemExit(
                f"error: the download of {spec['url']} is {problem}.\n"
                "       Delete it and retry; if it keeps happening the upstream file changed, "
                "and ARCHIVES in ml/build_dataset.py needs its new size and hash."
            )
    print(f"  verified {archive.name} ({spec['bytes'] / (1 << 20):,.0f} MiB, SHA-256 ok)")

    target = downloads / name
    shutil.rmtree(target, ignore_errors=True)
    with zipfile.ZipFile(archive) as bundle:
        # macOS resource forks (__MACOSX/, ._*) are not images, whatever their extension says.
        members = [
            m for m in bundle.namelist()
            if not m.startswith("__MACOSX/") and not Path(m).name.startswith("._")
        ]
        bundle.extractall(target, members=members)
    if not root.is_dir():
        raise SystemExit(f"error: {archive.name} did not contain {spec['root']} - the archive layout changed.")
    marker.write_text(spec["sha256"] + "\n", encoding="utf-8")
    print(f"  extracted to {root}")
    return root


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="build_dataset.py",
        description="Download the public waste datasets and build ml/source + ml/dataset for all 10 classes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--customwaste",
        default=str(CUSTOMWASTE_DEFAULT),
        help=f"Folder of the Kaggle {CUSTOMWASTE_KAGGLE} set (its train/ and test/ are both used)",
    )
    parser.add_argument(
        "--garbage12",
        default=None,
        help=f"Optional: folder of the Kaggle {GARBAGE12_KAGGLE} set, mainly for more textile",
    )
    parser.add_argument("--cap", type=int, default=1500, help="Maximum images per class after merging")
    parser.add_argument(
        "--max-side", type=int, default=512, help="Long side the split's images are re-encoded to"
    )
    parser.add_argument("--seed", type=int, default=42, help="Sampling and split seed")
    parser.add_argument("--downloads", default=str(DOWNLOADS_DIR), help="Where archives are cached")
    parser.add_argument("--offline", action="store_true", help="Never download; fail if an archive is missing")
    parser.add_argument("--dry-run", action="store_true", help="Download and plan, but write nothing to ml/")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    downloads = Path(args.downloads).expanduser().resolve()

    sources = []
    print("=" * 72)
    print("1/3  public datasets")
    print("=" * 72)
    for name in ARCHIVES:
        sources.append(f"{name}={fetch(name, downloads, args.offline)}")

    customwaste = Path(args.customwaste).expanduser().resolve()
    if customwaste.is_dir():
        sources.append(f"customwaste={customwaste}")
        print(f"customwaste: {customwaste}")
    else:
        print(
            f"\nWARNING: no customwaste set at {customwaste}.\n"
            "  It is the only source of `ewaste` and `hazardous`, so without it the model loses\n"
            "  those two classes. It needs a (free) Kaggle account:\n"
            f"    kaggle datasets download -d {CUSTOMWASTE_KAGGLE} --unzip -p custom-waste-classification-dataset\n"
            "  or copy the custom-waste-classification-dataset/ folder over from the machine that\n"
            "  has it, then re-run this script (or pass --customwaste <dir>)."
        )
    if args.garbage12:
        garbage12 = Path(args.garbage12).expanduser().resolve()
        if not garbage12.is_dir():
            raise SystemExit(f"error: --garbage12 {garbage12} is not a directory")
        sources.append(f"garbage12={garbage12}")

    print("\n" + "=" * 72)
    print(f"2/3  merge into ml/source (cap {args.cap} per class)")
    print("=" * 72)
    ingest_argv = ["--replace", "--link", "--cap", str(args.cap), "--seed", str(args.seed)]
    for source in sources:
        ingest_argv += ["--source", source]
    if args.dry_run:
        ingest_argv.append("--dry-run")
    ingest_sources.main(ingest_argv)

    print("\n" + "=" * 72)
    print("3/3  split into ml/dataset")
    print("=" * 72)
    prepare_argv = ["--overwrite", "--max-side", str(args.max_side), "--seed", str(args.seed)]
    if args.dry_run:
        prepare_argv.append("--dry-run")
    prepare_dataset.main(prepare_argv)

    print(
        "\nThe dataset is ready. Credit RealWaste (CC BY 4.0) and TrashNet (MIT) wherever you\n"
        "publish a model trained on it. Next:\n"
        "  make train-gpu        # on an NVIDIA machine (see ml/README.md)\n"
        "  make train            # anywhere else"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
