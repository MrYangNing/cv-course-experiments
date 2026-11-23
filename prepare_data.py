"""Fetch the public experiment inputs, or import the same files from a local bundle."""
import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
DOWNLOADS = "work/provenance/downloads/"
SOURCES = {
    "detection": {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt",
        DOWNLOADS + "coco128.zip": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
    },
    "segmentation": {
        DOWNLOADS + "images.tar.gz.incomplete": "https://thor.robots.ox.ac.uk/pets/images.tar.gz",
        DOWNLOADS + "annotations-full.tar.gz": "https://thor.robots.ox.ac.uk/pets/annotations.tar.gz",
    },
    "classification": {
        DOWNLOADS + "cifar-fastai.tgz": "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz",
    },
}
PREFIX_SIZE = 13_631_251


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(relative, url, expected, local):
    dest = ROOT / relative
    if dest.is_file() and digest(dest) == expected:
        print("Verified", relative)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        if local:
            shutil.copyfile(local / relative, temporary)
        else:
            prefix = relative.endswith(".incomplete")
            headers = {"Range": f"bytes=0-{PREFIX_SIZE - 1}"} if prefix else {}
            with urlopen(Request(url, headers=headers), timeout=90) as src, temporary.open("wb") as out:
                if prefix:
                    out.write(src.read(PREFIX_SIZE))
                else:
                    shutil.copyfileobj(src, out)
        if digest(temporary) != expected:
            raise ValueError(f"SHA-256 mismatch for {relative}. Check the source version or use --from-dir.")
        temporary.replace(dest)
        print("Ready", relative)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["all", *SOURCES])
    parser.add_argument("--from-dir", type=Path, help="Root of the unpacked experiment data bundle")
    args = parser.parse_args()
    hashes = json.loads((ROOT / "reference/input_hashes.json").read_text())
    for name, files in SOURCES.items():
        if args.task in ("all", name):
            for relative, url in files.items():
                prepare(relative, url, hashes[relative], args.from_dir)
