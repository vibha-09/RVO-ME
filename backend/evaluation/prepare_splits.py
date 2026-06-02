"""
prepare_splits.py — Create patient-level train / val / test splits.

Groups images by patient_id (extracted from the filename prefix before '_')
to prevent data leakage across splits.  Both mask and image must exist for
an entry to be included.

Outputs: train.txt, val.txt, test.txt  (one filename stem per line)

Usage
-----
  cd backend
  python -m evaluation.prepare_splits \\
      --images-dir ../dataset/RVO-Lesion/Image_Seg/images \\
      --masks-dir  ../dataset/RVO-Lesion/Image_Seg/masks  \\
      --output-dir ../dataset/RVO-Lesion/Image_Seg        \\
      --test-ratio 0.20 --val-ratio 0.10 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def create_splits(
    images_dir: Path,
    masks_dir:  Path,
    output_dir: Path,
    test_ratio: float = 0.20,
    val_ratio:  float = 0.10,
    seed:       int   = 42,
    overwrite:  bool  = False,
) -> dict:
    """
    Creates patient-level splits and writes .txt files.
    Returns a dict with counts.
    """
    # Check existing files
    existing = [(output_dir / f"{s}.txt") for s in ("train", "val", "test")]
    if any(p.exists() for p in existing) and not overwrite:
        print("Split files already exist. Pass --overwrite to regenerate.")
        for p in existing:
            if p.exists():
                n = sum(1 for ln in p.read_text().splitlines() if ln.strip())
                print(f"  {p.name}: {n} images")
        return {}

    # Collect stems with matching mask
    valid_stems: list[str] = []
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = img_path.stem
        if any((masks_dir / (stem + ext)).exists() for ext in IMAGE_EXTS):
            valid_stems.append(stem)

    if not valid_stems:
        raise ValueError(
            f"No valid image-mask pairs found.\n  images: {images_dir}\n  masks: {masks_dir}"
        )

    # Group by patient_id (prefix before first '_')
    patient_map: dict[str, list[str]] = defaultdict(list)
    for stem in valid_stems:
        pid = stem.split("_")[0]
        patient_map[pid].append(stem)

    patients = sorted(patient_map.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_total = len(patients)
    n_test  = max(1, round(n_total * test_ratio))
    n_val   = max(1, round(n_total * val_ratio))
    n_train = n_total - n_test - n_val

    if n_train <= 0:
        raise ValueError("Too few patients for the requested split ratios.")

    train_pats = patients[:n_train]
    val_pats   = patients[n_train : n_train + n_val]
    test_pats  = patients[n_train + n_val:]

    def _stems(pat_list: list[str]) -> list[str]:
        out: list[str] = []
        for p in pat_list:
            out.extend(sorted(patient_map[p]))
        return out

    splits = {
        "train": _stems(train_pats),
        "val":   _stems(val_pats),
        "test":  _stems(test_pats),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, stems in splits.items():
        path = output_dir / f"{name}.txt"
        path.write_text("\n".join(stems) + "\n", encoding="utf-8")
        n_pats = len({s.split("_")[0] for s in stems})
        print(f"  {name}.txt  — {len(stems):>5} images  ({n_pats} patients)  → {path}")

    return {
        "n_patients":     n_total,
        "n_train_images": len(splits["train"]),
        "n_val_images":   len(splits["val"]),
        "n_test_images":  len(splits["test"]),
        "n_train_pats":   len(train_pats),
        "n_val_pats":     len(val_pats),
        "n_test_pats":    len(test_pats),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create patient-level train/val/test splits")
    p.add_argument("--images-dir", required=True, type=Path,
                   help="Directory containing raw OCT images")
    p.add_argument("--masks-dir",  required=True, type=Path,
                   help="Directory containing ground-truth mask files")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Where to write train.txt / val.txt / test.txt")
    p.add_argument("--test-ratio", default=0.20, type=float,
                   help="Fraction of patients for the test set (default 0.20)")
    p.add_argument("--val-ratio",  default=0.10, type=float,
                   help="Fraction of patients for the validation set (default 0.10)")
    p.add_argument("--seed",       default=42,   type=int,
                   help="Random seed for reproducibility (default 42)")
    p.add_argument("--overwrite",  action="store_true",
                   help="Overwrite existing split files")
    return p.parse_args()


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    args = _parse_args()
    stats = create_splits(
        images_dir = args.images_dir,
        masks_dir  = args.masks_dir,
        output_dir = args.output_dir,
        test_ratio = args.test_ratio,
        val_ratio  = args.val_ratio,
        seed       = args.seed,
        overwrite  = args.overwrite,
    )
    if stats:
        print(f"\n  Total patients : {stats['n_patients']}")
        print(f"  Train : {stats['n_train_images']} images  ({stats['n_train_pats']} patients)")
        print(f"  Val   : {stats['n_val_images']} images  ({stats['n_val_pats']} patients)")
        print(f"  Test  : {stats['n_test_images']} images  ({stats['n_test_pats']} patients)")
