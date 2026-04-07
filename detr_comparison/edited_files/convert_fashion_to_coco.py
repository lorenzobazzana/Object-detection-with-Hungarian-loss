import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def load_fashion_annotation_file(label_path: Path) -> Dict:
    """Load a single Fashion JSON annotation file."""
    with label_path.open("r") as f:
        return json.load(f)


def build_coco_for_split(
    split_root: Path,
    image_ext: str = ".jpg",
) -> Dict:
    """
    Build a COCO-style dictionary for a given split directory.

    Expected structure under `split_root`:

        split_root/
          images/
            *.jpg
          annotations/
            *.json   # one JSON per image, same basename as image
    """
    images_dir = split_root / "images"
    ann_dir = split_root / "annotations"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {ann_dir}")

    images: List[Dict] = []
    annotations: List[Dict] = []
    categories_set = set()

    image_id = 1
    ann_id = 1

    # Iterate deterministically for reproducibility
    label_files: List[Path] = sorted(
        p for p in ann_dir.glob("*.json") if p.is_file()
    )

    for label_path in label_files:
        labels = load_fashion_annotation_file(label_path)

        # Derive image path from label path, matching src.lib.dataset.FashionDataset
        img_basename = label_path.name.replace(".json", image_ext)
        img_path = images_dir / img_basename
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image file for annotation not found: {img_path} "
                f"(derived from {label_path})"
            )

        # Read image size
        with Image.open(img_path) as img:
            width, height = img.size

        # Register image entry
        images.append(
            {
                "id": image_id,
                "file_name": img_basename,
                "width": width,
                "height": height,
            }
        )

        # Convert each 'item*' entry into a COCO annotation
        for key, obj in labels.items():
            if "item" not in key:
                continue

            category_id = int(obj["category_id"])
            x0, y0, x1, y1 = obj["bounding_box"]
            x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)

            # Ensure proper ordering
            x0, x1 = min(x0, x1), max(x0, x1)
            y0, y1 = min(y0, y1), max(y0, y1)

            w_box = max(0.0, x1 - x0)
            h_box = max(0.0, y1 - y0)

            # Skip degenerate boxes
            if w_box <= 0 or h_box <= 0:
                continue

            area = w_box * h_box

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x0, y0, w_box, h_box],  # COCO: [x, y, width, height]
                    "area": area,
                    "iscrowd": 0,
                    # We do not have segmentations; keep an empty list to satisfy COCO API.
                    "segmentation": [],
                }
            )
            categories_set.add(category_id)
            ann_id += 1

        image_id += 1

    # Build categories list from the observed category ids
    categories: List[Dict] = []
    for cid in sorted(categories_set):
        categories.append(
            {
                "id": int(cid),
                "name": f"category_{cid}",
                "supercategory": "fashion",
            }
        )

    # COCO format requires 'info' and 'licenses' fields
    coco_dict: Dict = {
        "info": {
            "description": "Fashion dataset converted to COCO format",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": "",
            }
        ],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    return coco_dict


def convert_fashion_root_to_coco(
    fashion_root: Path,
    output_dir: Path | None = None,
    image_ext: str = ".jpg",
) -> List[Tuple[str, Path]]:
    """
    Convert all supported splits under `fashion_root` into COCO format.

    We look for subdirectories named 'train' and 'test' (common in the project),
    but this can easily be extended if needed.
    """
    if output_dir is None:
        output_dir = fashion_root

    splits = ["train", "test"]
    written: List[Tuple[str, Path]] = []

    for split in splits:
        split_root = fashion_root / split
        if not split_root.exists():
            # Skip missing splits instead of failing
            continue

        coco_dict = build_coco_for_split(split_root, image_ext=image_ext)

        out_ann_dir = split_root
        out_ann_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_ann_dir / f"instances_{split}_fashion.json"

        with out_path.open("w") as f:
            json.dump(coco_dict, f)

        written.append((split, out_path))

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Fashion dataset (per-image JSON annotations) into "
            "COCO-style JSON files usable by DETR's COCO pipeline."
        )
    )
    parser.add_argument(
        "--fashion-root",
        type=str,
        required=True,
        help=(
            "Root directory of the Fashion dataset. "
            "Expected structure: "
            "'<root>/train/images', '<root>/train/annotations', "
            "'<root>/test/images', '<root>/test/annotations'."
        ),
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".jpg",
        help="Image file extension used in the dataset (default: .jpg).",
    )

    args = parser.parse_args()

    fashion_root = Path(args.fashion_root).expanduser().resolve()
    if not fashion_root.exists():
        raise FileNotFoundError(f"Fashion root does not exist: {fashion_root}")

    print("Starting creation of COCO annotations...")
    written = convert_fashion_root_to_coco(
        fashion_root=fashion_root,
        output_dir=fashion_root,
        image_ext=args.image_ext,
    )

    if not written:
        print(
            f"No splits were converted under {fashion_root}. "
            "Make sure 'train/' and/or 'test/' exist."
        )
    else:
        print("COCO-style annotation files written:")
        for split, path in written:
            print(f"  - split='{split}': {path}")


if __name__ == "__main__":
    main()

