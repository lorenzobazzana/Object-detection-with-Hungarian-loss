from pathlib import Path

from .coco import CocoDetection, make_coco_transforms


class FashionCocoDetection(CocoDetection):
    """
    COCO-style detection dataset for the Fashion data.

    This expects that the original per-image Fashion annotations have been
    converted into COCO JSON files using
    `src/convert_fashion_to_coco.py`.

    Expected layout under `args.fashion_path`:

        fashion_root/
          train/
            images/
              *.jpg
            annotations/
              instances_train_fashion.json
          test/
            images/
              *.jpg
            annotations/
              instances_test_fashion.json
    """

    pass


def build(image_set, args):
    """
    Build a COCO-style Fashion dataset using the same transform pipeline
    as COCO (RandomResize / Normalize, etc.).
    """
    root = Path(args.fashion_path)
    assert root.exists(), f"provided fashion root path {root} does not exist"

    mode = "instances"
    if image_set == "train":
        img_root = root / "train" / "images"
        ann_file = root / "train" / f"{mode}_train_fashion.json"
    elif image_set == "val":
        img_root = root / "test" / "images"
        ann_file = root / "test"  / f"{mode}_test_fashion.json"
    else:
        raise ValueError(f'Unsupported image_set "{image_set}" for fashion dataset')

    assert img_root.exists(), f"provided fashion image path {img_root} does not exist"
    assert ann_file.exists(), (
        f"COCO annotation file for Fashion dataset not found: {ann_file}. "
        "Have you run src/convert_fashion_to_coco.py?"
    )

    dataset = FashionCocoDetection(
        img_folder=img_root,
        ann_file=str(ann_file),
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
    )
    return dataset
