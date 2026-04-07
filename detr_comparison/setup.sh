#!/bin/bash

# Args:
# - fashion_dataset_root: root directory of the DeepFashion2 dataset.
#   Expected structure:
#   '<fashion_dataset_root>/train/images', '<fashion_dataset_root>/train/annotations'
#   '<fashion_dataset_root>/test/images', '<fashion_dataset_root>/test/annotations'

if [[ $# -ne 1 ]] || [[ $1 = "--help" ]] || [[ $1 = "-h" ]]; then
    echo -e "Usage:\n/bin/bash/setup.sh FASHION_DATASET_ROOT"
    echo -e "Options:\n-h, --help: print this message and exit"
    echo -e "FASHION_DATASET_ROOT: root directory of the DeepFashion2 dataset."
    echo -e "                      Expected structure:"
    echo -e "                      '<fashion_dataset_root>/train/images', '<fashion_dataset_root>/train/annotations'"
    echo -e "                      '<fashion_dataset_root>/test/images', '<fashion_dataset_root>/test/annotations'"
    exit
fi

root=`realpath $1`
cd edited_files

# Clone the original DETR repository
git clone https://github.com/facebookresearch/detr.git ../detr

cp main.py ../detr/
cp engine.py ../detr/
cp datasets/* ../detr/datasets/
cp models/* ../detr/models/

# Create COCO annotations for DeepFashion2
python3 convert_fashion_to_coco.py --fashion-root "$root"