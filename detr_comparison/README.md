# Setting up DETR for training on DeepFashion2

The original DETR code (which can be found [here](https://github.com/facebookresearch/detr/tree/main)) only works with COCO-style annotations. Since DeepFashion2 has a different annotation format, there are some steps that are required before you can start training DETR on DeepFashion2. This folder contains a `setup.sh` script, and a folder named `edited_files`.

- `edited_files` contains some files from the original DETR repo, that have been slightly modified to allow training on datasets that are not strictly COCO (but still have annotations in COCO's format). More precisely, a new dataset type ("fashion") has been added.
- `setup.sh` is a short script that clones the original DETR repo and replaces the files with the corresponding ones contained in `edited_files`. It also creates the right COCO-style annotations starting from the original DeepFashion2 annotations.

In order to start training, first launch `setup.sh`:
```
/bin/bash setup.sh FASHION_DATASET_ROOT
```

Where `FASHION_DATASET_ROOT` is the location of the root directory of the DeepFashion2 dataset. When the script is done, you can now start training DETR as usual, but it is necessary to specify the new dataset type in the train script args:
```
cd detr
python main.py --dataset_file fashion --fashion_path FASHION_DATASET_ROOT
```
The options `--dataset_file fashion` and `--fashion_path FASHION_DATASET_ROOT` specify that the type of the training dataset is DeepFashion2, and the data root location.

IMPORTANT the expected format of the dataset root is:
```
FASHION_DATASET_ROOT
📂
├── 📂 test
│   ├── 📂 annotations
│   ├── 📂 images
│   └──  instances_test_fashion.json
└── 📂 train
    ├── 📂 annotations
    ├── 📂 images
    └──  instances_train_fashion.json
```

`instances_train_fashion.json` and `instances_test_fashion.json` are created by the setup script.