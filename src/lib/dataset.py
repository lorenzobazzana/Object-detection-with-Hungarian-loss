import json
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.tv_tensors import BoundingBoxes
from torchvision.transforms.v2.functional import convert_bounding_box_format, resize

class FashionDataset(Dataset):
    def __init__(self, split, anno_path, img_path, transforms=None):
        super().__init__()
        self.split = split
        self.anno_path = anno_path
        self.img_path = img_path
        self.transforms = transforms

        self.anno = [f for f in glob.glob(os.path.join(anno_path, '*')) if os.path.isfile(f)]

    def __len__(self):
        return len(self.anno)
    
    def __getitem__(self, index):
        label_file = self.anno[index]

        with open(label_file, 'r') as f:
            labels = json.load(f)

        classes, bboxes = zip(*[(labels[key]["category_id"], labels[key]["bounding_box"]) for key in labels.keys() if "item" in key])
        
        img_basename = os.path.basename(label_file).replace(".json", ".jpg")
        img = Image.open(os.path.join(self.img_path, img_basename))
        w,h = img.size

        #classes = [c-1 for c in classes] # In the GT, the classes go from 1 to 13
        classes = np.array(classes)
        #bboxes = self.resize_box(bboxes, w, h)
        bboxes = BoundingBoxes(bboxes, format="XYXY", canvas_size=(h,w), dtype=torch.float32)

        if self.transforms is not None:
            transforms_all = self.transforms.get("all", None)
            transforms_img = self.transforms.get("img", None)
            
            if transforms_all is not None:
                img, bboxes = transforms_all(img, bboxes)
            if transforms_img is not None:
                img = transforms_img(img)
            h, w = img.shape[-2:]
        bboxes = convert_bounding_box_format(bboxes, new_format="CXCYWH")
        bboxes = self.resize_box(bboxes, w, h)

        return img, classes, bboxes
    
    def resize_box(self, boxes, w, h):
        resized_boxes = boxes / torch.tensor([w,h,w,h])
        return resized_boxes
