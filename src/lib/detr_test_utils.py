import torch
import detr.util.misc as utils
from PIL import Image
from torch.utils.data import DataLoader
import wandb
import tqdm
import math
import sys

def dataset_item_to_detr_target(item, image_id=None, label_offset=0, include_sizes=True):
    """
    item: (img, classes, bboxes) from FashionDataset
      - img: Tensor [C,H,W] (preferred) or PIL Image
      - classes: np.array/list of shape [N]
      - bboxes: Tensor-like [N,4], normalized CXCYWH (as returned by your dataset)

    Returns a DETR-style target dict (what criterion/matcher uses).
    """
    img, classes, boxes = item

    labels = torch.as_tensor(classes, dtype=torch.int64) - int(label_offset)
    #boxes  = torch.as_tensor(bboxes, dtype=torch.float32)

    # edge cases
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    if boxes.ndim == 1:
        boxes = boxes.unsqueeze(0)

    #boxes = boxes.clamp(0.0, 1.0)

    target = {
        "labels": labels,
        "boxes": boxes,  # normalized cxcywh
    }

    if image_id is not None:
        target["image_id"] = torch.as_tensor([image_id], dtype=torch.int64)

    if include_sizes:
        if torch.is_tensor(img):
            h, w = img.shape[-2:]
        else:
            w, h = img.size
        hw = torch.as_tensor([h, w], dtype=torch.int64)
        target["orig_size"] = hw.clone()
        target["size"] = hw.clone()

    # Optional keys (not needed for loss, but harmless)
    target["iscrowd"] = torch.zeros((labels.shape[0],), dtype=torch.int64)

    return target

def prepare_images_for_detr(images, device=None, do_normalize=True):
    """
    images: list of images, each either:
      - torch.Tensor [C,H,W] (float in [0,1] or uint8), or
      - PIL.Image

    Returns:
      samples: NestedTensor (padded batch + mask) ready for DETR forward.
    """
    processed = []
    for im in images:
        # PIL -> Tensor [C,H,W] float in [0,1]
        if isinstance(im, Image.Image):
            im = torch.to_tensor(im)

        if not torch.is_tensor(im):
            raise TypeError(f"Expected PIL image or torch.Tensor, got {type(im)}")

        #if device is not None:
        #    im = im.to(device)

        processed.append(im)

    samples = utils.nested_tensor_from_tensor_list(processed)
    #if device is not None:
    #    samples = samples.to(device)
    return samples

def detr_collate_fn(batch):
    imgs, targets = [], []
    for i, item in enumerate(batch):
        imgs.append(item[0])
        targets.append(dataset_item_to_detr_target(item, image_id=i, label_offset=1))
    imgs = prepare_images_for_detr(imgs)
    return imgs, targets

def train(model, loss_fn, optimizer, dataloader, device, n_epochs, use_wandb=False):

    if use_wandb:
        wandb.login()
        wandb.init(project="DeepFashion2", config={
            "model": model.name,
        })

    model.train()

    train_losses = []
    test_losses = []
    for i in range(n_epochs):
        print(f"Epoch {i+1}/{n_epochs}")
        train_epoch_loss = train_one_epoch(model, loss_fn, dataloader["train"], optimizer, device, i)
        print(f"Epoch loss (train): {train_epoch_loss}")
        train_losses.append(train_epoch_loss)

        test_epoch_loss = test(model, loss_fn, dataloader["test"], device)
        print(f"Epoch loss (test): {test_epoch_loss}")
        test_losses.append(test_epoch_loss)
        loss_fn.step()
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": train_epoch_loss,
                "test/epoch_loss": test_epoch_loss,
                #"train/class_loss": train_aux["class_loss"],
                #"train/box_loss": train_aux["box_loss"],
                #"train/giou_loss": train_aux["giou_loss"],
                #"test/class_loss": test_aux["class_loss"],
                #"test/box_loss": test_aux["box_loss"],
                #"test/giou_loss": test_aux["giou_loss"],
                #"loss_params/class": loss_fn.class_weight.item(),
                #"loss_params/box": loss_fn.box_weight.item(),
                #"loss_params/giou": loss_fn.giou_weight.item(),
            })


    return train_losses, test_losses

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()

    epoch_loss = 0.
    for samples, targets in tqdm.tqdm(data_loader, total=len(data_loader)):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        epoch_loss += loss_value
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        
    return epoch_loss

def test(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: DataLoader,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.eval()
    criterion.eval()
    epoch_loss = 0.
    with torch.no_grad():
        for samples, targets in tqdm.tqdm(data_loader, total=len(data_loader)):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                        for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()
            epoch_loss += loss_value
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
    return epoch_loss