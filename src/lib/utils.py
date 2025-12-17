from typing import Literal

import wandb
import tqdm

import torch
import torchvision.transforms.v2 as transforms

def collate(batch):
    images, categories, boxes = zip(*batch)

    images = torch.stack(images)


    categories = [torch.tensor(cat) for cat in categories]
    boxes = [torch.tensor(box) for box in boxes]
    return images, categories, boxes

def create_transforms(split: Literal["train", "test"], use_imagenet_norm=False):

    if use_imagenet_norm:
        norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    else:
        norm = transforms.Normalize(mean=[0.6068, 0.5726, 0.5510], std=[0.2902, 0.2961, 0.2872]) # DeepFashion2 mean and std                
                                
    if split == "train":
        transforms_all = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomApply(torch.nn.ModuleList([
                transforms.RandomResizedCrop(224, scale=(0.7,1)),
            ]), p=0.2),
            transforms.RandomZoomOut(fill=255, side_range=(1.2,1.8),p=0.2),
            transforms.Resize((224,224)),
        ])

        transforms_img = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            norm,
        ])     


    else:
        transforms_all = transforms.Resize((224,224))
        transforms_img = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            norm,
        ])

    transforms_to_apply = {
        "all": transforms_all,
        "img": transforms_img
    }

    return transforms_to_apply
    

def train_one_epoch(model, loss_fn, optimizer, dataloader, device):
    epoch_loss = 0.
    epoch_aux = {"class_loss": 0., "box_loss":0., "giou_loss":0.}
    for i, (images, categories, boxes) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        optimizer.zero_grad()
        images = images.to(device)
        categories = [cat.to(device) for cat in categories]
        boxes = [box.to(device) for box in boxes]
        pred_categories, pred_boxes = model(images)
        
        loss, aux = loss_fn(pred_categories, categories, pred_boxes, boxes)

        loss.backward()
        optimizer.step()    
        
        epoch_loss += loss.item()
        for key in epoch_aux.keys():
            epoch_aux[key] += aux[key]

    return epoch_loss/len(dataloader), {k:v/len(dataloader) for k,v in epoch_aux.items()} 

def test(model, loss_fn, dataloader, device):
    model.eval()
    epoch_loss = 0.
    epoch_aux = {"class_loss": 0., "box_loss":0., "giou_loss":0.}
    with torch.no_grad():
        for i, (images, categories, boxes) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            images = images.to(device)
            categories = [cat.to(device) for cat in categories]
            boxes = [box.to(device) for box in boxes]
            pred_categories, pred_boxes = model(images)
            
            loss, aux = loss_fn(pred_categories, categories, pred_boxes, boxes)

            epoch_loss += loss.item()
            for key in epoch_aux.keys():
                epoch_aux[key] += aux[key]
    
    model.train()

    return epoch_loss/len(dataloader), {k:v/len(dataloader) for k,v in epoch_aux.items()}

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
        train_epoch_loss, train_aux = train_one_epoch(model, loss_fn, optimizer, dataloader["train"], device)
        print(f"Epoch loss (train): {train_epoch_loss}")
        train_losses.append(train_epoch_loss)

        test_epoch_loss, test_aux = test(model, loss_fn, dataloader["test"], device)
        print(f"Epoch loss (test): {test_epoch_loss}")
        test_losses.append(test_epoch_loss)
        loss_fn.step()
        
        if use_wandb:
            wandb.log({
                "train/epoch_loss": train_epoch_loss,
                "test/epoch_loss": test_epoch_loss,
                "train/class_loss": train_aux["class_loss"],
                "train/box_loss": train_aux["box_loss"],
                "train/giou_loss": train_aux["giou_loss"],
                "test/class_loss": test_aux["class_loss"],
                "test/box_loss": test_aux["box_loss"],
                "test/giou_loss": test_aux["giou_loss"],
                "loss_params/class": loss_fn.class_weight.item(),
                "loss_params/box": loss_fn.box_weight.item(),
                "loss_params/giou": loss_fn.giou_weight.item(),
            })


    return train_losses, test_losses
