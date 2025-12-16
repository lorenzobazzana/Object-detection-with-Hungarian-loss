import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
import torch.nn.functional as F
from torchvision.ops import box_area

class Loss(nn.Module):
    def __init__(self, num_classes, class_weight, box_weight, giou_weight, no_detection_coeff=0.1, learn_balance_weights=False, epoch=0, warmup_epochs=3):
        super().__init__()

        self.class_weight = nn.Parameter(torch.tensor(class_weight, dtype=torch.float32), requires_grad=learn_balance_weights)
        self.giou_weight = nn.Parameter(torch.tensor(giou_weight, dtype=torch.float32), requires_grad=learn_balance_weights)
        self.box_weight = nn.Parameter(torch.tensor(box_weight, dtype=torch.float32), requires_grad=learn_balance_weights)

        self.learn_balance_weights = learn_balance_weights
        self.no_detection_coeff = no_detection_coeff
        self.class_balance = torch.ones(num_classes)
        self.class_balance[0] = no_detection_coeff

        self.epoch = epoch
        self.warmup_epochs = warmup_epochs

    def step(self):
        self.epoch += 1

    def hungarian_matcher(self, output_classes, target_classes, output_boxes, target_boxes):

        with torch.no_grad():
            output_classes = output_classes.softmax(-1)
            indices = []
            batch_size = output_classes.shape[0]
            for i in range(batch_size):
                probs = output_classes[i]         # [num_queries, num_classes]
                tgt_ids = target_classes[i]  # [num_target_boxes]
                cost_class = -probs[:, tgt_ids]  # [num_queries, num_target_boxes]
                converted_boxes = self.cwh_box_to_x1x2(output_boxes[i])
                converted_tgt_boxes = self.cwh_box_to_x1x2(target_boxes[i])
                cost_giou = -self.generalized_box_iou(converted_boxes, converted_tgt_boxes)
                cost_boxes = torch.cdist(converted_boxes, converted_tgt_boxes, p=1)
                cost = self.class_weight * cost_class + self.giou_weight * cost_giou + self.box_weight * cost_boxes 
                indices.append(linear_sum_assignment(cost.cpu().numpy()))

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def forward(self, output_classes, target_classes, output_boxes, target_boxes):

        device = output_classes.device
        if self.epoch < self.warmup_epochs:
            # Greedy match (first k predictions with first k targets)
            matched_indices = [(torch.arange(len(tgt)), torch.arange(len(tgt))) for tgt in target_classes]
        else:
            matched_indices = self.hungarian_matcher(output_classes, target_classes, output_boxes, target_boxes)
        indices_src, indices_tgt = zip(*matched_indices)

        batch_info = torch.cat([torch.full_like(matched_outs,fill_value=i) for i, matched_outs in enumerate(indices_src)])
        indices_src = torch.cat(indices_src)

        # Class loss (for matched and unmatched ("no object") boxes)
        _tgt_classes = torch.cat([tgt[idxs] for tgt,idxs in zip(target_classes, indices_tgt)])
        _tgt_classes_full = torch.zeros(output_classes.shape[:2], dtype=torch.long, device=device)
        _tgt_classes_full[batch_info, indices_src] = _tgt_classes

        class_loss = F.cross_entropy(output_classes.transpose(1,2), _tgt_classes_full, self.class_balance.to(device))

        # Box loss (only for matched boxes)
        matched_boxes = output_boxes[batch_info,indices_src]
        matched_boxes = self.cwh_box_to_x1x2(matched_boxes)
        _tgt_boxes = torch.cat([tgt[idxs] for tgt,idxs in zip(target_boxes, indices_tgt)])
        converted_tgt_boxes = self.cwh_box_to_x1x2(_tgt_boxes)

        box_loss = F.l1_loss(matched_boxes, converted_tgt_boxes, reduction='mean')
        giou_loss = 1 - self.generalized_box_iou(matched_boxes, converted_tgt_boxes).diag()
        giou_loss = giou_loss.mean()

        if self.learn_balance_weights:
            cost = (
                torch.exp(-self.class_weight) * class_loss + torch.log1p(torch.exp(self.class_weight)) +
                0.5 * torch.exp(-self.box_weight) * box_loss + torch.log1p(torch.exp(self.box_weight)) +
                0.5 * torch.exp(-self.giou_weight) * giou_loss + torch.log1p(torch.exp(self.giou_weight))
            )
        else:
            cost = self.class_weight * class_loss + self.giou_weight * giou_loss + self.box_weight * box_loss

        return cost, {"class_loss": class_loss, "box_loss":box_loss, "giou_loss":giou_loss}


    @staticmethod
    def cwh_box_to_x1x2(boxes):
        cx, cy, w, h = torch.unbind(boxes, -1)
        x1 = cx - w/2
        x2 = cx + w/2
        y1 = cy - h/2
        y2 = cy + h/2
        return torch.stack([x1,y1,x2,y2], -1)

    @staticmethod
    def box_iou(boxes1, boxes2):
        area1 = box_area(boxes1)
        area2 = box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[..., :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[..., 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    @staticmethod
    def generalized_box_iou(boxes1, boxes2, eps=1e-8):
        """
        Generalized IoU from https://giou.stanford.edu/

        The boxes should be in [x0, y0, x1, y1] format

        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[..., 2:] >= boxes2[..., :2]).all()
        iou, union = Loss.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / (area + eps)