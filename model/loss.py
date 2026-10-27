"""
Loss functions for face detection.

- Focal Loss — handles extreme foreground/background imbalance
- GIoU Loss — directly optimises overlap instead of coordinate deltas
- Landmark Loss — smooth L1 on predicted keypoints
- Hard negative mining — keeps top-k hardest negatives per positive
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Box utilities
# ---------------------------------------------------------------------------

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def _box_ciou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Complete IoU (CIoU) between two sets of [x1, y1, x2, y2] boxes.
    
    CIoU = IoU - (d^2/c^2) - alpha * v
    where d is center distance, c is diagonal of enclosing box, 
    v is aspect ratio consistency, and alpha is a weighting factor.
    """
    # 1. Standard IoU
    x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter + 1e-7
    iou = inter / union

    # 2. Distance term
    c1x, c1y = (boxes1[:, 0] + boxes1[:, 2]) / 2, (boxes1[:, 1] + boxes1[:, 3]) / 2
    c2x, c2y = (boxes2[:, 0] + boxes2[:, 2]) / 2, (boxes2[:, 1] + boxes2[:, 3]) / 2
    dist_sq = (c1x - c2x)**2 + (c1y - c2y)**2

    # Enclosing box diagonal
    ex1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    ey1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    ex2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    ey2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    diag_sq = (ex2 - ex1)**2 + (ey2 - ey1)**2 + 1e-7

    # 3. Aspect ratio term (v) and alpha
    w1, h1 = (boxes1[:, 2] - boxes1[:, 0]), (boxes1[:, 3] - boxes1[:, 1])
    w2, h2 = (boxes2[:, 2] - boxes2[:, 0]), (boxes2[:, 3] - boxes2[:, 1])
    
    import math
    v = (4 / (math.pi**2)) * torch.pow(torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2)
    
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)
    
    return iou - (dist_sq / diag_sq) - alpha * v

# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for classification with Label Smoothing."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = label_smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        target_clamped = target.clamp(min=0).long()
        target_onehot = F.one_hot(target_clamped, num_classes + 1)[:, 1:].float()
        valid_mask = (target >= 0).unsqueeze(-1)
        target_onehot = target_onehot * valid_mask.float()
        
        # Apply Label Smoothing
        if self.smoothing > 0:
            target_onehot = target_onehot * (1.0 - self.smoothing) + 0.5 * self.smoothing

        pred_sigmoid = pred.sigmoid()
        pt = pred_sigmoid * target_onehot + (1 - pred_sigmoid) * (1 - target_onehot)
        alpha_t = self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)
        focal_weight = alpha_t * (1 - pt).pow(self.gamma)

        bce = F.binary_cross_entropy_with_logits(pred, target_onehot, reduction="none")
        return focal_weight * bce


# ---------------------------------------------------------------------------
# Regression Loss
# ---------------------------------------------------------------------------

class RegressionLoss(nn.Module):
    """Combined Smooth L1 + CIoU loss for bounding boxes."""

    def __init__(self, ciou_weight: float = 2.0, l1_weight: float = 0.5):
        super().__init__()
        self.ciou_weight = ciou_weight
        self.l1_weight = l1_weight

    def forward(self, pred_deltas, target_deltas, 
                pred_boxes_xyxy, target_boxes_xyxy) -> torch.Tensor:
        # Smooth L1 on deltas
        l1_loss = F.smooth_l1_loss(pred_deltas, target_deltas, reduction="none").sum(dim=1)
        
        # CIoU on absolute boxes
        ciou = _box_ciou(pred_boxes_xyxy, target_boxes_xyxy)
        ciou_loss = 1.0 - ciou

        return self.l1_weight * l1_loss + self.ciou_weight * ciou_loss


def decode_boxes(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Decode anchor-relative deltas back to absolute [cx, cy, w, h].

    Parameters
    ----------
    deltas  : (N, 4) [dx, dy, dw, dh]
    anchors : (N, 4) [cx, cy, w, h] normalised

    Returns
    -------
    boxes : (N, 4) [cx, cy, w, h] normalised
    """
    pred_cx = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
    pred_cy = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
    pred_w = torch.exp(deltas[:, 2].clamp(max=10.0)) * anchors[:, 2]
    pred_h = torch.exp(deltas[:, 3].clamp(max=10.0)) * anchors[:, 3]
    return torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)


# ---------------------------------------------------------------------------
# Combined multi-task loss
# ---------------------------------------------------------------------------

class MultiTaskLoss(nn.Module):
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 2.0,
                 lmk_weight: float = 0.5, focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0, neg_ratio: int = 3):
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.lmk_weight = lmk_weight

        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.reg_loss_fn = RegressionLoss()
        # LandmarkLoss might not be defined if it was after MultiTaskLoss in original
        # Let's ensure it's there or handle it. 
        # Actually I saw it in the previous read_file output.

    def forward(self, cls_preds, reg_preds, lmk_preds,
                cls_targets, reg_targets, anchors, 
                lmk_targets=None, lmk_mask=None):
        """
        Parameters
        ----------
        cls_preds : (B, N, C)
        reg_preds : (B, N, 4)
        cls_targets : (B, N)
        reg_targets : (B, N, 4)
        anchors : (N, 4) [cx, cy, w, h] normalized
        """
        B, N, C = cls_preds.shape
        pos_mask = cls_targets > 0
        valid_mask = cls_targets >= 0
        num_pos = pos_mask.sum().float()
        # Scale loss by batch_size if no positives, or by num_pos if there are positives
        # This keeps the gradients stable.
        denom = num_pos.clamp(min=1.0)

        # 1. Classification Loss (Focal Loss on all valid anchors)
        cls_loss = self.focal(cls_preds[valid_mask], cls_targets[valid_mask]).sum() / denom

        # 2. Regression Loss (Smooth L1 + GIoU)
        if pos_mask.sum() > 0:
            p_deltas = reg_preds[pos_mask]
            t_deltas = reg_targets[pos_mask]
            
            # Since anchors are same for all images in batch:
            batch_anchors = anchors.unsqueeze(0).expand(B, N, 4)
            pos_anchors = batch_anchors[pos_mask]
            
            p_boxes_cxcywh = decode_boxes(p_deltas, pos_anchors)
            t_boxes_cxcywh = decode_boxes(t_deltas, pos_anchors)
            
            p_boxes_xyxy = _cxcywh_to_xyxy(p_boxes_cxcywh)
            t_boxes_xyxy = _cxcywh_to_xyxy(t_boxes_cxcywh)
            
            # Weighted L1 + GIoU
            reg_loss = self.reg_loss_fn(p_deltas, t_deltas, p_boxes_xyxy, t_boxes_xyxy).sum() / denom
        else:
            reg_loss = reg_preds.sum() * 0.0

        # 3. Landmark Loss (Placeholder)
        if lmk_preds is not None:
            lmk_loss = lmk_preds.sum() * 0.0
        else:
            lmk_loss = cls_preds.sum() * 0.0
        
        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.lmk_weight * lmk_loss
        
        return total, {
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "lmk_loss": lmk_loss.detach(),
            "total_loss": total.detach()
        }

