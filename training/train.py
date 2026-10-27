
import os
import sys
import time
import argparse

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.dataset import WIDERFaceDataset, collate_fn
from data.augment import apply_augmentations
from data.anchors import generate_anchor_grid, generate_anchor_grid_per_level
from model.backbone import MobileBackbone
from model.neck import FPN
from model.head import DetectionHead
from model.loss import MultiTaskLoss, decode_boxes
from training.scheduler import WarmupCosineScheduler, EarlyStopping
from training.checkpoint import CheckpointManager
from training.metrics import compute_map, evaluate_wider_face
from training.ema import ModelEMA


# ---------------------------------------------------------------------------
# Anchor-relative encoding / decoding
# ---------------------------------------------------------------------------

def encode_regression_targets(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode GT boxes as anchor-relative deltas.

    Parameters
    ----------
    gt_boxes : (N, 4) [cx, cy, w, h] normalised
    anchors  : (N, 4) [cx, cy, w, h] normalised

    Returns
    -------
    deltas : (N, 4) [dx, dy, dw, dh]
    """
    dx = (gt_boxes[:, 0] - anchors[:, 0]) / anchors[:, 2].clamp(min=1e-7)
    dy = (gt_boxes[:, 1] - anchors[:, 1]) / anchors[:, 3].clamp(min=1e-7)
    dw = torch.log(gt_boxes[:, 2].clamp(min=1e-7) / anchors[:, 2].clamp(min=1e-7))
    dh = torch.log(gt_boxes[:, 3].clamp(min=1e-7) / anchors[:, 3].clamp(min=1e-7))
    return torch.stack([dx, dy, dw, dh], dim=1)


def decode_boxes_np(deltas: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Numpy version of decode_boxes for inference/validation."""
    pred_cx = deltas[:, 0] * anchors[:, 2] + anchors[:, 0]
    pred_cy = deltas[:, 1] * anchors[:, 3] + anchors[:, 1]
    pred_w = np.exp(np.clip(deltas[:, 2], None, 10.0)) * anchors[:, 2]
    pred_h = np.exp(np.clip(deltas[:, 3], None, 10.0)) * anchors[:, 3]
    return np.stack([pred_cx, pred_cy, pred_w, pred_h], axis=1)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms_np(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    """Standard greedy NMS. Returns kept indices."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=int)

    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)

        remaining = np.where(iou <= iou_threshold)[0]
        order = order[remaining + 1]  # +1 because we excluded order[0]

    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# Anchor target assignment (assigns GT boxes to anchors)
# ---------------------------------------------------------------------------

def assign_targets(anchors: torch.Tensor, gt_boxes: torch.Tensor,
                   num_boxes: torch.Tensor, img_size: int = 640,
                   pos_iou: float = 0.4, neg_iou: float = 0.3):
    """Match ground-truth boxes to anchors via IoU.

    Parameters
    ----------
    anchors : (A, 4) [cx, cy, w, h] normalised
    gt_boxes : (B, max_boxes, 4) [cx, cy, w, h] normalised
    num_boxes : (B,)

    Returns
    -------
    cls_targets : (B, A) — 0=bg, 1=face, -1=ignore
    reg_targets : (B, A, 4) — anchor-relative deltas [dx, dy, dw, dh]
    """
    B = gt_boxes.shape[0]
    A = anchors.shape[0]
    device = gt_boxes.device

    cls_targets = torch.zeros(B, A, dtype=torch.long, device=device)
    reg_targets = torch.zeros(B, A, 4, dtype=torch.float32, device=device)

    # Pre-compute anchor corners
    a_x1 = anchors[:, 0] - anchors[:, 2] / 2
    a_y1 = anchors[:, 1] - anchors[:, 3] / 2
    a_x2 = anchors[:, 0] + anchors[:, 2] / 2
    a_y2 = anchors[:, 1] + anchors[:, 3] / 2

    for i in range(B):
        n = num_boxes[i].item()
        if n == 0:
            continue

        gt = gt_boxes[i, :n]  # (n, 4)

        # GT corners
        g_x1 = gt[:, 0] - gt[:, 2] / 2
        g_y1 = gt[:, 1] - gt[:, 3] / 2
        g_x2 = gt[:, 0] + gt[:, 2] / 2
        g_y2 = gt[:, 1] + gt[:, 3] / 2

        # IoU matrix (A, n)
        inter_x1 = torch.max(a_x1[:, None], g_x1[None, :])
        inter_y1 = torch.max(a_y1[:, None], g_y1[None, :])
        inter_x2 = torch.min(a_x2[:, None], g_x2[None, :])
        inter_y2 = torch.min(a_y2[:, None], g_y2[None, :])
        inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)

        area_a = (a_x2 - a_x1) * (a_y2 - a_y1)
        area_g = (g_x2 - g_x1) * (g_y2 - g_y1)
        union = area_a[:, None] + area_g[None, :] - inter
        iou = inter / (union + 1e-9)  # (A, n)

        # Best GT per anchor
        if n > 0:
            max_iou, best_gt_idx = iou.max(dim=1)

            # Assign positives / negatives
            cls_targets[i] = torch.where(max_iou >= pos_iou, torch.tensor(1, device=device),
                             torch.where(max_iou < neg_iou, torch.tensor(0, device=device),
                                         torch.tensor(-1, device=device)))  # -1 = ignore

            # Regression targets for positive anchors — encode as deltas
            pos_mask = cls_targets[i] == 1
            if pos_mask.sum() > 0:
                matched_gt = gt[best_gt_idx[pos_mask]]
                matched_anchors = anchors[pos_mask]
                reg_targets[i, pos_mask] = encode_regression_targets(matched_gt, matched_anchors)

            # Ensure each GT has at least one matching anchor (forced match)
            # but only if the IoU is somewhat reasonable (> 0.2).
            # This prevents tiny/far faces from ruining the regression branch.
            for g_idx in range(n):
                # box is [cx, cy, w, h]
                # Skip boxes that are too small to be meaningful (e.g. < 2px)
                if gt[g_idx, 2] * img_size < 2 or gt[g_idx, 3] * img_size < 2:
                    continue

                best_iou_val, best_a = iou[:, g_idx].max(dim=0)
                if best_iou_val > 0.15:
                    cls_targets[i, best_a] = 1
                    reg_targets[i, best_a] = encode_regression_targets(
                        gt[g_idx].unsqueeze(0), anchors[best_a].unsqueeze(0)
                    ).squeeze(0)

    return cls_targets, reg_targets


# ---------------------------------------------------------------------------
# Full model wrapper
# ---------------------------------------------------------------------------

class FaceDetectionModel(nn.Module):
    """Backbone + FPN Neck + Detection Head."""

    def __init__(self, cfg: dict):
        super().__init__()
        width_mult = cfg["model"].get("backbone_width_mult", 1.0)
        fpn_ch = cfg["model"].get("fpn_channels", 256)
        num_anchors = cfg["model"].get("num_anchors", 9)
        num_classes = cfg["model"].get("num_classes", 1)
        landmark_enabled = cfg["model"].get("landmark_enabled", False)

        self.backbone = MobileBackbone(width_mult=width_mult)
        self.neck = FPN(self.backbone.out_channels, out_channels=fpn_ch)
        self.head = DetectionHead(
            in_channels=fpn_ch,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_landmarks=5 if landmark_enabled else 0,
        )

    def forward(self, x):
        features = self.backbone(x)
        fpn_out = self.neck(features)
        cls, reg, lmk = self.head(fpn_out)
        return cls, reg, lmk


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Advanced Optimizers
# ---------------------------------------------------------------------------

class Lookahead(torch.optim.Optimizer):
    """Lookahead implementation: https://arxiv.org/abs/1907.08610
    
    Maintains 'slow weights' that follow 'fast weights' of an inner optimizer.
    """
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = self.optimizer.state
        self.fast_state = self.optimizer.state
        self.defaults = self.optimizer.defaults
        self.slow_weights = [[p.data.clone().detach() for p in group['params']]
                             for group in self.param_groups]
        self._step_count = 0

    def zero_grad(self, set_to_none=False):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count += 1
        if self._step_count % self.k == 0:
            for group, slow_weights in zip(self.param_groups, self.slow_weights):
                for p, slow_p in zip(group['params'], slow_weights):
                    if p.grad is None:
                        continue
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    p.data.copy_(slow_p)
        return loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)


class TrainTransform:
    """Picklable callable for dataset augmentations."""
    def __init__(self, aug_cfg):
        self.aug_cfg = aug_cfg
        
    def __call__(self, img, boxes, difficulties):
        from data.augment import apply_augmentations
        return apply_augmentations(img, boxes, self.aug_cfg, difficulties=difficulties)


class Trainer:
    """Manages the full training workflow."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Build model
        self.model = FaceDetectionModel(cfg).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")

        # EMA initialization
        self.ema = ModelEMA(self.model, decay=0.9999)

        # Optimizer - Switching to AdamW for faster convergence + Lookahead wrapper
        train_cfg = cfg["train"]
        base_optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["lr"],
            weight_decay=train_cfg.get("weight_decay", 0.01),
        )
        self.optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)

        # Scheduler
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=train_cfg["warmup_epochs"],
            max_epochs=train_cfg["max_epochs"],
            base_lr=train_cfg["lr"],
            min_lr=train_cfg.get("min_lr", 1e-6),
        )

        # Loss
        loss_cfg = cfg["loss"]
        self.criterion = MultiTaskLoss(
            cls_weight=loss_cfg.get("cls_weight", 1.0),
            reg_weight=loss_cfg.get("reg_weight", 2.0),
            lmk_weight=loss_cfg.get("lmk_weight", 0.5),
            focal_alpha=loss_cfg.get("focal_alpha", 0.25),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            neg_ratio=loss_cfg.get("neg_ratio", 3),
        )

        # Early stopping
        es_cfg = cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 15),
            min_delta=es_cfg.get("min_delta", 0.0),
            min_epochs=es_cfg.get("min_epochs", 0),
        )

        # Checkpoint manager
        ckpt_cfg = cfg.get("checkpoint", {})
        self.ckpt_mgr = CheckpointManager(
            save_dir=ckpt_cfg.get("save_dir", "checkpoints"),
            keep_top_k=ckpt_cfg.get("keep_top_k", 3),
        )

        # AMP scaler
        self.use_amp = train_cfg.get("amp", True) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Generate anchors for target assignment
        img_size = cfg["model"]["image_size"]
        # Feature map sizes at strides 8, 16, 32
        self.feature_sizes = [
            (img_size // 8, img_size // 8),
            (img_size // 16, img_size // 16),
            (img_size // 32, img_size // 32),
        ]
        scales = cfg["model"].get("anchor_scales", [16, 32, 64, 128, 256, 512])
        ratios = cfg["model"].get("anchor_ratios", [1.0, 1.25, 1.5])
        # Build anchor wh — scales[i] is width, ratios[i] is w/h for each anchor (paired, not cross-product)
        import numpy as np
        num_anchors_total = len(scales)  # total k-means anchors (e.g. 9)
        anchors_per_level = cfg["model"].get("num_anchors", 3)  # per FPN level
        if len(scales) == len(ratios):
            # Paired format from k-means: scales = widths, ratios = w/h
            anchor_wh = np.array([[s, s / r] for s, r in zip(scales, ratios)], dtype=np.float32)
        else:
            # Legacy format: cross-product of scales × ratios
            anchor_wh = []
            for s in scales:
                for r in ratios:
                    anchor_wh.append([s * np.sqrt(r), s / np.sqrt(r)])
            anchor_wh = np.array(anchor_wh[:num_anchors_total], dtype=np.float32)

        # Sort by area and use per-level assignment (3 anchors per FPN level)
        areas = anchor_wh[:, 0] * anchor_wh[:, 1]
        anchor_wh = anchor_wh[np.argsort(areas)]
        self.anchor_wh = anchor_wh
        self.anchors_per_level = anchors_per_level
        self.anchors = torch.from_numpy(
            generate_anchor_grid_per_level(
                self.feature_sizes, anchor_wh, img_size,
                anchors_per_level=anchors_per_level,
            )
        ).to(self.device)
        self._anchors_cache = {img_size: self.anchors}
        print(f"Anchors: {self.anchors.shape[0]} total "
              f"({anchors_per_level} per cell × {len(self.feature_sizes)} levels)")

        self.start_epoch = 0
        self.best_map = 0.0

    def _build_dataloader(self, split: str) -> DataLoader:
        """Build train or val DataLoader."""
        cfg = self.cfg
        paths = cfg["paths"]
        train_cfg = cfg["train"]
        aug_cfg = cfg.get("augment", {})

        is_train = (split == "train")
        images_root = paths["train_images"] if is_train else paths["val_images"]
        ann_path = paths["train_ann"] if is_train else paths["val_ann"]

        transform = None
        if is_train:
            transform = TrainTransform(aug_cfg)

        dataset = WIDERFaceDataset(
            images_root=images_root,
            ann_path=ann_path,
            img_size=cfg["model"]["image_size"],
            stride=32,
            difficulty="all",
            transform=transform,
            mosaic_prob=aug_cfg.get("mosaic_prob", 0.5) if is_train else 0.0,
        )
        print(f"Dataset {split}: {dataset}")

        loader = DataLoader(
            dataset,
            batch_size=train_cfg["batch_size"],
            shuffle=is_train,
            num_workers=train_cfg.get("num_workers", 4),
            pin_memory=train_cfg.get("pin_memory", True),
            persistent_workers=train_cfg.get("persistent_workers", True),
            collate_fn=collate_fn,
            drop_last=is_train,
        )
        return loader

    def _train_one_epoch(self, loader: DataLoader, epoch: int):
        """Run one training epoch."""
        self.model.train()
        running = {"cls_loss": 0, "reg_loss": 0, "lmk_loss": 0, "total_loss": 0}
        num_batches = 0
        grad_clip = self.cfg["train"].get("gradient_clip", 10.0)

        for batch_idx, (images, labels, diffs, num_boxes) in enumerate(loader):
            # MULTI-SCALE TRAINING: Every 10 batches, pick a new resolution
            # Range: [0.75 * 640, 1.25 * 640] = [480, 800]
            if batch_idx % 10 == 0:
                base_sz = self.cfg["model"]["image_size"]
                scale = np.random.choice([0.75, 0.875, 1.0, 1.125, 1.25])
                curr_sz = int(base_sz * scale // 32 * 32)
                
            if curr_sz != images.shape[-1]:
                images = torch.nn.functional.interpolate(images, size=(curr_sz, curr_sz), 
                                                        mode='bilinear', align_corners=False)

            if curr_sz not in self._anchors_cache:
                from data.anchors import generate_anchor_grid_per_level
                feat_sizes = [(curr_sz // 8, curr_sz // 8), (curr_sz // 16, curr_sz // 16), (curr_sz // 32, curr_sz // 32)]
                self._anchors_cache[curr_sz] = torch.from_numpy(
                    generate_anchor_grid_per_level(feat_sizes, self.anchor_wh, curr_sz, self.anchors_per_level)
                ).to(self.device)
            curr_anchors = self._anchors_cache[curr_sz]

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            num_boxes = num_boxes.to(self.device, non_blocking=True)

            # Assign targets
            cls_targets, reg_targets = assign_targets(
                curr_anchors, labels, num_boxes, 
                img_size=curr_sz
            )

            # Forward + loss
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    cls_preds, reg_preds, lmk_preds = self.model(images)
                    total_loss, loss_dict = self.criterion(
                        cls_preds, reg_preds, lmk_preds,
                        cls_targets, reg_targets, curr_anchors
                    )
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                cls_preds, reg_preds, lmk_preds = self.model(images)
                total_loss, loss_dict = self.criterion(
                    cls_preds, reg_preds, lmk_preds,
                    cls_targets, reg_targets, self.anchors
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                self.optimizer.step()

            # Update EMA weights
            self.ema.update(self.model)

            for k, v in loss_dict.items():
                running[k] += v.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                avg = {k: v / num_batches for k, v in running.items()}
                print(f"  [{epoch}][{batch_idx + 1}/{len(loader)}] "
                      f"cls={avg['cls_loss']:.4f} reg={avg['reg_loss']:.4f} "
                      f"total={avg['total_loss']:.4f}")

        avg = {k: v / max(num_batches, 1) for k, v in running.items()}
        return avg


    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> dict:
        """Run validation and compute mAP."""
        # Use EMA weights for validation if available
        val_model = self.ema.ema
        val_model.eval()

        predictions = []
        ground_truths = []

        anchors_np = self.anchors.cpu().numpy()
        nms_iou = self.cfg.get("inference", {}).get("nms_iou", 0.45)

        print(f"\n  [Val] Evaluating {len(loader)} batches...")
        for batch_idx, (images, labels, diffs, num_boxes) in enumerate(loader):
            if (batch_idx + 1) % 5 == 0:
                print(f"    processing batch {batch_idx + 1}/{len(loader)}...")

            images = images.to(self.device, non_blocking=True)
            outputs = val_model(images)
            cls_preds, reg_preds = outputs[0], outputs[1]

            # Sigmoid on classification
            cls_scores = cls_preds.sigmoid().squeeze(-1)  # (B, A)

            B = images.shape[0]
            for i in range(B):
                scores = cls_scores[i].cpu().numpy()
                deltas = reg_preds[i].cpu().numpy()
                n = num_boxes[i].item()
                gt_boxes = labels[i, :n].cpu().numpy()
                gt_diffs = diffs[i, :n].cpu().numpy()

                # Decode deltas to absolute boxes using anchors
                pred_boxes_cxcywh = decode_boxes_np(deltas, anchors_np)

                # Clamp to [0, 1]
                pred_boxes_cxcywh = np.clip(pred_boxes_cxcywh, 0, 1)

                # Filter by score (lower threshold during early training to see progress)
                mask = scores > 0.01
                pred_boxes_cxcywh = pred_boxes_cxcywh[mask]
                pred_scores = scores[mask]

                if len(pred_scores) == 0:
                    predictions.append({
                        "boxes": np.zeros((0, 4), dtype=np.float32),
                        "scores": np.zeros((0,), dtype=np.float32),
                    })
                    ground_truths.append({
                        "boxes": gt_boxes,
                        "difficulties": gt_diffs,
                    })
                    continue

                # Keep top 2000 before NMS to speed things up (increased for high recall)
                if len(pred_scores) > 2000:
                    order = np.argsort(-pred_scores)[:2000]
                    pred_boxes_cxcywh = pred_boxes_cxcywh[order]
                    pred_scores = pred_scores[order]

                # Convert [cx, cy, w, h] to [x1, y1, x2, y2] for NMS
                boxes_xyxy = np.zeros_like(pred_boxes_cxcywh)
                boxes_xyxy[:, 0] = pred_boxes_cxcywh[:, 0] - pred_boxes_cxcywh[:, 2] / 2
                boxes_xyxy[:, 1] = pred_boxes_cxcywh[:, 1] - pred_boxes_cxcywh[:, 3] / 2
                boxes_xyxy[:, 2] = pred_boxes_cxcywh[:, 0] + pred_boxes_cxcywh[:, 2] / 2
                boxes_xyxy[:, 3] = pred_boxes_cxcywh[:, 1] + pred_boxes_cxcywh[:, 3] / 2

                # Apply NMS
                keep = nms_np(boxes_xyxy, pred_scores, iou_threshold=nms_iou)

                # Keep top 1000 after NMS (WIDER FACE standard)
                if len(keep) > 1000:
                    keep = keep[:1000]

                # mAP metric expects [cx, cy, w, h] format
                final_boxes = pred_boxes_cxcywh[keep]
                final_scores = pred_scores[keep]

                predictions.append({
                    "boxes": final_boxes,
                    "scores": final_scores,
                })
                ground_truths.append({
                    "boxes": gt_boxes,
                    "difficulties": gt_diffs,
                })

        # Debug stats
        total_preds = sum(len(p['boxes']) for p in predictions)
        total_gt = sum(len(g['boxes']) for g in ground_truths)
        if total_gt == 0:
            print("  [Val] No ground-truth boxes in validation set.")
            return {"mAP": 0.0}
        else:
            print(f"  [Val] total_gt={total_gt}, total_preds={total_preds}")

        results = evaluate_wider_face(predictions, ground_truths, iou_threshold=0.5)
        
        print(f"  [Val] Easy mAP:   {results['easy']['mAP']:.4f}")
        print(f"  [Val] Medium mAP: {results['medium']['mAP']:.4f}")
        print(f"  [Val] Hard mAP:   {results['hard']['mAP']:.4f}")
        
        # We return the overall result to maintain compatibility with the rest of the script
        return results["overall"]

    def train(self, resume: bool = False, checkpoint_path: str = None, from_scratch: bool = False):
        """Main training loop."""
        cfg = self.cfg
        train_cfg = cfg["train"]
        max_epochs = train_cfg["max_epochs"]

        # Determine checkpoint to load
        ckpt_to_load = checkpoint_path
        if not from_scratch:
            if ckpt_to_load is None:
                ckpt_to_load = CheckpointManager.find_latest(
                    cfg.get("checkpoint", {}).get("save_dir", "checkpoints")
                )

        if ckpt_to_load and os.path.exists(ckpt_to_load) and not from_scratch:
            # Load weights and states
            loaded_epoch, metrics = CheckpointManager.load(
                ckpt_to_load, self.model, self.optimizer,
                self.scheduler, self.early_stopping,
                device=self.device, ema=self.ema
            )
            
            # Start from the next epoch
            self.start_epoch = loaded_epoch + 1

            # Update scheduler and optimizer parameters from config.yaml for continued training
            self.scheduler.max_epochs = max_epochs
            if "lr" in train_cfg:
                self.scheduler.base_lr = train_cfg["lr"]
            if "min_lr" in train_cfg:
                self.scheduler.min_lr = train_cfg["min_lr"]
            if "warmup_epochs" in train_cfg:
                self.scheduler.warmup_epochs = train_cfg["warmup_epochs"]

            for pg in self.optimizer.param_groups:
                pg["lr"] = train_cfg.get("lr", pg.get("lr", 1e-4))
                if "weight_decay" in train_cfg:
                    pg["weight_decay"] = train_cfg["weight_decay"]

            # Set best mAP and early stopping baselines
            self.best_map = metrics.get("mAP", 0.0)
            self.early_stopping.best_score = self.best_map
            self.early_stopping.counter = 0
            es_cfg = cfg.get("early_stopping", {})
            self.early_stopping.min_epochs = self.start_epoch + es_cfg.get("min_epochs", 5)

            print(f"✅ Resuming Training from {ckpt_to_load} (Epoch {self.start_epoch}/{max_epochs}, Base mAP: {self.best_map:.4f})")
        else:
            print("Starting training from scratch (epoch 0).")

        # Data loaders
        train_loader = self._build_dataloader("train")
        val_loader = self._build_dataloader("val")

        print(f"\nStarting training: epochs {self.start_epoch}-{max_epochs}")
        print(f"   AMP: {self.use_amp}")

        for epoch in range(self.start_epoch, max_epochs):
            t0 = time.time()

            # LR update
            self.scheduler.step(epoch)
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{max_epochs}  |  LR: {current_lr:.6f}")
            print(f"{'='*60}")

            # Train
            train_losses = self._train_one_epoch(train_loader, epoch)
            print(f"  [Train] cls={train_losses['cls_loss']:.4f} "
                  f"reg={train_losses['reg_loss']:.4f} "
                  f"total={train_losses['total_loss']:.4f}")

            # Validate
            val_result = self._validate(val_loader)
            mAP = val_result["mAP"]
            print(f"  [Val]   mAP@0.5 = {mAP:.4f}")

            elapsed = time.time() - t0
            print(f"  Epoch time: {elapsed:.1f}s")

            # Checkpoint
            metrics = {"mAP": mAP, **train_losses}
            self.ckpt_mgr.save(
                self.model, self.optimizer, self.scheduler,
                epoch, metrics, self.early_stopping,
                ema=self.ema
            )

            if mAP > self.best_map:
                self.best_map = mAP
                print(f"  New best mAP: {mAP:.4f}")

            # Early stopping
            if self.early_stopping(mAP, epoch):
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {self.early_stopping.patience} epochs; "
                      f"best={self.early_stopping.best_score:.4f})")
                break

        print(f"\nTraining complete. Best mAP: {self.best_map:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train face detection model")
    parser.add_argument("--config", default="config.yaml", help="Path to config")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint (auto-resume is also enabled if checkpoint exists)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to specific checkpoint to resume/fine-tune from",
    )
    parser.add_argument(
        "--from-scratch",
        action="store_true",
        help="Train from scratch, ignoring existing checkpoints",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    trainer = Trainer(cfg)
    trainer.train(resume=args.resume, checkpoint_path=args.checkpoint, from_scratch=args.from_scratch)


if __name__ == "__main__":
    main()
