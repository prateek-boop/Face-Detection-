"""
Face detector — inference pipeline with Soft-NMS and confidence calibration.

Goes beyond basic NMS: Soft-NMS decays scores of overlapping boxes with a
Gaussian kernel instead of hard-deleting them — critical for crowded scenes.
"""

import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model.backbone import MobileBackbone
from model.neck import FPN
from model.head import DetectionHead


class FaceDetector:
    """End-to-end face detection: preprocess → forward → decode → Soft-NMS.

    Parameters
    ----------
    checkpoint_path : str
        Path to a saved training checkpoint (.pt file).
    cfg : dict
        Full config dict (from config.yaml).
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, checkpoint_path: str, cfg: dict, device: str = None):
        self.cfg = cfg
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.img_size = cfg["model"]["image_size"]

        # Build model
        from training.train import FaceDetectionModel
        self.model = FaceDetectionModel(cfg).to(self.device)
        
        # Use FP16 for CUDA to double speed
        self.fp16 = self.device.type == "cuda"
        if self.fp16:
            self.model.half()
            # Enable cuDNN auto-tuner for max speed on fixed size inputs
            torch.backends.cudnn.benchmark = True

        # Load weights safely (prefer high-accuracy smoothed EMA weights if available)
        state = torch.load(checkpoint_path, map_location=self.device,
                           weights_only=True)
        if "ema_state_dict" in state and isinstance(state["ema_state_dict"], dict) and "model" in state["ema_state_dict"]:
            self.model.load_state_dict(state["ema_state_dict"]["model"])
            print(f"[Detector] Loaded EMA weights from {checkpoint_path}")
        elif "ema_state_dict" in state and isinstance(state["ema_state_dict"], dict):
            self.model.load_state_dict(state["ema_state_dict"])
            print(f"[Detector] Loaded EMA weights from {checkpoint_path}")
        else:
            self.model.load_state_dict(state["model_state_dict"])
            print(f"[Detector] Loaded model weights from {checkpoint_path}")
        self.model.eval()

        # Inference config
        inf_cfg = cfg.get("inference", {})
        self.conf_threshold = inf_cfg.get("conf_threshold", 0.5)
        self.nms_iou = inf_cfg.get("nms_iou", 0.45)
        self.soft_nms_sigma = inf_cfg.get("soft_nms_sigma", 0.5)
        self.max_detections = inf_cfg.get("max_detections", 300)

        # Calibration parameters (Platt scaling)
        self._cal_a = 1.0
        self._cal_b = 0.0

        # Build anchor grid
        from data.anchors import generate_anchor_grid_per_level
        feature_sizes = [
            (self.img_size // 8, self.img_size // 8),
            (self.img_size // 16, self.img_size // 16),
            (self.img_size // 32, self.img_size // 32),
        ]
        scales = cfg["model"].get("anchor_scales", [16, 32, 64, 128, 256, 512])
        ratios = cfg["model"].get("anchor_ratios", [1.0, 1.25, 1.5])
        anchors_per_level = cfg["model"].get("num_anchors", 3)
        
        # Build anchor wh
        anchor_wh = np.array([[s, s / r] for s, r in zip(scales, ratios)], dtype=np.float32)
        # Sort by area (to match training)
        areas = anchor_wh[:, 0] * anchor_wh[:, 1]
        anchor_wh = anchor_wh[np.argsort(areas)]
        
        self.anchors = torch.from_numpy(
            generate_anchor_grid_per_level(feature_sizes, anchor_wh, self.img_size, anchors_per_level)
        ).to(self.device)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray):
        """Letterbox resize + normalise (with centering).

        Returns
        -------
        tensor : (1, 3, H, W)
        scale : float (resize scale)
        pad : (off_x, off_y) — offset from top-left
        """
        h, w = image.shape[:2]
        scale = min(self.img_size / h, self.img_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center placement to match training/validation
        dw = self.img_size - new_w
        dh = self.img_size - new_h
        off_x, off_y = dw // 2, dh // 2

        canvas = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        canvas[off_y:off_y + new_h, off_x:off_x + new_w] = resized

        # HWC uint8 → CHW float32 [0, 1]
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))  # CHW

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean) / std

        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)
        if self.fp16:
            tensor = tensor.half()

        return tensor, scale, (off_x, off_y)

    # ------------------------------------------------------------------
    # Soft-NMS
    # ------------------------------------------------------------------

    @staticmethod
    def soft_nms(boxes: np.ndarray, scores: np.ndarray,
                 sigma: float = 0.5, score_threshold: float = 0.01):
        """Soft-NMS with Gaussian decay.

        Parameters
        ----------
        boxes : (N, 4) [x1, y1, x2, y2]
        scores : (N,)
        sigma : float  decay parameter
        score_threshold : float  minimum score to keep

        Returns
        -------
        keep_indices : list of int
        updated_scores : np.ndarray
        """
        N = len(boxes)
        indices = list(range(N))
        updated_scores = scores.copy()

        keep = []

        while len(indices) > 0:
            # Pick highest score
            max_idx = max(indices, key=lambda i: updated_scores[i])
            keep.append(max_idx)
            indices.remove(max_idx)

            if len(indices) == 0:
                break

            # Compute IoU with the picked box
            picked = boxes[max_idx]
            remaining = np.array([indices])

            for idx in list(indices):
                box = boxes[idx]
                inter_x1 = max(picked[0], box[0])
                inter_y1 = max(picked[1], box[1])
                inter_x2 = min(picked[2], box[2])
                inter_y2 = min(picked[3], box[3])
                inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area_a = (picked[2] - picked[0]) * (picked[3] - picked[1])
                area_b = (box[2] - box[0]) * (box[3] - box[1])
                iou = inter / (area_a + area_b - inter + 1e-7)

                # Gaussian decay
                decay = np.exp(-(iou ** 2) / sigma)
                updated_scores[idx] *= decay

                if updated_scores[idx] < score_threshold:
                    indices.remove(idx)

        return keep, updated_scores

    # ------------------------------------------------------------------
    # Calibration (Platt scaling)
    # ------------------------------------------------------------------

    def calibrate(self, images: list, gt_labels: list):
        """Fit Platt scaling on a small held-out set.

        Parameters
        ----------
        images : list of np.ndarray (BGR)
        gt_labels : list of np.ndarray (N, 4) boxes per image
        """
        all_scores = []
        all_targets = []

        for img, gt_boxes in zip(images, gt_labels):
            results = self._raw_detect(img)
            for score, box in zip(results["scores"], results["boxes_xyxy"]):
                # Check if this detection matches any GT
                is_tp = False
                for gt in gt_boxes:
                    iou = self._box_iou_single(box, gt)
                    if iou > 0.5:
                        is_tp = True
                        break
                all_scores.append(score)
                all_targets.append(1.0 if is_tp else 0.0)

        if len(all_scores) < 10:
            print("⚠️  Not enough detections for calibration")
            return

        # Simple Platt scaling via logistic regression
        from scipy.optimize import minimize

        scores = np.array(all_scores)
        targets = np.array(all_targets)

        def neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * scores + b)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        self._cal_a, self._cal_b = result.x
        print(f"📏 Calibration: a={self._cal_a:.4f}, b={self._cal_b:.4f}")

    def _calibrate_score(self, score: float) -> float:
        if self._cal_a == 1.0 and self._cal_b == 0.0:
            return score
        # Platt scaling is sigmoid(a * logit + b)
        # score is already sigmoid(logit), so we need logit
        eps = 1e-7
        score = np.clip(score, eps, 1.0 - eps)
        logit = np.log(score / (1.0 - score))
        return 1.0 / (1.0 + np.exp(-(self._cal_a * logit + self._cal_b)))

    @staticmethod
    def _box_iou_single(box1, box2):
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[2], box2[2])
        inter_y2 = min(box1[3], box2[3])
        inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-7)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _raw_detect(self, image: np.ndarray) -> dict:
        """Raw detection before NMS (used internally)."""
        if image.shape[2] == 3 and len(image.shape) == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        tensor, scale, pad = self._preprocess(rgb)
        off_x, off_y = pad

        with torch.no_grad():
            cls_preds, reg_preds, lmk_preds = self.model(tensor)

        scores = cls_preds[0].sigmoid().squeeze(-1).cpu().numpy()  # (A,)
        deltas = reg_preds[0].cpu().numpy()  # (A, 4) raw deltas
        anchors = self.anchors.cpu().numpy()  # (A, 4) [cx, cy, w, h]

        # Decode deltas relative to anchors
        from training.train import decode_boxes_np
        boxes_cxcywh = decode_boxes_np(deltas, anchors)

        # Clamp to [0, 1] normalised range
        boxes_cxcywh = np.clip(boxes_cxcywh, 0, 1)

        # Convert from normalised [cx, cy, w, h] to pixel [x1, y1, x2, y2] on canvas
        boxes_xyxy = np.zeros_like(boxes_cxcywh)
        boxes_xyxy[:, 0] = (boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2) * self.img_size
        boxes_xyxy[:, 1] = (boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2) * self.img_size
        boxes_xyxy[:, 2] = (boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2) * self.img_size
        boxes_xyxy[:, 3] = (boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2) * self.img_size

        # Subtract offset and rescale to original image coordinates
        boxes_xyxy[:, [0, 2]] -= off_x
        boxes_xyxy[:, [1, 3]] -= off_y
        boxes_xyxy /= scale

        # Landmarks
        landmarks = None
        if lmk_preds is not None:
            lmks = lmk_preds[0].cpu().numpy()  # (A, 10)
            # Subtract offset and rescale landmarks
            lmks = lmks.reshape(-1, 5, 2)
            lmks[..., 0] = (lmks[..., 0] * self.img_size - off_x) / scale
            lmks[..., 1] = (lmks[..., 1] * self.img_size - off_y) / scale
            landmarks = lmks.reshape(-1, 10)

        return {
            "scores": scores,
            "boxes_xyxy": boxes_xyxy,
            "landmarks": landmarks,
        }

    @torch.no_grad()
    def detect(self, image: np.ndarray, conf_threshold: float = None) -> dict:
        """Detect faces in an image.

        Parameters
        ----------
        image : np.ndarray (BGR, HWC)
        conf_threshold : float, optional
            Override default confidence threshold.

        Returns
        -------
        dict with keys:
            'boxes': (K, 4) [x1, y1, x2, y2] in original image coords
            'scores': (K,) confidence scores
            'landmarks': (K, 5, 2) or None
        """
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        raw = self._raw_detect(image)
        scores = raw["scores"]
        boxes = raw["boxes_xyxy"]

        # Score filter
        mask = scores > thresh
        scores = scores[mask]
        boxes = boxes[mask]
        landmarks = raw["landmarks"][mask] if raw["landmarks"] is not None else None

        if len(scores) == 0:
            return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0),
                    "landmarks": None}

        import torchvision.ops as ops
        
        # Convert to tensors for fast NMS
        t_boxes = torch.from_numpy(boxes).to(self.device)
        t_scores = torch.from_numpy(scores).to(self.device)
        
        # Move to GPU for fast NMS
        keep = ops.nms(t_boxes, t_scores, self.nms_iou)
        
        # Keep top-N
        keep = keep[:self.max_detections]
        final_boxes = boxes[keep.cpu().numpy()]
        final_scores = scores[keep.cpu().numpy()]

        # Apply calibration
        final_scores = np.array([self._calibrate_score(s) for s in final_scores])

        final_landmarks = None
        if landmarks is not None:
            # landmarks is already on device from _raw_detect
            l_on_device = torch.from_numpy(landmarks).to(self.device) if isinstance(landmarks, np.ndarray) else landmarks
            final_landmarks = l_on_device[keep].cpu().numpy().reshape(-1, 5, 2)

        return {
            "boxes": final_boxes,
            "scores": final_scores,
            "landmarks": final_landmarks,
        }

    def detect_and_draw(self, image: np.ndarray) -> np.ndarray:
        """Detect and draw bounding boxes on image."""
        result = self.detect(image)
        img = image.copy()

        for i in range(len(result["scores"])):
            x1, y1, x2, y2 = result["boxes"][i].astype(int)
            score = result["scores"][i]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw landmarks if available
            if result["landmarks"] is not None:
                for lx, ly in result["landmarks"][i]:
                    cv2.circle(img, (int(lx), int(ly)), 2, (255, 0, 0), -1)

        return img
