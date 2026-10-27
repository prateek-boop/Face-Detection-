"""
Evaluation metrics for face detection.

- mAP@IoU=0.5 — the standard face detection metric
- Per-difficulty precision-recall curves (WIDER FACE easy/medium/hard)
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------

def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of boxes.

    Parameters
    ----------
    boxes_a : (M, 4) [cx, cy, w, h]
    boxes_b : (N, 4) [cx, cy, w, h]

    Returns
    -------
    iou : (M, N)
    """
    # Convert to xyxy
    a = np.stack([
        boxes_a[:, 0] - boxes_a[:, 2] / 2,
        boxes_a[:, 1] - boxes_a[:, 3] / 2,
        boxes_a[:, 0] + boxes_a[:, 2] / 2,
        boxes_a[:, 1] + boxes_a[:, 3] / 2,
    ], axis=1)  # (M, 4)

    b = np.stack([
        boxes_b[:, 0] - boxes_b[:, 2] / 2,
        boxes_b[:, 1] - boxes_b[:, 3] / 2,
        boxes_b[:, 0] + boxes_b[:, 2] / 2,
        boxes_b[:, 1] + boxes_b[:, 3] / 2,
    ], axis=1)  # (N, 4)

    inter_x1 = np.maximum(a[:, 0:1], b[:, 0:1].T)  # (M, N)
    inter_y1 = np.maximum(a[:, 1:2], b[:, 1:2].T)
    inter_x2 = np.minimum(a[:, 2:3], b[:, 2:3].T)
    inter_y2 = np.minimum(a[:, 3:4], b[:, 3:4].T)

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return inter / (union + 1e-7)


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute Average Precision using all-point interpolation.

    Parameters
    ----------
    recall : (N,) sorted ascending
    precision : (N,)

    Returns
    -------
    ap : float
    """
    # Prepend sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Make precision monotonically decreasing (right to left)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find points where recall changes
    change_idx = np.where(mrec[1:] != mrec[:-1])[0]

    # Area under curve
    ap = np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1])
    return float(ap)


# ---------------------------------------------------------------------------
# mAP computation
# ---------------------------------------------------------------------------

def compute_map(predictions: list, ground_truths: list,
                iou_threshold: float = 0.5,
                eval_difficulty: int = 2) -> dict:
    """Compute mAP@IoU for a set of images.

    Parameters
    ----------
    predictions : list of dicts
        Each: {"boxes": (K, 4), "scores": (K,)}  — boxes in [cx, cy, w, h]
    ground_truths : list of dicts
        Each: {"boxes": (M, 4), "difficulties": (M,)}
    iou_threshold : float
    eval_difficulty : int
        0=Easy, 1=Medium, 2=Hard. GTs harder than this are ignored (no TP, no FP).

    Returns
    -------
    result : dict with keys 'mAP', 'AP', 'precision', 'recall'
    """
    all_scores = []
    all_tp = []
    total_gt = 0

    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        gt_boxes = gt["boxes"]
        gt_diffs = gt.get("difficulties", np.zeros(len(gt_boxes), dtype=np.int32))

        # GTs harder than eval_difficulty are ignored
        if len(gt_diffs) > 0:
            is_difficult = gt_diffs > eval_difficulty
        else:
            is_difficult = np.zeros(len(gt_boxes), dtype=bool)

        # Only count valid GTs towards total recall denominator
        total_gt += int((~is_difficult).sum())

        if len(pred_boxes) == 0:
            continue

        # Sort by score descending
        order = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        matched = np.zeros(len(gt_boxes), dtype=bool)

        iou_mat = None
        if len(gt_boxes) > 0:
            iou_mat = _iou_matrix(pred_boxes, gt_boxes)

        for j in range(len(pred_boxes)):
            if len(gt_boxes) == 0:
                all_scores.append(pred_scores[j])
                all_tp.append(0)
                continue

            ious = iou_mat[j]
            best_gt = np.argmax(ious)

            if ious[best_gt] >= iou_threshold and not matched[best_gt]:
                matched[best_gt] = True
                if is_difficult[best_gt]:
                    # Matched an ignored GT — don't count as TP or FP
                    continue
                all_scores.append(pred_scores[j])
                all_tp.append(1)
            else:
                # If the max IoU was with an ignored GT that was already matched, 
                # or if it was below threshold, it's an FP. 
                # Let's check if it overlaps heavily with ANY ignored GT
                ignore_overlap = ious[is_difficult]
                if len(ignore_overlap) > 0 and np.max(ignore_overlap) >= iou_threshold:
                    continue # It overlaps with an ignored GT, don't penalize
                    
                all_scores.append(pred_scores[j])
                all_tp.append(0)

    if total_gt == 0:
        return {"mAP": 0.0, "AP": 0.0, "precision": np.array([]),
                "recall": np.array([])}

    all_scores = np.array(all_scores)
    all_tp = np.array(all_tp)

    # Sort all detections by score
    order = np.argsort(-all_scores)
    all_tp = all_tp[order]

    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(1 - all_tp)

    recall = tp_cumsum / total_gt
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)

    ap = compute_ap(recall, precision)

    return {"mAP": ap, "AP": ap, "precision": precision, "recall": recall}


# ---------------------------------------------------------------------------
# Per-difficulty evaluation (WIDER FACE easy/medium/hard)
# ---------------------------------------------------------------------------

def evaluate_wider_face(predictions: list, ground_truths: list,
                        iou_threshold: float = 0.5) -> dict:
    """Evaluate mAP broken down by WIDER FACE difficulty."""
    results = {}
    
    results["easy"] = compute_map(predictions, ground_truths, iou_threshold, eval_difficulty=0)
    results["medium"] = compute_map(predictions, ground_truths, iou_threshold, eval_difficulty=1)
    results["hard"] = compute_map(predictions, ground_truths, iou_threshold, eval_difficulty=2)
    results["overall"] = results["hard"]

    return results


# ---------------------------------------------------------------------------
# Precision-Recall curve storage
# ---------------------------------------------------------------------------

class PrecisionRecallCurve:
    """Stores and optionally plots precision-recall curves."""

    def __init__(self):
        self.curves = {}

    def add(self, name: str, precision: np.ndarray, recall: np.ndarray,
            ap: float):
        self.curves[name] = {
            "precision": precision,
            "recall": recall,
            "ap": ap,
        }

    def summary(self) -> str:
        lines = ["Precision-Recall Summary:"]
        for name, data in self.curves.items():
            lines.append(f"  {name}: AP = {data['ap']:.4f}")
        return "\n".join(lines)

    def plot(self, save_path: str = None):
        """Plot all curves. Requires matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping PR plot")
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        for name, data in self.curves.items():
            ax.plot(data["recall"], data["precision"],
                    label=f"{name} (AP={data['ap']:.4f})")

        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"📊 PR curve saved to {save_path}")
        else:
            plt.show()

        plt.close(fig)
