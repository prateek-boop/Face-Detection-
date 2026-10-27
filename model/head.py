"""
Detection head — classification, bounding-box regression, and landmark branches.

Applied independently to each FPN level (P3, P4, P5) with shared weights.
Supports both anchored and anchor-free decoding.
"""

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Multi-task detection head.

    Architecture per FPN level:
        Shared tower  (4 × Conv3×3-BN-ReLU)
            ├─ Classification branch  → (B, num_anchors × num_classes, H, W)
            ├─ Regression branch      → (B, num_anchors × 4, H, W)
            └─ Landmark branch        → (B, num_anchors × 10, H, W)  [optional]

    Parameters
    ----------
    in_channels : int
        Input channels from FPN (e.g. 256).
    num_anchors : int
        Anchors per spatial position (e.g. 9 for 3 scales × 3 ratios).
    num_classes : int
        Number of object classes (1 for face detection).
    num_landmarks : int
        Number of landmark points (5 for face: eyes, nose, mouth corners).
        Set to 0 to disable.
    tower_depth : int
        Number of 3×3 convs in the shared tower.
    """

    def __init__(self, in_channels: int = 256, num_anchors: int = 9,
                 num_classes: int = 1, num_landmarks: int = 5,
                 tower_depth: int = 4):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_landmarks = num_landmarks

        # Shared conv tower
        tower = []
        for _ in range(tower_depth):
            tower.extend([
                nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ])
        self.tower = nn.Sequential(*tower)

        # Classification branch
        self.cls_head = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, padding=1
        )

        # Bounding-box regression branch [dx, dy, dw, dh]
        self.reg_head = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, padding=1
        )

        # Landmark branch: 5 points × (x, y) = 10 outputs per anchor
        if num_landmarks > 0:
            self.lmk_head = nn.Conv2d(
                in_channels, num_anchors * num_landmarks * 2,
                kernel_size=3, padding=1
            )
        else:
            self.lmk_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.cls_head or m is self.reg_head:
                    nn.init.normal_(m.weight, std=0.01)
                else:
                    nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                            nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Initialise classification bias for focal loss stability
        # prior=0.01 makes the model start by predicting 'background' with 99% confidence
        import math
        prior = 0.01
        bias_val = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_head.bias, bias_val)

    def forward(self, fpn_features):
        """
        Parameters
        ----------
        fpn_features : tuple of tensors (P3, P4, P5)

        Returns
        -------
        cls_preds : (B, total_anchors, num_classes)
        reg_preds : (B, total_anchors, 4)
        lmk_preds : (B, total_anchors, num_landmarks * 2) or None
        """
        all_cls = []
        all_reg = []
        all_lmk = []

        for feat in fpn_features:
            B = feat.shape[0]
            t = self.tower(feat)

            # Classification
            cls = self.cls_head(t)                          # (B, A*C, H, W)
            cls = cls.permute(0, 2, 3, 1).contiguous()     # (B, H, W, A*C)
            cls = cls.view(B, -1, self.num_classes)         # (B, H*W*A, C)
            all_cls.append(cls)

            # Regression
            reg = self.reg_head(t)
            reg = reg.permute(0, 2, 3, 1).contiguous()
            reg = reg.view(B, -1, 4)
            all_reg.append(reg)

            # Landmarks
            if self.lmk_head is not None:
                lmk = self.lmk_head(t)
                lmk = lmk.permute(0, 2, 3, 1).contiguous()
                lmk = lmk.view(B, -1, self.num_landmarks * 2)
                all_lmk.append(lmk)

        cls_preds = torch.cat(all_cls, dim=1)   # (B, total_anchors, C)
        reg_preds = torch.cat(all_reg, dim=1)   # Raw deltas [dx, dy, dw, dh]

        if self.lmk_head is not None:
            lmk_preds = torch.cat(all_lmk, dim=1)
        else:
            lmk_preds = None

        return cls_preds, reg_preds, lmk_preds
