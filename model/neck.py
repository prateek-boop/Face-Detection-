"""
Feature Pyramid Network (FPN) neck.

Fuses multi-scale backbone features (C3, C4, C5) via lateral connections
and a top-down pathway so the model handles faces from ~10px to ~500px.
This is the single biggest upgrade for multi-scale face detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Feature Pyramid Network.

    Takes C3, C4, C5 from the backbone (at strides 8, 16, 32) and produces
    P3, P4, P5 — each with ``out_channels`` channels.

    Architecture:
        C5 → 1×1 lateral → P5
              ↓ upsample + add
        C4 → 1×1 lateral → P4
              ↓ upsample + add
        C3 → 1×1 lateral → P3

        Each P level gets a 3×3 smoothing conv to reduce aliasing.

    Parameters
    ----------
    in_channels_list : list of int
        Channel dims from the backbone (e.g. [32, 64, 160]).
    out_channels : int
        Unified channel dim for all pyramid levels (default 256).
    """

    def __init__(self, in_channels_list: list, out_channels: int = 256):
        super().__init__()
        assert len(in_channels_list) == 3, "FPN expects exactly 3 input levels"

        # Lateral 1×1 convs to reduce each Ci to out_channels
        self.lateral_c3 = nn.Conv2d(in_channels_list[0], out_channels, 1)
        self.lateral_c4 = nn.Conv2d(in_channels_list[1], out_channels, 1)
        self.lateral_c5 = nn.Conv2d(in_channels_list[2], out_channels, 1)

        # Smoothing 3×3 convs on merged features
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        """
        Parameters
        ----------
        features : tuple of (C3, C4, C5) tensors

        Returns
        -------
        (P3, P4, P5) : tuple of tensors, each with out_channels channels
        """
        c3, c4, c5 = features

        # Top level
        p5 = self.lateral_c5(c5)

        # Upsample P5 and add to lateral C4
        p4 = self.lateral_c4(c4) + F.interpolate(
            p5, size=c4.shape[2:], mode="nearest"
        )

        # Upsample P4 and add to lateral C3
        p3 = self.lateral_c3(c3) + F.interpolate(
            p4, size=c3.shape[2:], mode="nearest"
        )

        # Smooth
        p3 = self.smooth_p3(p3)
        p4 = self.smooth_p4(p4)
        p5 = self.smooth_p5(p5)

        return p3, p4, p5
