"""
MobileNetV2-style backbone with depthwise-separable convolutions.

Outputs feature maps at 3 strides (8×, 16×, 32×) for FPN consumption.
Uses inverted residual blocks with expansion factors for efficiency —
~8-9× fewer FLOPs than standard convolutions.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBNMish(nn.Module):
    """Conv2d → BatchNorm → Mish."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        padding = (kernel - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding,
                      groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.Mish() if hasattr(nn, 'Mish') else Mish(),
        )

    def forward(self, x):
        return self.block(x)


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block.

    1×1 expand  →  3×3 depthwise  →  1×1 project (linear)
    with residual connection when stride=1 and in_ch == out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: float = 6.0):
        super().__init__()
        self.use_residual = (stride == 1 and in_ch == out_ch)
        mid_ch = int(in_ch * expand_ratio)

        layers = []
        # Expand (skip if ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBNMish(in_ch, mid_ch, kernel=1))
        # Depthwise
        layers.append(ConvBNMish(mid_ch, mid_ch, kernel=3, stride=stride,
                                 groups=mid_ch))
        # Project (linear — no activation)
        layers.extend([
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class MobileBackbone(nn.Module):
    """Lightweight backbone producing C3, C4, C5 feature maps.

    Architecture (MobileNetV2-like):
        Stem  → stride 2
        Stage 1 (t=1, c=16,  n=1, s=1)   →  stride 2
        Stage 2 (t=6, c=24,  n=2, s=2)   →  stride 4
        Stage 3 (t=6, c=32,  n=3, s=2)   →  stride 8    → C3
        Stage 4 (t=6, c=64,  n=4, s=2)   →  stride 16   → C4
        Stage 5 (t=6, c=96,  n=3, s=1)   →  stride 16
        Stage 6 (t=6, c=160, n=3, s=2)   →  stride 32   → C5
        Stage 7 (t=6, c=320, n=1, s=1)   →  stride 32

    Parameters
    ----------
    width_mult : float
        Channel width multiplier (default 1.0).
    """

    # (expand_ratio, out_channels, num_blocks, stride)
    STAGE_SETTINGS = [
        (1,  16,  1, 1),   # stage 1
        (6,  24,  2, 2),   # stage 2
        (6,  32,  3, 2),   # stage 3  → C3 output
        (6,  64,  4, 2),   # stage 4  → C4 output
        (6,  96,  3, 1),   # stage 5
        (6, 160,  3, 2),   # stage 6  → C5 output
        (6, 320,  1, 1),   # stage 7
    ]

    # Which stages produce the C3, C4, C5 outputs (0-indexed)
    OUTPUT_STAGES = {2, 3, 5}

    def __init__(self, width_mult: float = 1.0):
        super().__init__()

        def _ch(c):
            return max(8, int(c * width_mult + 0.5) // 8 * 8)

        # Stem: 3 → 32, stride 2
        stem_ch = _ch(32)
        self.stem = ConvBNMish(3, stem_ch, kernel=3, stride=2)

        # Build stages
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        self._out_channels = []

        for stage_idx, (t, c, n, s) in enumerate(self.STAGE_SETTINGS):
            out_ch = _ch(c)
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, out_ch, stride=stride,
                                               expand_ratio=t))
                in_ch = out_ch
            self.stages.append(nn.Sequential(*blocks))

            if stage_idx in self.OUTPUT_STAGES:
                self._out_channels.append(out_ch)

        self._init_weights()

    @property
    def out_channels(self) -> list:
        """Channel dims for C3, C4, C5 outputs."""
        return list(self._out_channels)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Returns (C3, C4, C5) feature maps."""
        x = self.stem(x)

        outputs = []
        output_idx = 0
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx in self.OUTPUT_STAGES:
                outputs.append(x)
                output_idx += 1

        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        return tuple(outputs)
