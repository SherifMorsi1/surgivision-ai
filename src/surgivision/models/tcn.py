"""Multi-stage temporal convolutional networks for surgical phase modelling.

Stage two of the pipeline. A frozen spatial backbone turns each frame into a
feature vector; this module learns the temporal structure over those vectors —
that phases follow a broadly fixed order, run for minutes rather than frames,
and rarely revisit an earlier phase. Because it consumes cached features rather
than pixels, an entire video is a single short sequence and training costs
seconds per epoch, which is what makes the approach tractable without a GPU.

The design follows MS-TCN (Farha and Gall) as adapted to surgical workflow by
TeCNO. Each stage refines the previous stage's predictions, and stacking
dilated convolutions grows the receptive field exponentially with depth, so a
ten-layer stage sees roughly a thousand frames of context.

Setting ``causal=True`` restricts every convolution to past frames only. That
is the honest configuration for intra-operative use, where future frames do not
exist, and it costs a few points of accuracy relative to the offline setting —
a gap worth reporting rather than hiding.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DilatedResidualLayer(nn.Module):
    """A dilated 1-D convolution with a residual connection."""

    def __init__(self, dilation: int, channels: int, dropout: float = 0.5) -> None:
        super().__init__()
        if dilation < 1:
            raise ValueError("Dilation must be at least one")
        self.dilation = dilation
        self.causal = False
        self.conv_dilated = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def _pad(self, x: Tensor) -> Tensor:
        if self.causal:
            # Pad only on the left so no output position sees a future frame.
            return F.pad(x, (2 * self.dilation, 0))
        return F.pad(x, (self.dilation, self.dilation))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        out = F.relu(self.conv_dilated(self._pad(x)))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        out = x + out
        if mask is not None:
            out = out * mask
        return out


class SingleStageTCN(nn.Module):
    """One refinement stage: 1x1 projection, dilated stack, classifier."""

    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        num_features: int,
        num_classes: int,
        dropout: float = 0.5,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("A stage needs at least one layer")
        self.conv_in = nn.Conv1d(in_channels, num_features, kernel_size=1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(2**index, num_features, dropout) for index in range(num_layers)
        )
        for layer in self.layers:
            layer.causal = causal
        self.conv_out = nn.Conv1d(num_features, num_classes, kernel_size=1)

    @property
    def receptive_field(self) -> int:
        """Frames of context visible to a single output position."""

        return 1 + 2 * sum(layer.dilation for layer in self.layers)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out)
        if mask is not None:
            out = out * mask
        return out


class MultiStageTCN(nn.Module):
    """Stacked refinement stages over cached frame features.

    Every stage emits logits, and all of them are returned. Supervising the
    intermediate stages as well as the last is what stabilizes training, so the
    loss is computed over the full stack rather than the final output alone.
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_features: int = 64,
        in_channels: int = 2048,
        num_classes: int = 7,
        dropout: float = 0.5,
        causal: bool = False,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError("At least one stage is required")
        self.causal = causal
        self.num_classes = num_classes
        self.stage1 = SingleStageTCN(
            num_layers, in_channels, num_features, num_classes, dropout, causal
        )
        self.refinement = nn.ModuleList(
            SingleStageTCN(num_layers, num_classes, num_features, num_classes, dropout, causal)
            for _ in range(num_stages - 1)
        )

    @property
    def receptive_field(self) -> int:
        stages = [self.stage1, *self.refinement]
        return 1 + sum(stage.receptive_field - 1 for stage in stages)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Map ``(batch, in_channels, time)`` to ``(stages, batch, classes, time)``."""

        if x.dim() != 3:
            raise ValueError("Expected a (batch, channels, time) tensor")
        outputs = [self.stage1(x, mask)]
        for stage in self.refinement:
            outputs.append(stage(F.softmax(outputs[-1], dim=1), mask))
        return torch.stack(outputs)
