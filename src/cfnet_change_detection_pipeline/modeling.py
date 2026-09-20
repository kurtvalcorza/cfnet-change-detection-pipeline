"""CFNet — Content Focuser Network for bi-temporal remote-sensing change detection — in plain PyTorch.

Vendored from the authors' training code (`wifiBlack/CFNet`, `model/{CFNet,encoder,content_decoder,change_decoder,
focuser,utils}.py` at commit `54acadab23b9d9395ec6814386c2d4a1253eac5a`, Apache-2.0), rewritten as one module with
no training-time scaffolding (no argument parsing, no `autocast` inside `forward`, no feature-map dumps). The only
external component is the encoder backbone, the stem and first four stages of torchvision's EfficientNet-B5
(`torchvision.models.efficientnet_b5(weights=None).features[0:5]`), taken from the installed `torchvision` package
at a pinned version and never downloaded — the fine-tuned checkpoint carries its weights. Parameter and buffer
names reproduce the upstream state dict exactly (`encoder._backbone.*`, `content_decoder_1.*`,
`content_decoder_2.*`, `change_decoder.*`), which is how the conversion can load it with `strict=True`.

Architecture (Wu et al., 2025):

* `Encoder` — the shared EfficientNet-B5 stem + stages 1–4 applied to both dates; the outputs of stages 1–4
  (24 / 40 / 64 / 128 channels at 1/2, 1/4, 1/8 and 1/16 of the input) are kept per date.
* `ContentDecoder` (one per date) — a top-down path of three `Aggregation` blocks (transposed-convolution
  upsampling, 1 × 1 fusion, two residual blocks) producing four content maps from coarse to fine (128 → 64 → 40 →
  24 channels). Its CBAM attention modules are constructed and carry weights in the checkpoint but, as upstream,
  their outputs are never used (`forward` discards them), so this port keeps the modules and does not apply them.
* `Focuser` — per scale, the cosine distance between the two dates' content maps through `tanh` gives a
  change-focus map in [0, 1) that reweights the fused features.
* `ChangeDecoder` — per scale, a 3-D fusion convolution over the two dates, weighted by the focus map and
  aggregated top-down as in the content decoder; a final stride-2 transposed convolution returns a one-channel map
  at the input resolution through `tanh`. A fifth fusion block for the 3-channel input level exists in the checkpoint
  and, as upstream, is never called.

The network takes two standardised (B, 3, H, W) images — H and W multiples of 32 — and returns the change map
(B, H, W) in (−1, 1); upstream declares a pixel changed where the map exceeds 0.5.
"""

from __future__ import annotations

import torch
from torch import nn

CHANNELS: tuple[int, ...] = (3, 24, 40, 64, 128)  # input, then the four kept EfficientNet-B5 stages
BACKBONE_STAGES = 5  # features[0] (stem) + stages 1..4


def _efficientnet_b5_features() -> nn.ModuleList:
    """The stem and first four stages of torchvision's EfficientNet-B5, randomly initialised (no download)."""
    from torchvision.models import efficientnet_b5

    features = efficientnet_b5(weights=None).features
    return nn.ModuleList(list(features.children())[:BACKBONE_STAGES])


class Encoder(nn.Module):
    """Shared backbone applied to both dates; returns the four stage outputs per date (fine to coarse)."""

    def __init__(self) -> None:
        super().__init__()
        self._backbone = _efficientnet_b5_features()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        y1: list[torch.Tensor] = []
        y2: list[torch.Tensor] = []
        for index, layer in enumerate(self._backbone):
            x1 = layer(x1)
            x2 = layer(x2)
            if index != 0:
                y1.append(x1)
                y2.append(x2)
        return y1, y2


class CBA1x1(nn.Module):
    """1 × 1 convolution → BatchNorm → ReLU (upstream `CBA1x1`)."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=1), nn.BatchNorm2d(out_channel), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BasicBlock(nn.Module):
    """ResNet-style residual block (upstream `BasicBlock`, no downsampling)."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + x)


class Aggregation(nn.Module):
    """Upsample the coarse map by a stride-2 transposed convolution, concatenate the finer map, fuse 1 × 1, two
    residual blocks (upstream `Aggregation`)."""

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channel, in_channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
        )
        self.conv1 = CBA1x1(in_channel + out_channel, out_channel)
        self.residual1 = BasicBlock(out_channel, out_channel)
        self.residual2 = BasicBlock(out_channel, out_channel)

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor) -> torch.Tensor:
        y = self.conv1(torch.cat([self.upsample(coarse), fine], dim=1))
        return self.residual2(self.residual1(y))


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """Convolutional block attention (upstream `CBAM`); present in the checkpoint, unused by the forward pass."""

    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.channel_attention(x) * x
        return self.spatial_attention(x_out) * x_out


class ContentDecoder(nn.Module):
    """Top-down aggregation of one date's four stage outputs into four content maps, coarse to fine."""

    def __init__(self, channel_list: tuple[int, ...] = CHANNELS) -> None:
        super().__init__()
        coarse_to_fine = tuple(reversed(channel_list))  # 128, 64, 40, 24, 3
        targets = tuple(reversed(channel_list[1:-1]))  # 64, 40, 24
        self.aggregations = nn.ModuleList([Aggregation(i, o) for i, o in zip(coarse_to_fine, targets, strict=False)])
        self.cbam = nn.ModuleList([CBAM(c) for c in reversed(channel_list[1:])])  # weights present, never applied

    def forward(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        maps = [features[-1]]
        for index, fine in enumerate(reversed(features[:-1])):
            maps.append(self.aggregations[index](maps[index], fine))
        # Upstream applies `self.cbam[idx]` to each map here and discards the result; the maps are returned as they are.
        return maps


class Focuser(nn.Module):
    """Per-scale change-focus maps: tanh of the cosine distance between the two dates' content maps."""

    def __init__(self) -> None:
        super().__init__()
        self.cos_sim = nn.CosineSimilarity(dim=1)

    def forward(self, maps_1: list[torch.Tensor], maps_2: list[torch.Tensor]) -> list[torch.Tensor]:
        return [torch.tanh(1.0 - self.cos_sim(a, b)) for a, b in zip(maps_1, maps_2, strict=True)]


class FuseConv3d(nn.Module):
    """Fuse the two dates' maps at one scale with a (2, 3, 3) 3-D convolution → BatchNorm3d → ReLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.BatchNorm3d(channels),
            nn.ReLU(),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2)
        return self.conv(x).squeeze(2)


class ChangeDecoder(nn.Module):
    """Fuse both dates per scale, weight by the focus maps, aggregate top-down, and decode one change map."""

    def __init__(self, channel_list: tuple[int, ...] = CHANNELS) -> None:
        super().__init__()
        coarse_to_fine = tuple(reversed(channel_list))  # 128, 64, 40, 24, 3 — the last (3) is never called
        targets = tuple(reversed(channel_list[1:-1]))
        self.fuseconv3ds = nn.ModuleList([FuseConv3d(c) for c in coarse_to_fine])
        self.aggregations = nn.ModuleList([Aggregation(i, o) for i, o in zip(coarse_to_fine, targets, strict=False)])
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(channel_list[1], 1, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        )

    def forward(self, maps_1: list[torch.Tensor], maps_2: list[torch.Tensor], focuses: list[torch.Tensor]) -> torch.Tensor:
        fused: list[torch.Tensor] = []
        for index, (a, b, focus) in enumerate(zip(maps_1, maps_2, focuses, strict=True)):
            y = self.fuseconv3ds[index](a, b) * focus.unsqueeze(1)
            if index > 0:
                y = self.aggregations[index - 1](fused[index - 1], y)
            fused.append(y)
        return torch.tanh(self.upconv(fused[-1]).squeeze(1))


class CFNet(nn.Module):
    """Two standardised (B, 3, H, W) dates → (B, H, W) change map in (−1, 1)."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.content_decoder_1 = ContentDecoder()
        self.content_decoder_2 = ContentDecoder()
        self.change_decoder = ChangeDecoder()
        self.focuser = Focuser()

    def content(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """The two dates' content maps (four scales each, coarse to fine)."""
        y1, y2 = self.encoder(x1, x2)
        return self.content_decoder_1(y1), self.content_decoder_2(y2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if tuple(x1.shape) != tuple(x2.shape):
            raise ValueError(f"the two dates must have the same shape, got {tuple(x1.shape)} and {tuple(x2.shape)}")
        if x1.shape[-1] % 32 or x1.shape[-2] % 32:
            raise ValueError(f"height and width must be multiples of 32, got {tuple(x1.shape[-2:])}")
        maps_1, maps_2 = self.content(x1, x2)
        focuses = self.focuser(maps_1, maps_2)
        return self.change_decoder(maps_1, maps_2, focuses)

    def forward_with_content(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Change map plus the content maps and focus maps — what the upstream loss consumes."""
        maps_1, maps_2 = self.content(x1, x2)
        focuses = self.focuser(maps_1, maps_2)
        return self.change_decoder(maps_1, maps_2, focuses), maps_1, maps_2, focuses


def check_shapes(module: nn.Module) -> dict[str, int]:
    """Sanity numbers used by the tests and the conversion: tensors, elements and parameters."""
    state = module.state_dict()
    return {
        "tensors": len(state),
        "elements": sum(int(v.numel()) for v in state.values()),
        "parameters": sum(p.numel() for p in module.parameters()),
    }
