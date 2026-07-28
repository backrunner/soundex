"""Multi-scale discriminator for GAN training."""

import torch
import torch.nn as nn


class DiscriminatorBlock(nn.Module):
    """Single discriminator scale."""

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=(1, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=(1, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=(1, 2), padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator operating at different frequency resolutions."""

    def __init__(self, scales: list[int] | None = None, in_channels: int = 1) -> None:
        super().__init__()
        if scales is None:
            scales = [1, 2, 4]
        self.discriminators = nn.ModuleList([DiscriminatorBlock(in_channels) for _ in scales])
        self.scales = scales
        self.downsample = nn.AvgPool2d((1, 2))

    def forward(self, x: torch.Tensor) -> list[list[torch.Tensor]]:
        """Returns features from each scale."""
        all_features = []
        current = x
        for disc in self.discriminators:
            features = disc(current)
            all_features.append(features)
            current = self.downsample(current)
        return all_features
