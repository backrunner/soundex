"""SoundEx Generator: Lightweight dual-stream U-Net for spectral bandwidth extension.

Architecture fuses:
- AERO: Spectral-domain U-Net encoder-decoder
- AP-BWE: Dual-stream amplitude + phase prediction
- UL-UNAS: Depthwise separable convolutions + inverted residuals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.spectral_refiner import SpectralRefiner


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block (from UL-UNAS)."""

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            (1, kernel_size),
            (1, stride),
            (0, padding),
            groups=channels,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(channels, channels, 1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""

    def __init__(self, channels: int, expand_ratio: int = 4) -> None:
        super().__init__()
        hidden = channels * expand_ratio
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 1), groups=hidden),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Encoder: DWS conv + frequency-axis downsample."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, 3), stride=(1, 2), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            DepthwiseSeparableConv(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DecoderBlock(nn.Module):
    """Decoder: upsample + DWS conv + skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_ch,
            in_ch,
            kernel_size=(1, 4),
            stride=(1, 2),
            padding=(0, 1),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, (1, 3), padding=(0, 1)),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            DepthwiseSeparableConv(out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = match_frequency_size(x, skip)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


def match_frequency_size(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Crop or right-pad only the frequency axis to match ``reference``."""
    if x.shape[-2] != reference.shape[-2]:
        raise ValueError(
            "Frequency-only generator cannot change the time dimension: "
            f"got {x.shape[-2]} and {reference.shape[-2]}"
        )
    difference = reference.shape[-1] - x.shape[-1]
    if difference > 0:
        return F.pad(x, (0, difference))
    if difference < 0:
        return x[..., : reference.shape[-1]]
    return x


class SpectralStream(nn.Module):
    """Single stream for amplitude or phase prediction."""

    def __init__(
        self, channels: list[int], bottleneck_blocks: int = 2, expand_ratio: int = 4
    ) -> None:
        super().__init__()
        # Encoder
        self.encoders = nn.ModuleList()
        in_ch = 1  # Single channel input (log-mag or phase)
        self.encoder_channels = []
        for out_ch in channels:
            self.encoders.append(EncoderBlock(in_ch, out_ch))
            self.encoder_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[InvertedResidual(channels[-1], expand_ratio) for _ in range(bottleneck_blocks)]
        )

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            self.decoders.append(DecoderBlock(channels[i], channels[i - 1], channels[i - 1]))

        # Output
        self.output_conv = nn.ConvTranspose2d(
            channels[0],
            1,
            kernel_size=(1, 4),
            stride=(1, 2),
            padding=(0, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode with skip connections
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decode with skip connections
        for i, decoder in enumerate(self.decoders):
            skip_idx = len(skips) - 2 - i
            x = decoder(x, skips[skip_idx])

        # Final upsample to original resolution
        x = self.output_conv(x)
        return x


class SoundExGenerator(nn.Module):
    """Dual-stream U-Net generator for audio bandwidth extension.

    Input: full degraded log-magnitude + phase [B, 2, T, F]
    Output: candidate clean log-magnitude + phase [B, 2, T, F]
    """

    def __init__(
        self,
        channels: list[int] | None = None,
        bottleneck_blocks: int = 2,
        expand_ratio: int = 4,
        cross_stream_interactions: bool = False,
        circular_phase_features: bool = False,
        spectral_refiner: dict | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(cross_stream_interactions, bool):
            raise ValueError("cross_stream_interactions must be a boolean")
        if not isinstance(circular_phase_features, bool):
            raise ValueError("circular_phase_features must be a boolean")
        self.circular_phase_features = circular_phase_features
        if channels is None:
            channels = [24, 48, 96, 96]

        # Amplitude stream
        self.amp_stream = SpectralStream(channels, bottleneck_blocks, expand_ratio)
        # Phase stream
        self.phase_stream = SpectralStream(channels, bottleneck_blocks, expand_ratio)

        # Expose the unit phasor without removing the legacy raw-phase feature.
        # Added channels start at zero influence and preserve the control RNG.
        if circular_phase_features:
            original = self.phase_stream.encoders[0].conv[0]
            with torch.random.fork_rng(devices=[]):
                expanded = nn.Conv2d(
                    3,
                    original.out_channels,
                    original.kernel_size,
                    stride=original.stride,
                    padding=original.padding,
                )
            with torch.no_grad():
                expanded.weight.zero_()
                expanded.weight[:, :1].copy_(original.weight)
                expanded.bias.copy_(original.bias)
            self.phase_stream.encoders[0].conv[0] = expanded

        # Optional simultaneous amplitude/phase interactions at encoder scales.
        # Zero initialization preserves a migrated parent's exact predictions.
        self.interactions = nn.ModuleList()
        if cross_stream_interactions:
            # These projections start at zero; their discarded random initialization
            # must not alter crop/sampler RNG relative to the legacy control model.
            with torch.random.fork_rng(devices=[]):
                for width in channels:
                    pair = nn.ModuleList([nn.Conv2d(width, width, 1, bias=False) for _ in range(2)])
                    for projection in pair:
                        nn.init.zeros_(projection.weight)
                    self.interactions.append(pair)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 16, (1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(16, 2, (1, 3), padding=(0, 1)),
        )
        # Preserve legacy initialization and the crop/sampling RNG across arms.
        self.spectral_refiner = None
        if spectral_refiner is not None:
            with torch.random.fork_rng(devices=[]):
                self.spectral_refiner = SpectralRefiner(**spectral_refiner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Full degraded features [B, 2, T, F] (log-mag + phase)

        Returns:
            Candidate clean features [B, 2, T, F] (log-mag + phase)
        """
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected input shape [B, 2, T, F], got {tuple(x.shape)}")
        reference = x

        # Split input into amplitude and phase channels
        amp_in = x[:, 0:1, :, :]
        phase_in = x[:, 1:2, :, :]
        if self.circular_phase_features:
            phase_in = torch.cat((phase_in, torch.sin(phase_in), torch.cos(phase_in)), dim=1)

        # Dual-stream prediction
        if self.interactions:
            amp_out, phase_out = self._interacting_streams(amp_in, phase_in)
        else:
            amp_out = self.amp_stream(amp_in)
            phase_out = self.phase_stream(phase_in)

        # The streams are structurally identical and must remain shape-compatible.
        if amp_out.shape != phase_out.shape:
            raise RuntimeError(
                "Amplitude and phase streams produced different shapes: "
                f"{tuple(amp_out.shape)} and {tuple(phase_out.shape)}"
            )

        # Fusion
        combined = torch.cat([amp_out, phase_out], dim=1)  # [B, 2, T, F]
        residual = self.fusion(combined)
        output = combined + residual

        output = match_frequency_size(output, reference)
        if self.spectral_refiner is not None:
            output = output + self.spectral_refiner(reference)
        return output

    def _interacting_streams(
        self, amplitude: torch.Tensor, phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude_skips, phase_skips = [], []
        for amp_encoder, phase_encoder, pair in zip(
            self.amp_stream.encoders, self.phase_stream.encoders, self.interactions, strict=True
        ):
            amp_encoded = amp_encoder(amplitude)
            phase_encoded = phase_encoder(phase)
            amplitude = amp_encoded + pair[0](phase_encoded)
            phase = phase_encoded + pair[1](amp_encoded)
            amplitude_skips.append(amplitude)
            phase_skips.append(phase)
        amplitude = self.amp_stream.bottleneck(amplitude)
        phase = self.phase_stream.bottleneck(phase)
        for index, (amp_decoder, phase_decoder) in enumerate(
            zip(self.amp_stream.decoders, self.phase_stream.decoders, strict=True)
        ):
            skip_index = len(amplitude_skips) - 2 - index
            amplitude = amp_decoder(amplitude, amplitude_skips[skip_index])
            phase = phase_decoder(phase, phase_skips[skip_index])
        return self.amp_stream.output_conv(amplitude), self.phase_stream.output_conv(phase)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick parameter count verification
    model = SoundExGenerator()
    print(f"Generator parameters: {model.count_parameters():,}")
    # Test forward pass
    dummy = torch.randn(2, 2, 1, 513)
    out = model(dummy)
    print(f"Input shape: {dummy.shape} → Output shape: {out.shape}")
