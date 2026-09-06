"""Personalized dual-branch sEMG/IMU CNN with per-class late fusion."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
import torch.nn.functional as functional


INITIAL_EMG_WEIGHTS = torch.tensor(
    [0.50, 0.70, 0.70, 0.30, 0.30, 0.20, 0.20, 0.80],
    dtype=torch.float32,
)


class ModelOutput(NamedTuple):
    fused_logits: torch.Tensor
    emg_logits: torch.Tensor
    imu_logits: torch.Tensor
    emg_weights: torch.Tensor


class _ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: tuple[int, int, int],
        kernels: tuple[int, int, int],
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_channels
        for output_channels, kernel in zip(channels, kernels, strict=True):
            layers.extend((
                nn.Conv1d(previous, output_channels, kernel, padding=kernel // 2, bias=False),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
            ))
            previous = output_channels
        layers.extend((nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers)
        self.output_features = channels[-1]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DualBranchCNN(nn.Module):
    """Always evaluate both sensors and learn one bounded EMG weight per class."""

    def __init__(self, class_count: int = 8, dropout: float = 0.25):
        super().__init__()
        if type(class_count) is not int or class_count != 8:
            raise ValueError("this architecture is fixed to eight gesture classes")
        self.class_count = class_count
        self.emg_encoder = _ConvEncoder(8, (32, 64, 128), (9, 7, 5), dropout)
        self.imu_encoder = _ConvEncoder(6, (24, 48, 96), (5, 5, 3), dropout)
        self.emg_head = nn.Linear(self.emg_encoder.output_features, class_count)
        self.imu_head = nn.Linear(self.imu_encoder.output_features, class_count)
        priors = INITIAL_EMG_WEIGHTS.clamp(1e-6, 1.0 - 1e-6)
        self.fusion_logits = nn.Parameter(torch.logit(priors))

    def emg_weights(self) -> torch.Tensor:
        return torch.sigmoid(self.fusion_logits)

    def forward(self, emg: torch.Tensor, imu: torch.Tensor) -> ModelOutput:
        if emg.ndim != 3 or tuple(emg.shape[1:]) != (8, 250):
            raise ValueError(f"EMG tensor shape must be [N, 8, 250], got {tuple(emg.shape)}")
        if imu.ndim != 3 or tuple(imu.shape[1:]) != (6, 52):
            raise ValueError(f"IMU tensor shape must be [N, 6, 52], got {tuple(imu.shape)}")
        if emg.shape[0] != imu.shape[0]:
            raise ValueError("EMG and IMU tensors must have the same batch size")

        emg_logits = self.emg_head(self.emg_encoder(emg))
        imu_logits = self.imu_head(self.imu_encoder(imu))
        emg_weights = self.emg_weights()
        fused_logits = emg_weights * emg_logits + (1.0 - emg_weights) * imu_logits
        return ModelOutput(fused_logits, emg_logits, imu_logits, emg_weights)


def fused_training_loss(
    output: ModelOutput,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    auxiliary_weight: float = 0.1,
) -> torch.Tensor:
    """Calculate fused CE plus small branch-head auxiliary CE terms."""
    if output.fused_logits.ndim != 2:
        raise ValueError("fused logits must be a two-dimensional tensor")
    class_count = output.fused_logits.shape[1]
    if output.emg_logits.shape != output.fused_logits.shape or output.imu_logits.shape != output.fused_logits.shape:
        raise ValueError("all model logits must have the same shape")
    if class_weights.ndim != 1 or class_weights.numel() != class_count:
        raise ValueError(f"class weights must have shape [{class_count}]")
    if targets.ndim != 1 or targets.numel() != output.fused_logits.shape[0]:
        raise ValueError("targets must have one value per model output row")
    if not 0 <= auxiliary_weight <= 1:
        raise ValueError("auxiliary weight must be between zero and one")

    fused_loss = functional.cross_entropy(output.fused_logits, targets, weight=class_weights)
    emg_loss = functional.cross_entropy(output.emg_logits, targets, weight=class_weights)
    imu_loss = functional.cross_entropy(output.imu_logits, targets, weight=class_weights)
    return fused_loss + auxiliary_weight * emg_loss + auxiliary_weight * imu_loss
