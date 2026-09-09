"""Controlled model variants used for modality and fusion ablation."""

from __future__ import annotations

import torch
import torch.nn.functional as functional

from .model import DualBranchCNN, INITIAL_EMG_WEIGHTS, ModelOutput, fused_training_loss


TRAINING_VARIANTS = (
    "emg_only",
    "imu_only",
    "fixed_50_50",
    "learned_neutral",
    "learned_physics",
)


def build_model_for_variant(variant: str, *, class_count: int = 8) -> DualBranchCNN:
    """Construct one controlled ablation variant with unused parameters frozen."""
    if variant not in TRAINING_VARIANTS:
        raise ValueError(f"unsupported training variant: {variant}")
    neutral = torch.full((class_count,), 0.5, dtype=torch.float32)
    if variant == "learned_physics":
        model = DualBranchCNN(
            class_count=class_count,
            initial_emg_weights=INITIAL_EMG_WEIGHTS,
            trainable_fusion=True,
        )
    elif variant == "learned_neutral":
        model = DualBranchCNN(
            class_count=class_count,
            initial_emg_weights=neutral,
            trainable_fusion=True,
        )
    else:
        model = DualBranchCNN(
            class_count=class_count,
            initial_emg_weights=neutral,
            trainable_fusion=False,
        )

    if variant == "emg_only":
        for module in (model.imu_encoder, model.imu_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    elif variant == "imu_only":
        for module in (model.emg_encoder, model.emg_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return model


def primary_head_for_variant(variant: str) -> str:
    if variant == "emg_only":
        return "emg_only"
    if variant == "imu_only":
        return "imu_only"
    if variant == "fixed_50_50":
        return "fixed_50_50"
    if variant in ("learned_neutral", "learned_physics"):
        return "learned_fusion"
    raise ValueError(f"unsupported training variant: {variant}")


def prediction_logits(output: ModelOutput, head: str) -> torch.Tensor:
    """Select or recombine logits for a named diagnostic prediction head."""
    if head == "emg_only":
        return output.emg_logits
    if head == "imu_only":
        return output.imu_logits
    if head == "fixed_50_50":
        return 0.5 * output.emg_logits + 0.5 * output.imu_logits
    if head == "learned_fusion":
        return output.fused_logits
    raise ValueError(f"unsupported prediction head: {head}")


def variant_training_loss(
    output: ModelOutput,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    auxiliary_weight: float,
    variant: str,
) -> torch.Tensor:
    """Return the controlled objective associated with an ablation variant."""
    if variant == "emg_only":
        return functional.cross_entropy(output.emg_logits, targets, weight=class_weights)
    if variant == "imu_only":
        return functional.cross_entropy(output.imu_logits, targets, weight=class_weights)
    if variant in ("fixed_50_50", "learned_neutral", "learned_physics"):
        return fused_training_loss(output, targets, class_weights, auxiliary_weight)
    raise ValueError(f"unsupported training variant: {variant}")
