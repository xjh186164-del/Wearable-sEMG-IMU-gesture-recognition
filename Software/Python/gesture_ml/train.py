"""Train the fixed personalized dual-branch gesture CNN."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .artifacts import ARTIFACT_SUFFIXES, DEFAULT_SUBJECT_ALIASES
from .intervals import CLASS_LABELS
from .metrics import classification_metrics
from .model import DualBranchCNN, fused_training_loss
from .training_data import (
    EXPECTED_SESSIONS,
    NormalizationStats,
    compute_normalization,
    fixed_session_split,
    inverse_sqrt_class_weights,
    normalize_arrays,
)
from .windowing import WindowBatch


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 20260821
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 100
    patience: int = 12
    auxiliary_weight: float = 0.1
    device: str = "cuda"


@dataclass(frozen=True)
class TrainingDataset:
    path: Path
    manifest: dict
    emg: np.ndarray
    imu: np.ndarray
    labels: np.ndarray
    physical_subject_ids: np.ndarray
    acquisition_subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    block_ids: np.ndarray
    interval_ids: np.ndarray


@dataclass(frozen=True)
class PreparedTrainingData:
    normalized_emg: np.ndarray
    normalized_imu: np.ndarray
    labels: np.ndarray
    indices: dict[str, np.ndarray]
    split: dict[str, tuple[str, ...]]
    stats: NormalizationStats
    class_weights: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read dataset manifest: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("dataset manifest root must be an object")
    return value


def load_training_dataset(dataset_path: str | Path) -> TrainingDataset:
    """Load and fail closed on any deviation from the official window contract."""
    path = Path(dataset_path).resolve()
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise ValueError(f"dataset NPZ does not exist: {path}")
    manifest_path = path.with_suffix(".manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest does not exist: {manifest_path}")
    manifest = _load_json_object(manifest_path)
    expected_manifest_fields = {
        "schema_version",
        "protocol_version",
        "class_labels",
        "window_seconds",
        "step_seconds",
        "guard_seconds",
        "emg_shape_per_window",
        "imu_shape_per_window",
        "subject_aliases",
        "accepted_sessions",
        "rejected_sessions",
        "window_count",
        "class_counts",
        "source_sha256",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("dataset manifest fields do not match the official build contract")
    if manifest.get("schema_version") != "gesture_window_dataset_v1":
        raise ValueError("dataset schema is not gesture_window_dataset_v1")
    if manifest.get("protocol_version") != "eight_pose_v5":
        raise ValueError("dataset protocol is not eight_pose_v5")
    if manifest.get("class_labels") != list(CLASS_LABELS):
        raise ValueError("dataset class order does not match the training contract")
    if manifest.get("emg_shape_per_window") != [8, 250] or manifest.get("imu_shape_per_window") != [6, 52]:
        raise ValueError("dataset manifest tensor shapes do not match the training contract")
    if (
        manifest.get("window_seconds") != 0.5
        or manifest.get("step_seconds") != 0.1
        or manifest.get("guard_seconds") != 0.4
    ):
        raise ValueError("dataset window or guard settings do not match the training contract")
    if manifest.get("subject_aliases") != DEFAULT_SUBJECT_ALIASES:
        raise ValueError("dataset subject aliases do not map S001/S002/S003 to P001")
    accepted = manifest.get("accepted_sessions")
    if not isinstance(accepted, dict) or set(accepted) != set(EXPECTED_SESSIONS):
        raise ValueError("dataset manifest session set does not match the fixed split")
    expected_rejections = [{
        "session_id": "S003_D07_R02",
        "reason": "quality gate did not pass cleanly",
    }]
    if manifest.get("rejected_sessions") != expected_rejections:
        raise ValueError("dataset rejected-session record does not match the approved exclusion")
    source_hashes = manifest.get("source_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(EXPECTED_SESSIONS):
        raise ValueError("dataset source SHA-256 session set is incomplete")
    sha_pattern = re.compile(r"[0-9a-f]{64}")
    for session in EXPECTED_SESSIONS:
        expected_files = {f"{session}{suffix}" for suffix in ARTIFACT_SUFFIXES}
        session_hashes = source_hashes[session]
        if not isinstance(session_hashes, dict) or set(session_hashes) != expected_files:
            raise ValueError(f"dataset source SHA-256 files are incomplete for {session}")
        if any(not isinstance(value, str) or sha_pattern.fullmatch(value) is None for value in session_hashes.values()):
            raise ValueError(f"dataset source SHA-256 value is invalid for {session}")

    required_fields = set(WindowBatch.__dataclass_fields__)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required_fields:
                raise ValueError("dataset NPZ fields do not match the window contract")
            arrays = {name: archive[name].copy() for name in required_fields}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("dataset NPZ"):
            raise
        raise ValueError(f"could not read dataset NPZ: {error}") from error

    emg = arrays["emg"]
    imu = arrays["imu"]
    labels = arrays["labels"]
    row_count = labels.size
    if emg.shape != (row_count, 8, 250) or imu.shape != (row_count, 6, 52):
        raise ValueError("dataset sensor array shape does not match [N,8,250] and [N,6,52]")
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("dataset labels must be a one-dimensional integer array")
    if row_count == 0 or int(labels.min()) < 0 or int(labels.max()) >= len(CLASS_LABELS):
        raise ValueError("dataset labels are empty or outside the class range")
    if not np.all(np.isfinite(emg)) or not np.all(np.isfinite(imu)):
        raise ValueError("dataset sensor arrays must contain only finite values")
    for name in required_fields - {"emg", "imu", "labels"}:
        if arrays[name].ndim != 1 or arrays[name].size != row_count:
            raise ValueError(f"dataset field {name} must have one value per window")
        if arrays[name].dtype.kind == "O":
            raise ValueError(f"dataset field {name} must not use object dtype")
    if set(arrays["physical_subject_ids"].astype(str)) != {"P001"}:
        raise ValueError("dataset must contain only physical user P001")
    if not set(arrays["acquisition_subject_ids"].astype(str)).issubset({"S001", "S002", "S003"}):
        raise ValueError("dataset contains an unsupported acquisition subject ID")
    split = fixed_session_split(arrays["session_ids"].astype(str))
    del split
    observed_counts = {
        session: int(np.count_nonzero(arrays["session_ids"].astype(str) == session))
        for session in EXPECTED_SESSIONS
    }
    try:
        manifest_counts = {str(key): int(value) for key, value in accepted.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("dataset manifest accepted-session counts are invalid") from error
    if manifest_counts != observed_counts or manifest.get("window_count") != row_count:
        raise ValueError("dataset manifest window counts do not match the NPZ")
    observed_class_counts = {
        label: int(np.count_nonzero(labels == index))
        for index, label in enumerate(CLASS_LABELS)
    }
    if manifest.get("class_counts") != observed_class_counts:
        raise ValueError("dataset manifest class counts do not match the NPZ")

    return TrainingDataset(path=path, manifest=manifest, **arrays)


def prepare_training_data(dataset: TrainingDataset) -> PreparedTrainingData:
    split = fixed_session_split(dataset.session_ids.astype(str))
    indices = {
        name: np.flatnonzero(np.isin(dataset.session_ids.astype(str), sessions))
        for name, sessions in split.items()
    }
    if any(value.size == 0 for value in indices.values()):
        raise ValueError("every fixed split must contain at least one window")
    stats = compute_normalization(dataset.emg, dataset.imu, indices["train"])
    normalized_emg, normalized_imu = normalize_arrays(dataset.emg, dataset.imu, stats)
    class_weights = inverse_sqrt_class_weights(
        dataset.labels,
        indices["train"],
        len(CLASS_LABELS),
    )
    return PreparedTrainingData(
        normalized_emg=normalized_emg,
        normalized_imu=normalized_imu,
        labels=dataset.labels.astype(np.int64, copy=False),
        indices=indices,
        split=split,
        stats=stats,
        class_weights=class_weights,
    )


def set_reproducible(seed: int):
    if not isinstance(seed, int) or seed < 0 or seed > 2**32 - 1:
        raise ValueError("training seed must be an integer in [0, 2^32-1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _validate_config(config: TrainingConfig) -> torch.device:
    if config.batch_size <= 0 or config.max_epochs <= 0 or config.patience <= 0:
        raise ValueError("batch size, maximum epochs, and patience must be positive")
    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("learning rate must be positive and weight decay nonnegative")
    if not 0 <= config.auxiliary_weight <= 1:
        raise ValueError("auxiliary weight must be between zero and one")
    requested = torch.device(config.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but CUDA is unavailable")
    return requested


def _loader(
    prepared: PreparedTrainingData,
    split_name: str,
    config: TrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    index = prepared.indices[split_name]
    dataset = TensorDataset(
        torch.from_numpy(prepared.normalized_emg[index]),
        torch.from_numpy(prepared.normalized_imu[index]),
        torch.from_numpy(prepared.labels[index]),
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=config.device.startswith("cuda"),
        generator=generator,
    )


def _run_epoch(
    model: DualBranchCNN,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    auxiliary_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    row_count = 0
    true_parts = []
    predicted_parts = []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for emg, imu, labels in loader:
            emg = emg.to(device, non_blocking=True)
            imu = imu.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            output = model(emg, imu)
            for name, value in (
                ("fused logits", output.fused_logits),
                ("EMG logits", output.emg_logits),
                ("IMU logits", output.imu_logits),
                ("fusion weights", output.emg_weights),
            ):
                if not bool(torch.isfinite(value).all().item()):
                    raise FloatingPointError(f"model produced non-finite {name}")
            loss = fused_training_loss(output, labels, class_weights, auxiliary_weight)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("training produced a non-finite loss")
            if training:
                loss.backward()
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item()):
                        raise FloatingPointError(f"training produced a non-finite gradient in {name}")
                optimizer.step()
                for name, value in model.state_dict().items():
                    if not bool(torch.isfinite(value).all().item()):
                        raise FloatingPointError(f"training produced non-finite model state in {name}")
            batch_rows = labels.numel()
            loss_sum += float(loss.detach().item()) * batch_rows
            row_count += batch_rows
            true_parts.append(labels.detach().cpu().numpy())
            predicted_parts.append(output.fused_logits.argmax(dim=1).detach().cpu().numpy())
    return (
        loss_sum / row_count,
        np.concatenate(true_parts),
        np.concatenate(predicted_parts),
    )


def _evaluate(
    model: DualBranchCNN,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    auxiliary_weight: float,
) -> dict:
    loss, truth, prediction = _run_epoch(
        model, loader, device, class_weights, auxiliary_weight, optimizer=None,
    )
    result = classification_metrics(truth, prediction, CLASS_LABELS)
    result["loss"] = float(loss)
    return result


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for name, value in model.state_dict().items():
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"cannot checkpoint non-finite model state in {name}")
        result[name] = value.detach().cpu().clone()
    return result


def _write_json(path: Path, value: dict):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_history(path: Path, history: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_loss", "validation_loss", "validation_macro_f1"),
        )
        writer.writeheader()
        writer.writerows(history)


def train_model(
    dataset_path: str | Path,
    output_dir: str | Path,
    config: TrainingConfig = TrainingConfig(),
) -> dict:
    """Train once and atomically publish a complete, non-overwriting run directory."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"training output directory already exists: {output}")
    device = _validate_config(config)
    set_reproducible(config.seed)
    dataset = load_training_dataset(dataset_path)
    prepared = prepare_training_data(dataset)
    dataset_hash = _sha256(dataset.path)
    manifest_hash = _sha256(dataset.path.with_suffix(".manifest.json"))

    model = DualBranchCNN(class_count=len(CLASS_LABELS)).to(device)
    class_weights = torch.from_numpy(prepared.class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    train_loader = _loader(prepared, "train", config, shuffle=True)
    validation_loader = _loader(prepared, "validation", config, shuffle=False)
    test_loader = _loader(prepared, "test", config, shuffle=False)

    best_epoch = 0
    best_validation_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss, _, _ = _run_epoch(
            model,
            train_loader,
            device,
            class_weights,
            config.auxiliary_weight,
            optimizer,
        )
        validation = _evaluate(
            model,
            validation_loader,
            device,
            class_weights,
            config.auxiliary_weight,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation["loss"],
            "validation_macro_f1": validation["macro_f1"],
        })
        print(
            f"Epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_loss={validation['loss']:.6f} val_macro_f1={validation['macro_f1']:.6f}",
            flush=True,
        )
        if validation["macro_f1"] > best_validation_f1 + 1e-12:
            best_validation_f1 = validation["macro_f1"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a best model state")
    model.load_state_dict(best_state)
    validation = _evaluate(
        model, validation_loader, device, class_weights, config.auxiliary_weight,
    )
    test = _evaluate(model, test_loader, device, class_weights, config.auxiliary_weight)
    learned_weights = model.emg_weights().detach().cpu().tolist()

    metrics = {
        "schema_version": "personalized_dual_cnn_metrics_v1",
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(best_validation_f1),
        "validation": validation,
        "test": test,
        "learned_emg_weights": {
            label: float(learned_weights[index]) for index, label in enumerate(CLASS_LABELS)
        },
        "learned_imu_weights": {
            label: float(1.0 - learned_weights[index]) for index, label in enumerate(CLASS_LABELS)
        },
    }
    split_document = {
        "schema_version": "personalized_session_split_v1",
        "dataset_sha256": dataset_hash,
        "dataset_manifest_sha256": manifest_hash,
        "class_labels": list(CLASS_LABELS),
        "sessions": {name: list(values) for name, values in prepared.split.items()},
        "window_counts": {name: int(values.size) for name, values in prepared.indices.items()},
    }
    checkpoint = {
        "schema_version": "personalized_dual_cnn_v1",
        "model_state": _cpu_state_dict(model),
        "normalization": {
            "emg_mean": torch.from_numpy(prepared.stats.emg_mean.copy()),
            "emg_std": torch.from_numpy(prepared.stats.emg_std.copy()),
            "imu_mean": torch.from_numpy(prepared.stats.imu_mean.copy()),
            "imu_std": torch.from_numpy(prepared.stats.imu_std.copy()),
        },
        "class_weights": torch.from_numpy(prepared.class_weights.copy()),
        "class_labels": list(CLASS_LABELS),
        "split": {name: list(values) for name, values in prepared.split.items()},
        "hyperparameters": asdict(config),
        "learned_emg_weights": learned_weights,
        "dataset_sha256": dataset_hash,
        "dataset_manifest_sha256": manifest_hash,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": float(best_validation_f1),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_json(stage / "split.json", split_document)
        _write_json(stage / "metrics.json", metrics)
        _write_history(stage / "history.csv", history)
        torch.save(checkpoint, stage / "best_model.pt")
        os.replace(stage, output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--auxiliary-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    config = TrainingConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        auxiliary_weight=args.auxiliary_weight,
        device=args.device,
    )
    metrics = train_model(args.dataset, args.output_dir, config)
    print(f"Best epoch: {metrics['best_epoch']}")
    print(f"Validation Macro-F1: {metrics['validation']['macro_f1']:.6f}")
    print(f"Test Macro-F1: {metrics['test']['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
