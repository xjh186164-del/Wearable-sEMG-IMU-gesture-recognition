"""Run post-hoc diagnostics and controlled retraining ablations."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .intervals import CLASS_LABELS
from .metrics import classification_metrics
from .model import DualBranchCNN, ModelOutput
from .training_data import NormalizationStats, normalize_arrays
from .train import TrainingConfig, load_training_dataset, train_model
from .variants import (
    TRAINING_VARIANTS,
    build_model_for_variant,
    prediction_logits,
    primary_head_for_variant,
)


POSTHOC_HEADS = ("emg_only", "imu_only", "fixed_50_50", "learned_fusion")


def _json_write(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _majority(values: np.ndarray, class_count: int) -> int:
    counts = np.bincount(values.astype(np.int64, copy=False), minlength=class_count)
    return int(np.flatnonzero(counts == counts.max())[0])


def summarize_grouped_results(
    truth: np.ndarray,
    prediction: np.ndarray,
    session_ids: np.ndarray,
    trial_ids: np.ndarray,
    class_labels=CLASS_LABELS,
) -> dict:
    """Summarise all windows per session and non-REST target trials by majority vote."""
    truth = np.asarray(truth, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    sessions = np.asarray(session_ids).astype(str)
    trials = np.asarray(trial_ids)
    if not (truth.shape == prediction.shape == sessions.shape == trials.shape) or truth.ndim != 1:
        raise ValueError("grouped-evaluation arrays must be aligned one-dimensional arrays")
    if truth.size == 0:
        raise ValueError("grouped-evaluation arrays must not be empty")

    per_session = {}
    for session in sorted(set(sessions.tolist())):
        mask = sessions == session
        per_session[session] = classification_metrics(
            truth[mask], prediction[mask], class_labels,
        )
        per_session[session]["window_count"] = int(np.count_nonzero(mask))

    target_mask = truth != 0
    group_keys = sorted(set(zip(sessions[target_mask].tolist(), trials[target_mask].tolist())))
    trial_truth = []
    trial_prediction = []
    trial_records = []
    for session, trial in group_keys:
        mask = target_mask & (sessions == session) & (trials == trial)
        labels = np.unique(truth[mask])
        if labels.size != 1:
            raise ValueError(
                f"session {session} trial {trial} contains multiple target labels",
            )
        expected = int(labels[0])
        predicted = _majority(prediction[mask], len(class_labels))
        trial_truth.append(expected)
        trial_prediction.append(predicted)
        trial_records.append({
            "session_id": session,
            "trial_id": int(trial),
            "true_index": expected,
            "true_label": str(class_labels[expected]),
            "predicted_index": predicted,
            "predicted_label": str(class_labels[predicted]),
            "window_count": int(np.count_nonzero(mask)),
            "correct": bool(expected == predicted),
        })

    trial_truth_array = np.asarray(trial_truth, dtype=np.int64)
    trial_prediction_array = np.asarray(trial_prediction, dtype=np.int64)
    correct = int(np.count_nonzero(trial_truth_array == trial_prediction_array))
    per_class = []
    for index, label in enumerate(class_labels[1:], start=1):
        mask = trial_truth_array == index
        support = int(np.count_nonzero(mask))
        class_correct = int(np.count_nonzero(trial_prediction_array[mask] == index))
        per_class.append({
            "label": str(label),
            "correct_trials": class_correct,
            "trial_count": support,
            "recall": float(class_correct / support) if support else None,
        })
    return {
        "per_session": per_session,
        "target_trial_majority": {
            "accuracy": float(correct / len(trial_records)),
            "correct_trials": correct,
            "trial_count": len(trial_records),
            "per_class": per_class,
            "trials": trial_records,
        },
    }


def _load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        raise ValueError(f"malformed checkpoint: {checkpoint_path}")
    hyperparameters = checkpoint.get("hyperparameters", {})
    variant = hyperparameters.get("variant", "learned_physics")
    model = build_model_for_variant(variant, class_count=len(CLASS_LABELS))
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    normalization = checkpoint["normalization"]
    stats = NormalizationStats(
        emg_mean=normalization["emg_mean"].numpy().copy(),
        emg_std=normalization["emg_std"].numpy().copy(),
        imu_mean=normalization["imu_mean"].numpy().copy(),
        imu_std=normalization["imu_std"].numpy().copy(),
    )
    return checkpoint, variant, model, stats


def evaluate_checkpoint(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    *,
    heads: tuple[str, ...] | None = None,
    device: str = "cuda",
    batch_size: int = 512,
) -> tuple[dict, list[dict]]:
    """Evaluate selected heads and return metrics plus window-level records."""
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation was requested but CUDA is unavailable")
    dataset = load_training_dataset(dataset_path)
    checkpoint, variant, model, stats = _load_checkpoint_model(Path(checkpoint_path), requested)
    normalized_emg, normalized_imu = normalize_arrays(dataset.emg, dataset.imu, stats)
    test_sessions = tuple(checkpoint["split"]["test"])
    index = np.flatnonzero(np.isin(dataset.session_ids.astype(str), test_sessions))
    if index.size == 0:
        raise ValueError("checkpoint test split contains no dataset windows")
    selected_heads = heads or (primary_head_for_variant(variant),)
    unknown = set(selected_heads) - set(POSTHOC_HEADS)
    if unknown:
        raise ValueError(f"unsupported evaluation heads: {sorted(unknown)}")
    predictions = {head: [] for head in selected_heads}
    with torch.inference_mode():
        for start in range(0, index.size, batch_size):
            rows = index[start:start + batch_size]
            emg = torch.from_numpy(normalized_emg[rows]).to(requested)
            imu = torch.from_numpy(normalized_imu[rows]).to(requested)
            output = model(emg, imu)
            for head in selected_heads:
                logits = prediction_logits(output, head)
                predictions[head].append(logits.argmax(dim=1).cpu().numpy())

    truth = dataset.labels[index].astype(np.int64, copy=False)
    sessions = dataset.session_ids[index].astype(str)
    trials = dataset.trial_ids[index]
    result = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "trained_variant": variant,
        "test_window_count": int(index.size),
        "heads": {},
    }
    concatenated = {}
    for head in selected_heads:
        predicted = np.concatenate(predictions[head]).astype(np.int64, copy=False)
        concatenated[head] = predicted
        metrics = classification_metrics(truth, predicted, CLASS_LABELS)
        metrics["grouped"] = summarize_grouped_results(
            truth, predicted, sessions, trials, CLASS_LABELS,
        )
        result["heads"][head] = metrics

    records = []
    for position, dataset_index in enumerate(index):
        record = {
            "dataset_index": int(dataset_index),
            "session_id": str(sessions[position]),
            "trial_id": int(trials[position]),
            "true_index": int(truth[position]),
            "true_label": str(CLASS_LABELS[int(truth[position])]),
        }
        for head in selected_heads:
            predicted = int(concatenated[head][position])
            record[f"{head}_index"] = predicted
            record[f"{head}_label"] = str(CLASS_LABELS[predicted])
        records.append(record)
    return result, records


def _write_records(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _mean_std(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_ablation(
    dataset_path: str | Path,
    baseline_checkpoint: str | Path,
    output_dir: str | Path,
    *,
    seeds: tuple[int, ...] = (20260821, 20260822, 20260823),
    device: str = "cuda",
) -> dict:
    """Run post-hoc diagnostics followed by five independently trained variants."""
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"ablation output directory already exists: {output}")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("ablation seeds must be a nonempty unique tuple")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        posthoc_dir = stage / "posthoc_existing_checkpoint"
        posthoc_dir.mkdir()
        posthoc, posthoc_records = evaluate_checkpoint(
            dataset_path,
            baseline_checkpoint,
            heads=POSTHOC_HEADS,
            device=device,
        )
        posthoc["interpretation"] = (
            "Diagnostic only: the two branch heads were co-trained with the learned-fusion "
            "objective, so these results are not substitutes for independently trained ablations."
        )
        _json_write(posthoc_dir / "metrics.json", posthoc)
        _write_records(posthoc_dir / "test_predictions.csv", posthoc_records)

        run_rows = []
        run_documents = []
        for variant in TRAINING_VARIANTS:
            for seed in seeds:
                run_dir = stage / "retrained" / variant / f"seed_{seed}"
                config = TrainingConfig(seed=seed, device=device, variant=variant)
                print(f"Starting variant={variant} seed={seed}", flush=True)
                training_metrics = train_model(dataset_path, run_dir, config)
                evaluation, records = evaluate_checkpoint(
                    dataset_path,
                    run_dir / "best_model.pt",
                    device=device,
                )
                head = primary_head_for_variant(variant)
                evaluated = evaluation["heads"][head]
                _json_write(run_dir / "grouped_test_metrics.json", evaluated["grouped"])
                _write_records(run_dir / "test_predictions.csv", records)
                trial = evaluated["grouped"]["target_trial_majority"]
                session_macro = [
                    value["macro_f1"] for value in evaluated["grouped"]["per_session"].values()
                ]
                final_weights = training_metrics["learned_emg_weights"]
                row = {
                    "variant": variant,
                    "seed": seed,
                    "best_epoch": training_metrics["best_epoch"],
                    "trainable_parameters": training_metrics["trainable_parameters"],
                    "test_accuracy": evaluated["accuracy"],
                    "test_balanced_accuracy": evaluated["balanced_accuracy"],
                    "test_macro_f1": evaluated["macro_f1"],
                    "mean_session_macro_f1": float(np.mean(session_macro)),
                    "target_trial_accuracy": trial["accuracy"],
                    "correct_target_trials": trial["correct_trials"],
                    "target_trial_count": trial["trial_count"],
                }
                for label in CLASS_LABELS:
                    row[f"emg_weight_{label}"] = final_weights[label]
                run_rows.append(row)
                run_documents.append({
                    "variant": variant,
                    "seed": seed,
                    "training_metrics": training_metrics,
                    "evaluation": evaluation,
                })

        with (stage / "run_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
            writer.writeheader()
            writer.writerows(run_rows)

        aggregate = {}
        aggregate_rows = []
        for variant in TRAINING_VARIANTS:
            rows = [row for row in run_rows if row["variant"] == variant]
            summary = {
                key: _mean_std([float(row[key]) for row in rows])
                for key in (
                    "test_accuracy",
                    "test_balanced_accuracy",
                    "test_macro_f1",
                    "mean_session_macro_f1",
                    "target_trial_accuracy",
                )
            }
            summary["seeds"] = list(seeds)
            summary["trainable_parameters"] = int(rows[0]["trainable_parameters"])
            summary["best_epochs"] = [int(row["best_epoch"]) for row in rows]
            summary["final_emg_weights"] = {
                label: _mean_std([float(row[f"emg_weight_{label}"]) for row in rows])
                for label in CLASS_LABELS
            }
            aggregate[variant] = summary
            aggregate_rows.append({
                "variant": variant,
                "trainable_parameters": summary["trainable_parameters"],
                "accuracy_mean_percent": 100.0 * summary["test_accuracy"]["mean"],
                "accuracy_sd_percentage_points": 100.0 * summary["test_accuracy"]["standard_deviation"],
                "balanced_accuracy_mean_percent": 100.0 * summary["test_balanced_accuracy"]["mean"],
                "balanced_accuracy_sd_percentage_points": 100.0 * summary["test_balanced_accuracy"]["standard_deviation"],
                "macro_f1_mean_percent": 100.0 * summary["test_macro_f1"]["mean"],
                "macro_f1_sd_percentage_points": 100.0 * summary["test_macro_f1"]["standard_deviation"],
                "mean_session_macro_f1_percent": 100.0 * summary["mean_session_macro_f1"]["mean"],
                "target_trial_accuracy_percent": 100.0 * summary["target_trial_accuracy"]["mean"],
            })
        with (stage / "aggregate_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
            writer.writeheader()
            writer.writerows(aggregate_rows)
        document = {
            "schema_version": "gesture_ablation_v1",
            "dataset": str(Path(dataset_path).resolve()),
            "baseline_checkpoint": str(Path(baseline_checkpoint).resolve()),
            "seeds": list(seeds),
            "posthoc": posthoc,
            "aggregate": aggregate,
            "runs": run_documents,
        }
        _json_write(stage / "ablation_results.json", document)
        os.replace(stage, output)
        return document
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--seeds",
        default="20260821,20260822,20260823",
        help="comma-separated fixed random seeds",
    )
    args = parser.parse_args(argv)
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    result = run_ablation(
        args.dataset,
        args.baseline_checkpoint,
        args.output_dir,
        seeds=seeds,
        device=args.device,
    )
    for variant in TRAINING_VARIANTS:
        score = result["aggregate"][variant]["test_macro_f1"]
        print(
            f"{variant}: test Macro-F1={100 * score['mean']:.3f}% "
            f"+/- {100 * score['standard_deviation']:.3f} percentage points",
        )


if __name__ == "__main__":
    main()


__all__ = [
    "TRAINING_VARIANTS",
    "build_model_for_variant",
    "prediction_logits",
    "summarize_grouped_results",
    "evaluate_checkpoint",
    "run_ablation",
]
