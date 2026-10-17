from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import dump_config
from .data.dataset import ChangeDetectionDataset, collate_change_batch
from .losses import compute_loss
from .metrics import BinaryConfusion, ThresholdSearch
from .models import TBDINO, build_model
from .utils import (
    append_jsonl,
    atomic_torch_save,
    choose_device,
    cosine_warmup_lambda,
    set_seed,
    trainable_parameter_summary,
    write_json,
)


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    output = dict(batch)
    for key in ("images", "mask", "valid_mask", "temporal_mask", "alpha"):
        value = output.get(key)
        if isinstance(value, torch.Tensor):
            output[key] = value.to(device, non_blocking=True)
    return output


def build_dataset_config(config: dict[str, Any]) -> dict[str, Any]:
    data_config = dict(config.get("data", {}))
    data_config["use_alphaearth"] = bool(
        config.get("model", {}).get("use_alphaearth", False)
    )
    return data_config


def build_loader(
    manifest: str,
    config: dict[str, Any],
    *,
    training: bool,
) -> DataLoader:
    data_config = build_dataset_config(config)
    dataset = ChangeDetectionDataset(
        manifest,
        data_config,
        training=training,
        validate_files=True,
    )
    batch_size = int(
        data_config.get("batch_size" if training else "eval_batch_size", 1)
    )
    workers = int(data_config.get("num_workers", 0))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(data_config.get("pin_memory", True)),
        drop_last=bool(data_config.get("drop_last", False) and training),
        collate_fn=collate_change_batch,
        persistent_workers=workers > 0,
    )


def _forward(model: TBDINO, batch: dict[str, Any]):
    return model(
        batch["images"],
        batch["temporal_mask"],
        alpha=batch.get("alpha"),
        change_mask=batch.get("mask"),
        valid_mask=batch.get("valid_mask"),
    )


@torch.no_grad()
def evaluate_model(
    model: TBDINO,
    loader: DataLoader,
    device: torch.device,
    loss_config: dict[str, Any],
    *,
    threshold: float = 0.5,
    search_threshold: bool = False,
) -> dict[str, float | int]:
    model.eval()
    tracker: BinaryConfusion | ThresholdSearch
    tracker = ThresholdSearch() if search_threshold else BinaryConfusion(threshold)
    totals: dict[str, float] = defaultdict(float)
    batches = 0
    for batch in tqdm(loader, desc="评估", leave=False):
        batch = move_batch(batch, device)
        output = _forward(model, batch)
        _, losses = compute_loss(
            output, batch["mask"], batch["valid_mask"], loss_config
        )
        for key, value in losses.items():
            totals[key] += value
        probabilities = torch.sigmoid(output.logits)
        tracker.update(probabilities, batch["mask"], batch["valid_mask"])
        batches += 1
    metrics = tracker.best() if isinstance(tracker, ThresholdSearch) else tracker.compute()
    for key, value in totals.items():
        metrics[f"val_{key}"] = value / max(batches, 1)
    return metrics


def save_checkpoint(
    path: str | Path,
    model: TBDINO,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.cuda.amp.GradScaler | None,
    config: dict[str, Any],
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    threshold: float,
) -> None:
    save_full_model = bool(config.get("training", {}).get("save_full_model", False))
    if save_full_model:
        model_state = {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        }
        state_mode = "full"
    else:
        trainable_names = {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        model_state = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
            if key in trainable_names
        }
        state_mode = "trainable"
    if not model_state:
        raise RuntimeError("检查点中没有可保存的模型参数")
    atomic_torch_save(
        {
            "version": 1,
            "model_state": model_state,
            "model_state_mode": state_mode,
            "optimizer_state": optimizer.state_dict() if optimizer else None,
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict() if scaler else None,
            "config": {k: v for k, v in config.items() if not k.startswith("_")},
            "epoch": epoch,
            "global_step": global_step,
            "metrics": metrics,
            "threshold": threshold,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: TBDINO,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    strict = checkpoint.get("model_state_mode", "full") == "full"
    incompatible = model.load_state_dict(checkpoint["model_state"], strict=strict)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"检查点包含当前模型无法识别的参数: {incompatible.unexpected_keys}"
        )
    if optimizer is not None and checkpoint.get("optimizer_state"):
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=bool(config.get("deterministic", True)))
    training = config.get("training", {})
    data = config.get("data", {})
    output_dir = Path(config.get("output_dir", "outputs/run")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dump_config(config, output_dir / "resolved_config.yaml")
    metrics_log = output_dir / "metrics.jsonl"
    if metrics_log.exists() and not training.get("resume"):
        metrics_log.unlink()

    device = choose_device(str(training.get("device", "auto")))
    model = build_model(config.get("model", {})).to(device)
    parameter_summary = trainable_parameter_summary(model)
    write_json(parameter_summary, output_dir / "parameters.json")
    if parameter_summary["trainable"] == 0:
        raise RuntimeError("模型没有可训练参数")

    train_manifest = str(data["train_manifest"])
    val_manifest = str(data["val_manifest"])
    train_loader = build_loader(train_manifest, config, training=True)
    val_loader = build_loader(val_manifest, config, training=False)

    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(training.get("learning_rate", 2e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    epochs = int(training.get("epochs", 50))
    accumulation = max(1, int(training.get("gradient_accumulation", 1)))
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_steps = max(1, epochs * optimizer_steps_per_epoch)
    warmup_steps = int(training.get("warmup_steps", 0))
    if not warmup_steps:
        warmup_steps = int(total_steps * float(training.get("warmup_ratio", 0.05)))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_lambda(
            step, warmup_steps, total_steps
        ),
    )
    amp_enabled = bool(training.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    global_step = 0
    best_f1 = -1.0
    threshold = float(training.get("threshold", 0.5))

    resume_path = training.get("resume")
    if resume_path:
        checkpoint = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        threshold = float(checkpoint.get("threshold", threshold))
        best_f1 = float(checkpoint.get("metrics", {}).get("f1", best_f1))

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(output_dir / "tensorboard")
    except Exception:
        writer = None

    loss_config = config.get("loss", {})
    gradient_clip = float(training.get("gradient_clip", 1.0))
    optimizer.zero_grad(set_to_none=True)
    final_metrics: dict[str, Any] = {}
    for epoch in range(start_epoch, epochs):
        model.train()
        totals: dict[str, float] = defaultdict(float)
        batches = 0
        progress = tqdm(train_loader, desc=f"训练 {epoch + 1}/{epochs}")
        for batch_index, batch in enumerate(progress):
            batch = move_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                output = _forward(model, batch)
                loss, values = compute_loss(
                    output, batch["mask"], batch["valid_mask"], loss_config
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % accumulation == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in model.parameters()
                            if parameter.requires_grad
                        ],
                        gradient_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
            for key, value in values.items():
                totals[key] += value
            batches += 1
            progress.set_postfix(loss=f"{values['loss']:.4f}")

        train_metrics = {
            f"train_{key}": value / max(batches, 1) for key, value in totals.items()
        }
        validation_metrics = evaluate_model(
            model,
            val_loader,
            device,
            loss_config,
            threshold=threshold,
        )
        final_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **train_metrics,
            **validation_metrics,
        }
        append_jsonl(final_metrics, metrics_log)
        if writer is not None:
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, epoch)
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch,
            global_step,
            validation_metrics,
            threshold,
        )
        if float(validation_metrics["f1"]) > best_f1:
            best_f1 = float(validation_metrics["f1"])
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                epoch,
                global_step,
                validation_metrics,
                threshold,
            )
        print(
            f"Epoch {epoch + 1}: F1={validation_metrics['f1']:.4f}, "
            f"IoU={validation_metrics['iou']:.4f}"
        )
    if writer is not None:
        writer.close()
    return final_metrics


def run_evaluation(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    manifest: str | None = None,
    search_threshold: bool = True,
    write_threshold: bool = False,
) -> dict[str, Any]:
    evaluation = config.get("evaluation", {})
    device = choose_device(str(evaluation.get("device", "auto")))
    model = build_model(config.get("model", {})).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    target_manifest = manifest or config.get("data", {}).get("test_manifest")
    if not target_manifest:
        raise ValueError("评估需要 manifest 或 data.test_manifest")
    loader = build_loader(str(target_manifest), config, training=False)
    metrics = evaluate_model(
        model,
        loader,
        device,
        config.get("loss", {}),
        threshold=float(checkpoint.get("threshold", 0.5)),
        search_threshold=search_threshold,
    )
    output_dir = Path(config.get("output_dir", "outputs/run")).expanduser().resolve()
    write_json(metrics, output_dir / "evaluation.json")
    if write_threshold:
        checkpoint["threshold"] = float(metrics["threshold"])
        checkpoint["evaluation_metrics"] = metrics
        atomic_torch_save(checkpoint, checkpoint_path)
    return metrics
