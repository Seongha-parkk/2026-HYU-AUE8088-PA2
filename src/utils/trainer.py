"""Generic multi-task trainer.

Loss policy: weighted sum of per-attribute losses.

    L = w_w * L_weather + w_s * L_scene + w_t * L_timeofday

Students may extend this to Uncertainty Weighting (Kendall+ 2018) or
GradNorm — see Level 1 / Level 3 of the assignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.bdd_attr import ATTRIBUTES
from src.utils.metrics import average_macro_f1, collect_predictions, per_attribute_macro_f1
from src.utils.wandb_logger import WandbLogger


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 5e-4
    loss_weights: dict = field(default_factory=lambda: {"weather": 1.0, "scene": 1.0, "timeofday": 1.0})
    grad_clip: float | None = 1.0
    log_every: int = 50
    amp: bool = True


class MultiTaskTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        loss_fns: dict[str, Callable],
        device: torch.device,
        config: TrainConfig,
        logger: WandbLogger | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fns = loss_fns
        self.device = device
        self.cfg = config
        self.scaler = torch.amp.GradScaler(enabled=config.amp)
        # No-op logger by default — calls are silently ignored.
        self.logger = logger if logger is not None else WandbLogger(project=None)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        history = {"train_loss": [], "val_avg_mf1": [], "val_per_mf1": []}

        for epoch in range(self.cfg.epochs):
            train_loss = self._train_one_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            history["train_loss"].append(train_loss)
            history["val_avg_mf1"].append(val_metrics["avg_macro_f1"])
            history["val_per_mf1"].append(val_metrics["per_macro_f1"])

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"[epoch {epoch+1:02d}/{self.cfg.epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_avg_MF1={val_metrics['avg_macro_f1']:.4f}  "
                f"per={val_metrics['per_macro_f1']}"
            )

            log_payload = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "val/avg_macro_f1": val_metrics["avg_macro_f1"],
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            for a, v in val_metrics["per_macro_f1"].items():
                log_payload[f"val/mf1_{a}"] = v
            self.logger.log(log_payload, step=epoch)

        return history

    def _train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        running = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"train e{epoch+1}", leave=False)
        for batch in pbar:
            x = batch["image"].to(self.device, non_blocking=True)
            y = {a: batch[a].to(self.device, non_blocking=True) for a in ATTRIBUTES}

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.cfg.amp):
                logits = self.model(x)
                loss = sum(
                    self.cfg.loss_weights[a] * self.loss_fns[a](logits[a], y[a])
                    for a in ATTRIBUTES
                )

            self.scaler.scale(loss).backward()
            if self.cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return running / max(n_batches, 1)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        preds, probs, targets, _ = collect_predictions(self.model, loader, self.device)
        return {
            "avg_macro_f1": average_macro_f1(preds, targets),
            "per_macro_f1": per_attribute_macro_f1(preds, targets),
            "preds": preds,
            "probs": probs,
            "targets": targets,
        }
