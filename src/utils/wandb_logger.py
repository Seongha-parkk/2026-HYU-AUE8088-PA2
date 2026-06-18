"""Lightweight Weights & Biases wrapper for the multi-task trainer.

Behaves as a no-op when:
  - ``project`` is None,
  - the ``WANDB_DISABLED`` env var is set to a truthy value, or
  - the ``wandb`` package is not installed.

This means student code that uses ``WandbLogger`` runs unmodified whether
or not wandb is configured — calling ``.log(...)`` on a disabled logger
is silently ignored.

Typical usage in a notebook:

    logger = WandbLogger(
        project="aue8088-pa2",
        run_name="level1-resnet18",
        config={"lr": 3e-4, "epochs": 20, "seed": 42},
        tags=["level1", "resnet18"],
    )
    trainer = MultiTaskTrainer(..., logger=logger)
    trainer.fit(train_loader, val_loader)
    logger.finish()
"""
from __future__ import annotations

import os
from typing import Any, Mapping


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


class WandbLogger:
    """Thin wrapper that silently degrades to a no-op."""

    def __init__(
        self,
        project: str | None,
        run_name: str | None = None,
        config: dict | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
    ) -> None:
        self.run = None

        if project is None:
            return
        if _truthy(os.environ.get("WANDB_DISABLED")):
            print("[wandb] WANDB_DISABLED is set — skipping wandb.")
            return

        try:
            import wandb  # noqa: WPS433  (deferred import is intentional)
        except ImportError:
            print("[wandb] package not installed — skipping wandb logging.")
            return

        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            notes=notes,
            reinit=True,  # allow multiple runs from one notebook session
        )

    @property
    def enabled(self) -> bool:
        return self.run is not None

    # --- Scalar / metric logging --------------------------------------------------

    def log(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(dict(metrics), step=step)

    def update_config(self, **kwargs: Any) -> None:
        if self.run is not None:
            self.run.config.update(kwargs, allow_val_change=True)

    # --- Image / figure logging ---------------------------------------------------

    def log_image(self, name: str, image: Any, step: int | None = None) -> None:
        """``image`` may be a matplotlib Figure, a PIL Image, or an HxWxC array."""
        if self.run is None:
            return
        import wandb
        self.run.log({name: wandb.Image(image)}, step=step)

    def log_confusion_matrix(
        self,
        name: str,
        cm,
        class_names: list[str],
        step: int | None = None,
    ) -> None:
        """Log a normalized confusion matrix as a heatmap image."""
        if self.run is None:
            return
        import matplotlib.pyplot as plt
        import seaborn as sns
        import wandb

        fig, ax = plt.subplots(figsize=(4.5, 4))
        sns.heatmap(
            cm,
            annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            cbar=False, ax=ax,
        )
        ax.set_xlabel("pred"); ax.set_ylabel("true"); ax.set_title(name)
        fig.tight_layout()
        self.run.log({name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def log_table(self, name: str, columns: list[str], rows: list[list[Any]]) -> None:
        if self.run is None:
            return
        import wandb
        self.run.log({name: wandb.Table(columns=columns, data=rows)})

    # --- Lifecycle ----------------------------------------------------------------

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN001
        self.finish()
