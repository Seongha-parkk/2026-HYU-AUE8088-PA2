"""Build a Kaggle submission CSV from model predictions."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.datasets.bdd_attr import (
    SCENE_CLASSES,
    TIMEOFDAY_CLASSES,
    WEATHER_CLASSES,
)


def write_submission(
    out_path: str | Path,
    image_ids: list[str],
    preds: dict[str, np.ndarray],
) -> None:
    """Submission format:

        image_id, weather, scene, timeofday
        b1c66a42-6f7d68ca, clear, city street, daytime
        ...
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "weather", "scene", "timeofday"])
        for i, image_id in enumerate(image_ids):
            w.writerow([
                image_id,
                WEATHER_CLASSES[int(preds["weather"][i])],
                SCENE_CLASSES[int(preds["scene"][i])],
                TIMEOFDAY_CLASSES[int(preds["timeofday"][i])],
            ])
