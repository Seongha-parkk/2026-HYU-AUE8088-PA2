"""BDD100K Scene-Attribute Multi-task Dataset.

The instructor pre-resizes BDD100K images to 224x224 and parses the
attribute JSON into a flat schema:

    labels.json  ->  {
        "<image_id>": {
            "weather":   "clear" | "overcast" | "rainy" | "snowy" | "foggy" | "partly cloudy",
            "scene":     "city street" | "highway" | "residential",
            "timeofday": "daytime" | "night" | "dawn/dusk"
        },
        ...
    }

Set A is fully labeled and split into train / val / test.
Set B is unlabeled and used only for Level 5 (the 1,000-Pick).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset


# Canonical class orderings. DO NOT change — Kaggle expects these indices.
WEATHER_CLASSES = ["clear", "overcast", "rainy", "snowy", "foggy", "partly cloudy"]
SCENE_CLASSES = ["city street", "highway", "residential"]
TIMEOFDAY_CLASSES = ["daytime", "night", "dawn/dusk"]

ATTRIBUTES = ("weather", "scene", "timeofday")
NUM_CLASSES = {
    "weather": len(WEATHER_CLASSES),
    "scene": len(SCENE_CLASSES),
    "timeofday": len(TIMEOFDAY_CLASSES),
}


def _str_to_idx(name: str, classes: list[str]) -> int:
    return classes.index(name)


def _make_sample(image_id: str, image_path: Path, attr: dict) -> "Sample":
    """attr 가 비어 있으면 -1 (unlabeled) 로 채운 Sample 을 반환."""
    return Sample(
        image_id=image_id,
        image_path=image_path,
        weather=_str_to_idx(attr["weather"], WEATHER_CLASSES) if "weather" in attr else -1,
        scene=_str_to_idx(attr["scene"], SCENE_CLASSES) if "scene" in attr else -1,
        timeofday=_str_to_idx(attr["timeofday"], TIMEOFDAY_CLASSES) if "timeofday" in attr else -1,
    )


@dataclass
class Sample:
    image_id: str
    image_path: Path
    weather: int = -1     # -1 means unlabeled (Set B)
    scene: int = -1
    timeofday: int = -1


class BDDAttrDataset(Dataset):
    """Multi-task scene-attribute classification dataset.

    Args:
        root: Path to ``data/set_a`` or ``data/set_b``.
        split: One of ``train``, ``val``, ``test``, or ``mining``
               (the latter for unlabeled Set B).
        transform: Albumentations / torchvision transform. Receives a PIL
                   image and must return a tensor of shape (3, H, W).
        extra_picks: Optional list of (image_id, weather_idx, scene_idx,
                     timeofday_idx) tuples — used by Level 5 to inject
                     student-curated samples (with student-provided
                     pseudo-labels) into the training set.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        extra_picks: Optional[list[tuple[str, int, int, int]]] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.samples: list[Sample] = []
        self._load_split()

        if extra_picks:
            self._inject_extra(extra_picks)

    def _load_split(self) -> None:
        # Set B (mining): 이미지 풀 + 라벨이 함께 제공됨 (Level 5 의 1,000-Pick 용)
        if self.split == "mining":
            meta = json.loads((self.root / "metadata.json").read_text())
            labels = json.loads((self.root / "labels.json").read_text())
            for image_id in meta["image_ids"]:
                self.samples.append(_make_sample(
                    image_id=image_id,
                    image_path=self.root / "images" / f"{image_id}.jpg",
                    attr=labels.get(image_id, {}),
                ))
            return

        # Set A: train/val 은 라벨 포함, test 는 라벨 비공개 (Kaggle 채점용)
        labels_path = self.root / "labels.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}

        split_path = self.root / f"{self.split}_ids.txt"
        ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]

        for image_id in ids:
            self.samples.append(_make_sample(
                image_id=image_id,
                image_path=self.root / self.split / f"{image_id}.jpg",
                attr=labels.get(image_id, {}),    # test 의 경우 빈 dict → -1 라벨
            ))

    def _inject_extra(
        self, picks: list[tuple[str, int, int, int]]
    ) -> None:
        """Append Level-5 picks. Pseudo-labels are student-supplied."""
        # Set B images live next to metadata.json, not in our split folder.
        set_b_root = self.root.parent / "set_b" / "images"
        for image_id, w, s, t in picks:
            self.samples.append(
                Sample(
                    image_id=image_id,
                    image_path=set_b_root / f"{image_id}.jpg",
                    weather=w,
                    scene=s,
                    timeofday=t,
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {
            "image": img,
            "image_id": s.image_id,
            "weather": torch.tensor(s.weather, dtype=torch.long),
            "scene": torch.tensor(s.scene, dtype=torch.long),
            "timeofday": torch.tensor(s.timeofday, dtype=torch.long),
        }

    def class_counts(self, attribute: str) -> torch.Tensor:
        """Number of samples per class (for weighted-loss / sampler design)."""
        counts = torch.zeros(NUM_CLASSES[attribute], dtype=torch.long)
        for s in self.samples:
            label = getattr(s, attribute)
            if label >= 0:
                counts[label] += 1
        return counts
