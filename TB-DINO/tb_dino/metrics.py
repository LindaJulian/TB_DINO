from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BinaryConfusion:
    threshold: float = 0.5
    true_positive: int = 0
    true_negative: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        prediction = probabilities >= self.threshold
        truth = target >= 0.5
        valid = (
            valid_mask >= 0.5
            if valid_mask is not None
            else torch.ones_like(truth, dtype=torch.bool)
        )
        prediction = prediction[valid]
        truth = truth[valid]
        self.true_positive += int((prediction & truth).sum())
        self.true_negative += int((~prediction & ~truth).sum())
        self.false_positive += int((prediction & ~truth).sum())
        self.false_negative += int((~prediction & truth).sum())

    def compute(self) -> dict[str, float | int]:
        tp = self.true_positive
        tn = self.true_negative
        fp = self.false_positive
        fn = self.false_negative
        epsilon = 1e-12
        oa = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, epsilon)
        recall = tp / max(tp + fn, epsilon)
        iou = tp / max(tp + fp + fn, epsilon)
        f1 = 2 * precision * recall / max(precision + recall, epsilon)
        return {
            "oa": oa,
            "ua_precision": precision,
            "pa_recall": recall,
            "iou": iou,
            "f1": f1,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "threshold": self.threshold,
        }


class ThresholdSearch:
    def __init__(self, thresholds: list[float] | None = None) -> None:
        thresholds = thresholds or [index / 20 for index in range(2, 19)]
        self.trackers = [BinaryConfusion(value) for value in thresholds]

    def update(
        self,
        probabilities: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        for tracker in self.trackers:
            tracker.update(probabilities, target, valid_mask)

    def best(self) -> dict[str, float | int]:
        results = [tracker.compute() for tracker in self.trackers]
        return max(results, key=lambda item: (float(item["f1"]), float(item["iou"])))

