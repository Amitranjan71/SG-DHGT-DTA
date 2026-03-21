from __future__ import annotations

import math

from .config import SGDHGTDTAConfig
from .model import ModelOutputs
from .tensor_ops import l2_normalize_rows, mean


def info_nce_loss(embeddings: list[list[float]]) -> float:
    normalized = l2_normalize_rows(embeddings)
    similarities = []
    for i, row in enumerate(normalized):
        similarities.append(sum(row[j] * normalized[i][j] for j in range(len(row))))
    return 1.0 - mean(similarities)


def graph_consistency_loss(dynamic_scores: list[float], known_mask_diag: list[float]) -> float:
    diffs = [(score * flag - score * flag) ** 2 for score, flag in zip(dynamic_scores, known_mask_diag)]
    return mean(diffs)


def huber_loss(pred: list[list[float]], target: list[list[float]], delta: float = 1.0) -> float:
    values = []
    for p, t in zip(pred, target):
        err = abs(p[0] - t[0])
        values.append(0.5 * err * err if err <= delta else delta * (err - 0.5 * delta))
    return mean(values)


def binary_calibration_loss(pred: list[list[float]], target: list[list[float]]) -> float:
    values = []
    for p, t in zip(pred, target):
        clipped = min(max(p[0], 1e-6), 1.0 - 1e-6)
        label = 1.0 / (1.0 + math.exp(-t[0]))
        values.append(-(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped)))
    return mean(values)


def compute_multi_objective_losses(outputs: ModelOutputs, labels: list[list[float]], known_mask_diag: list[float], config: SGDHGTDTAConfig) -> dict[str, float]:
    regression = huber_loss(outputs.affinity, labels)
    contrastive = info_nce_loss(outputs.contrastive_projection)
    consistency = graph_consistency_loss(outputs.dynamic_scores, known_mask_diag)
    calibration = binary_calibration_loss(outputs.uncertainty, labels)
    total = (
        config.regression_weight * regression
        + config.contrastive_weight * contrastive
        + config.graph_consistency_weight * consistency
        + config.uncertainty_weight * calibration
    )
    return {
        "total": total,
        "regression": regression,
        "contrastive": contrastive,
        "graph_consistency": consistency,
        "uncertainty": calibration,
    }
