from __future__ import annotations

import math
import random
from typing import Iterable


def shape3(data: list[list[list[float]]]) -> tuple[int, int, int]:
    return len(data), len(data[0]), len(data[0][0])


def shape2(data: list[list[float]]) -> tuple[int, int]:
    return len(data), len(data[0])


def zeros3(a: int, b: int, c: int) -> list[list[list[float]]]:
    return [[[0.0 for _ in range(c)] for _ in range(b)] for _ in range(a)]


def zeros2(a: int, b: int) -> list[list[float]]:
    return [[0.0 for _ in range(b)] for _ in range(a)]


def random3(a: int, b: int, c: int, seed: int) -> list[list[list[float]]]:
    rng = random.Random(seed)
    return [[[rng.uniform(-1.0, 1.0) for _ in range(c)] for _ in range(b)] for _ in range(a)]


def random2(a: int, b: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    return [[rng.uniform(-1.0, 1.0) for _ in range(b)] for _ in range(a)]


def randint2(a: int, b: int, low: int, high: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randint(low, high) for _ in range(b)] for _ in range(a)]


def symmetric_binary_adjacency(batch: int, nodes: int, seed: int) -> list[list[list[float]]]:
    rng = random.Random(seed)
    out = zeros3(batch, nodes, nodes)
    for b in range(batch):
        for i in range(nodes):
            for j in range(i, nodes):
                value = 1.0 if i == j or rng.random() > 0.5 else 0.0
                out[b][i][j] = value
                out[b][j][i] = value
    return out


def linear3(data: list[list[list[float]]], out_dim: int) -> list[list[list[float]]]:
    batch, items, in_dim = shape3(data)
    out = zeros3(batch, items, out_dim)
    for b in range(batch):
        for i in range(items):
            base = sum(data[b][i]) / max(in_dim, 1)
            for o in range(out_dim):
                out[b][i][o] = math.tanh(base + (o + 1) * 0.01)
    return out


def mean_pool(data: list[list[list[float]]]) -> list[list[float]]:
    batch, items, dim = shape3(data)
    out = zeros2(batch, dim)
    for b in range(batch):
        for d in range(dim):
            out[b][d] = sum(data[b][i][d] for i in range(items)) / max(items, 1)
    return out


def concat2(*arrays: list[list[float]]) -> list[list[float]]:
    batch = len(arrays[0])
    return [sum((arr[b] for arr in arrays), []) for b in range(batch)]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def normalize_rows(matrix: list[list[float]]) -> list[list[float]]:
    out = []
    for row in matrix:
        total = sum(math.exp(v) for v in row)
        out.append([math.exp(v) / total for v in row])
    return out


def matmul2(matrix: list[list[float]], embeddings: list[list[float]]) -> list[list[float]]:
    rows = len(matrix)
    cols = len(embeddings[0])
    inner = len(matrix[0])
    out = zeros2(rows, cols)
    for i in range(rows):
        for j in range(cols):
            out[i][j] = sum(matrix[i][k] * embeddings[k][j] for k in range(inner))
    return out


def l2_normalize_rows(data: list[list[float]]) -> list[list[float]]:
    out = []
    for row in data:
        norm = math.sqrt(sum(v * v for v in row)) or 1.0
        out.append([v / norm for v in row])
    return out


def diag(matrix: list[list[float]]) -> list[float]:
    return [matrix[i][i] for i in range(len(matrix))]


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)
