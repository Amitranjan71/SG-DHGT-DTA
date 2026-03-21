from __future__ import annotations

from dataclasses import dataclass

from .tensor_ops import linear3, matmul2, mean, mean_pool, sigmoid, zeros2, zeros3


class GraphMessagePassing:
    def __init__(self, hidden_dim: int) -> None:
        self.hidden_dim = hidden_dim

    def __call__(self, features: list[list[list[float]]], adjacency: list[list[list[float]]]) -> list[list[list[float]]]:
        batch, nodes, _ = len(features), len(features[0]), len(features[0][0])
        out = zeros3(batch, nodes, self.hidden_dim)
        for b in range(batch):
            for i in range(nodes):
                neighbors = [j for j, edge in enumerate(adjacency[b][i]) if edge > 0.0]
                for d in range(self.hidden_dim):
                    out[b][i][d] = mean(features[b][j][d] for j in neighbors)
        return out


class HierarchicalPool:
    def __call__(self, lower: list[list[list[float]]], assignment: list[list[int]], supernode_features: list[list[list[float]]]) -> list[list[list[float]]]:
        batch, num_supernodes, hidden_dim = len(supernode_features), len(supernode_features[0]), len(supernode_features[0][0])
        pooled = zeros3(batch, num_supernodes, hidden_dim)
        for b in range(batch):
            for idx in range(num_supernodes):
                members = [i for i, target in enumerate(assignment[b]) if target == idx]
                if not members:
                    pooled[b][idx] = supernode_features[b][idx][:]
                    continue
                for d in range(hidden_dim):
                    pooled[b][idx][d] = mean(lower[b][i][d] for i in members) + supernode_features[b][idx][d]
        return pooled


class CrossAttentionBlock:
    def __call__(self, source: list[list[list[float]]], context: list[list[list[float]]]) -> list[list[list[float]]]:
        batch, source_len, hidden_dim = len(source), len(source[0]), len(source[0][0])
        out = zeros3(batch, source_len, hidden_dim)
        for b in range(batch):
            context_summary = mean_pool([context[b]])[0]
            for i in range(source_len):
                for d in range(hidden_dim):
                    out[b][i][d] = (source[b][i][d] + context_summary[d]) / 2.0
        return out


class GatedFusion:
    def __call__(self, left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
        out = zeros2(len(left), len(left[0]))
        for b in range(len(left)):
            gate = sigmoid(mean(left[b]) - mean(right[b]))
            for d in range(len(left[0])):
                out[b][d] = gate * left[b][d] + (1.0 - gate) * right[b][d]
        return out


@dataclass(slots=True)
class DynamicGraphState:
    adjacency: list[list[float]]
    scores: list[float]
    confidence: list[float]


class DynamicAffinityGraph:
    def __init__(self, top_k: int, confidence_threshold: float) -> None:
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

    def __call__(self, drug_graph: list[list[float]], protein_graph: list[list[float]], known_mask: list[list[float]]) -> tuple[list[list[float]], DynamicGraphState]:
        batch = len(drug_graph)
        pair_scores = [mean(drug_graph[i]) + mean(protein_graph[i]) for i in range(batch)]
        confidence = [sigmoid(score) for score in pair_scores]
        ranked = sorted(range(batch), key=lambda i: pair_scores[i] * confidence[i], reverse=True)
        adjacency = [row[:] for row in known_mask]
        for idx in ranked[: min(self.top_k, batch)]:
            if confidence[idx] >= self.confidence_threshold:
                adjacency[idx][idx] = 1.0
        external = zeros2(batch, len(protein_graph[0]))
        for i in range(batch):
            neighbors = [j for j, flag in enumerate(adjacency[i]) if flag > 0.0]
            for d in range(len(protein_graph[0])):
                external[i][d] = mean(protein_graph[j][d] for j in neighbors)
        return external, DynamicGraphState(adjacency=adjacency, scores=pair_scores, confidence=confidence)


class SimilarityBranch:
    def __call__(self, drug_graph: list[list[float]], protein_graph: list[list[float]], drug_similarity: list[list[float]], protein_similarity: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        sim_drug = matmul2(drug_similarity, drug_graph)
        sim_protein = matmul2(protein_similarity, protein_graph)
        fused_drug = zeros2(len(drug_graph), len(drug_graph[0]))
        fused_protein = zeros2(len(protein_graph), len(protein_graph[0]))
        for b in range(len(drug_graph)):
            drug_gate = sigmoid(mean(drug_graph[b]) - mean(sim_drug[b]))
            protein_gate = sigmoid(mean(protein_graph[b]) - mean(sim_protein[b]))
            for d in range(len(drug_graph[0])):
                fused_drug[b][d] = drug_gate * drug_graph[b][d] + (1.0 - drug_gate) * sim_drug[b][d]
                fused_protein[b][d] = protein_gate * protein_graph[b][d] + (1.0 - protein_gate) * sim_protein[b][d]
        return fused_drug, fused_protein
