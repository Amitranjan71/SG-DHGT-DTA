from __future__ import annotations

from dataclasses import dataclass

from .config import SGDHGTDTAConfig
from .data import SGDHGTBatch
from .modules import CrossAttentionBlock, DynamicAffinityGraph, GatedFusion, GraphMessagePassing, HierarchicalPool, SimilarityBranch
from .tensor_ops import concat2, linear3, mean_pool


@dataclass(slots=True)
class ModelOutputs:
    affinity: list[list[float]]
    contrastive_projection: list[list[float]]
    uncertainty: list[list[float]]
    dynamic_scores: list[float]
    dynamic_confidence: list[float]


class DrugEncoder:
    def __init__(self, config: SGDHGTDTAConfig) -> None:
        self.hidden_dim = config.hidden_dim
        self.local = GraphMessagePassing(config.hidden_dim)
        self.pool = HierarchicalPool()

    def __call__(self, atom_features, atom_adj, atom_to_motif, motif_features):
        atoms = self.local(linear3(atom_features, self.hidden_dim), atom_adj)
        motifs = self.pool(atoms, atom_to_motif, linear3(motif_features, self.hidden_dim))
        graph = mean_pool(motifs)
        return atoms, motifs, graph


class ProteinEncoder:
    def __init__(self, config: SGDHGTDTAConfig) -> None:
        self.hidden_dim = config.hidden_dim
        self.local = GraphMessagePassing(config.hidden_dim)
        self.pool = HierarchicalPool()

    def __call__(self, residue_features, residue_adj, residue_to_pocket, pocket_features):
        residues = self.local(linear3(residue_features, self.hidden_dim), residue_adj)
        pockets = self.pool(residues, residue_to_pocket, linear3(pocket_features, self.hidden_dim))
        graph = mean_pool(pockets)
        return residues, pockets, graph


class SGDHGTDTA:
    def __init__(self, config: SGDHGTDTAConfig) -> None:
        self.config = config
        self.drug_encoder = DrugEncoder(config)
        self.protein_encoder = ProteinEncoder(config)
        self.atom_residue = CrossAttentionBlock()
        self.motif_pocket = CrossAttentionBlock()
        self.cross_fusion = GatedFusion()
        self.dynamic_graph = DynamicAffinityGraph(config.top_k_unlabeled, config.confidence_threshold)
        self.similarity = SimilarityBranch()

    def __call__(self, batch: SGDHGTBatch) -> ModelOutputs:
        atoms, motifs, drug_graph = self.drug_encoder(batch.atom_features, batch.atom_adj, batch.atom_to_motif, batch.motif_features)
        residues, pockets, protein_graph = self.protein_encoder(batch.residue_features, batch.residue_adj, batch.residue_to_pocket, batch.pocket_features)
        atom_context = mean_pool(self.atom_residue(atoms, residues))
        pocket_context = mean_pool(self.motif_pocket(motifs, pockets))
        cross_scale = self.cross_fusion(atom_context, pocket_context)
        external_embedding, graph_state = self.dynamic_graph(drug_graph, protein_graph, batch.known_affinity_mask)
        fused_drug, fused_protein = self.similarity(drug_graph, protein_graph, batch.drug_similarity, batch.protein_similarity)
        final_pair = concat2(cross_scale, external_embedding, fused_drug, fused_protein)
        affinity = [[sum(row) / len(row)] for row in final_pair]
        uncertainty = [[abs(sum(row)) / len(row) / 2.0] for row in final_pair]
        contrastive_projection = [row[: self.config.pair_dim] for row in final_pair]
        return ModelOutputs(
            affinity=affinity,
            contrastive_projection=contrastive_projection,
            uncertainty=uncertainty,
            dynamic_scores=graph_state.scores,
            dynamic_confidence=graph_state.confidence,
        )
