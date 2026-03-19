from __future__ import annotations

from dataclasses import dataclass

from .config import SGDHGTDTAConfig
from .tensor_ops import normalize_rows, random2, random3, randint2, symmetric_binary_adjacency, zeros2


@dataclass(slots=True)
class SGDHGTBatch:
    atom_features: list[list[list[float]]]
    atom_adj: list[list[list[float]]]
    atom_to_motif: list[list[int]]
    motif_features: list[list[list[float]]]
    residue_features: list[list[list[float]]]
    residue_adj: list[list[list[float]]]
    residue_to_pocket: list[list[int]]
    pocket_features: list[list[list[float]]]
    known_affinity_mask: list[list[float]]
    pair_labels: list[list[float]]
    drug_similarity: list[list[float]]
    protein_similarity: list[list[float]]


def build_demo_batch(config: SGDHGTDTAConfig, batch_size: int = 3) -> SGDHGTBatch:
    atoms, motifs, residues, pockets = 10, 4, 12, 3
    known_affinity_mask = zeros2(batch_size, batch_size)
    for i in range(batch_size):
        known_affinity_mask[i][i] = 1.0
    return SGDHGTBatch(
        atom_features=random3(batch_size, atoms, config.atom_input_dim, seed=11),
        atom_adj=symmetric_binary_adjacency(batch_size, atoms, seed=12),
        atom_to_motif=randint2(batch_size, atoms, 0, motifs - 1, seed=13),
        motif_features=random3(batch_size, motifs, config.motif_input_dim, seed=14),
        residue_features=random3(batch_size, residues, config.residue_input_dim, seed=15),
        residue_adj=symmetric_binary_adjacency(batch_size, residues, seed=16),
        residue_to_pocket=randint2(batch_size, residues, 0, pockets - 1, seed=17),
        pocket_features=random3(batch_size, pockets, config.pocket_input_dim, seed=18),
        known_affinity_mask=known_affinity_mask,
        pair_labels=random2(batch_size, 1, seed=19),
        drug_similarity=normalize_rows(random2(batch_size, batch_size, seed=20)),
        protein_similarity=normalize_rows(random2(batch_size, batch_size, seed=21)),
    )
