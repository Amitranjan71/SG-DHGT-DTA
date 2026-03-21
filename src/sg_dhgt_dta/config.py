from dataclasses import dataclass


@dataclass(slots=True)
class SGDHGTDTAConfig:
    atom_input_dim: int = 32
    bond_input_dim: int = 8
    motif_input_dim: int = 16
    residue_input_dim: int = 48
    pocket_input_dim: int = 16
    hidden_dim: int = 64
    pair_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.1
    top_k_unlabeled: int = 6
    confidence_threshold: float = 0.55
    diversity_temperature: float = 0.7
    graph_consistency_weight: float = 0.2
    contrastive_weight: float = 0.1
    regression_weight: float = 1.0
    ranking_weight: float = 0.0
    uncertainty_weight: float = 0.05
