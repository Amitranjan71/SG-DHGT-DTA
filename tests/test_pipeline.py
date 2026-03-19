import math

from sg_dhgt_dta import SGDHGTDTA, SGDHGTDTAConfig, build_demo_batch, compute_multi_objective_losses
from sg_dhgt_dta.tensor_ops import diag


def test_forward_pass_shapes() -> None:
    config = SGDHGTDTAConfig(hidden_dim=32, pair_dim=24)
    batch = build_demo_batch(config, batch_size=4)
    model = SGDHGTDTA(config)

    outputs = model(batch)

    assert len(outputs.affinity) == 4
    assert len(outputs.affinity[0]) == 1
    assert len(outputs.contrastive_projection) == 4
    assert len(outputs.contrastive_projection[0]) == 24
    assert len(outputs.uncertainty) == 4
    assert len(outputs.dynamic_scores) == 4
    assert len(outputs.dynamic_confidence) == 4


def test_multi_objective_losses_are_finite() -> None:
    config = SGDHGTDTAConfig()
    batch = build_demo_batch(config, batch_size=3)
    model = SGDHGTDTA(config)
    outputs = model(batch)

    losses = compute_multi_objective_losses(outputs, batch.pair_labels, diag(batch.known_affinity_mask), config)

    assert set(losses) == {"total", "regression", "contrastive", "graph_consistency", "uncertainty"}
    for value in losses.values():
        assert math.isfinite(value)
