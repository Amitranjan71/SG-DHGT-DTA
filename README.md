# SG-DHGT-DTA

A modular pure-Python reference scaffold for **Structure-Grounded Dynamic Hierarchical Graph Transformer for Drug–Target Affinity (SG-DHGT-DTA)**.

## Included components

- Hierarchical drug encoder with atom-level message passing and motif pooling.
- Structure-grounded protein encoder with residue-level updates and pocket pooling.
- Cross-scale interaction stack with atom-residue and motif-pocket attention.
- Dynamic affinity graph refiner with uncertainty-aware unlabeled edge expansion.
- Similarity fusion branch for cold-start drugs and proteins.
- Multi-objective training utilities for regression, contrastive learning, and graph consistency.

## Quick start

```python
from sg_dhgt_dta import SGDHGTDTAConfig, SGDHGTDTA, build_demo_batch

config = SGDHGTDTAConfig()
model = SGDHGTDTA(config)
batch = build_demo_batch(config)
outputs = model(batch)
print(len(outputs.affinity), len(outputs.affinity[0]))
```
