# LiSA on DrugOOD IC50

This adapter applies the official graph-level LiSA joint-generator objective to the cached DrugOOD IC50 PyG splits. It follows the repository-wide 4-layer GIN and training protocol.

The documented LiSA `loss penalty weight` is mapped to the distribution-risk variance coefficient (`--loss-penalty-weight`). The official graph-level defaults are retained for the independent KL coefficient (`0.1`), three joint generators, and 20 generator inner steps.

Use `train_ic50.py` for one run and `sweep_ic50.py` for `{1, 0.1, 0.01, 0.001}`.
