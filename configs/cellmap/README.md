# CellMap protocol snapshots

`paper_dataset200/` is the intended leakage-free paper configuration. Its
`imagesTs` cases remain outside the training set.

The source Dataset200 `dataset.json` declared 215 training cases, while the raw
snapshot and every saved fold contain 202 cases. The paper copy corrects
`numTraining` to 202 and `scripts/verify_protocol.py` enforces that count.

`reference_dataset202/` is retained only to reconstruct the completed July 2026
run. Dataset202's metadata explicitly says Dataset200 `imagesTr + imagesTs` and
`labelsTr + labelsTs` were merged into training. Results from that checkpoint on
Dataset200 `imagesTs` are therefore not held-out results.

These JSON files contain metadata, plans, fingerprints, and split identifiers;
they do not redistribute image data.
