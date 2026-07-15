# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/sweep_20260715_175816/lr1e4_temp007_b64`
- export dir: `project/results/contrastive/sweep_20260715_175816/lr1e4_temp007_b64/export_val2048`
- checkpoint: `project/results/contrastive/sweep_20260715_175816/lr1e4_temp007_b64/best.pt`
- split: `val`
- objects exported: `2048`

## Training Summary

### Final Epoch

- epoch: `8`
- n_train: `8192`
- n_val: `2048`
- train_loss: `2.987973`
- val_loss: `5.046295`
- val_i2s_recall@1: `0.001953`
- val_i2s_recall@5: `0.007324`
- val_i2s_recall@10: `0.013672`
- val_s2i_recall@1: `0.001465`
- val_s2i_recall@5: `0.007812`
- val_s2i_recall@10: `0.015137`

### Best Validation Checkpoint

- best_epoch: `3`
- best_train_loss: `3.775361`
- best_val_loss: `3.857881`
- best_val_i2s_recall@1: `0.000000`
- best_val_i2s_recall@5: `0.009766`
- best_val_i2s_recall@10: `0.018555`
- best_val_s2i_recall@1: `0.002930`
- best_val_s2i_recall@5: `0.009277`
- best_val_s2i_recall@10: `0.016602`

## Export Retrieval

- i2s_recall@1: `0.000000`
- i2s_recall@5: `0.009766`
- i2s_recall@10: `0.018555`
- s2i_recall@1: `0.002930`
- s2i_recall@5: `0.009277`
- s2i_recall@10: `0.016602`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.031519`
- spectrum mean std: `0.038617`
- pair cosine mean: `0.368462`
- pair cosine std: `0.063547`
- pair distance p50: `0.639296`
- pair distance p95: `0.719006`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
