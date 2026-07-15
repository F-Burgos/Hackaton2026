# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/medium_20260715_161821`
- export dir: `project/results/contrastive/medium_20260715_161821/export_test2048`
- checkpoint: `project/results/contrastive/medium_20260715_161821/best.pt`
- split: `test`
- objects exported: `2048`

## Training Summary

- epoch: `5`
- n_train: `8192`
- n_val: `2048`
- train_loss: `3.553306`
- val_loss: `4.311142`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.008789`
- val_i2s_recall@10: `0.016602`
- val_s2i_recall@1: `0.000977`
- val_s2i_recall@5: `0.008301`
- val_s2i_recall@10: `0.016602`

## Export Retrieval

- i2s_recall@1: `0.001953`
- i2s_recall@5: `0.009277`
- i2s_recall@10: `0.017090`
- s2i_recall@1: `0.001465`
- s2i_recall@5: `0.009766`
- s2i_recall@10: `0.021484`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.031917`
- spectrum mean std: `0.033494`
- pair cosine mean: `0.507305`
- pair cosine std: `0.067288`
- pair distance p50: `0.486715`
- pair distance p95: `0.609406`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
