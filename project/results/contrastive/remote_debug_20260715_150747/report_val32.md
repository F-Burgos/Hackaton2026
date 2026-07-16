# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/remote_debug_20260715_150747`
- export dir: `project/results/contrastive/remote_debug_20260715_150747/export_val32`
- checkpoint: `project/results/contrastive/remote_debug_20260715_150747/best.pt`
- split: `val`
- objects exported: `32`

## Training Summary

- epoch: `1`
- n_train: `64`
- n_val: `32`
- train_loss: `2.096475`
- val_loss: `2.085508`
- val_i2s_recall@1: `0.031250`
- val_i2s_recall@5: `0.218750`
- val_i2s_recall@10: `0.343750`
- val_s2i_recall@1: `0.031250`
- val_s2i_recall@5: `0.156250`
- val_s2i_recall@10: `0.343750`

## Export Retrieval

- i2s_recall@1: `0.031250`
- i2s_recall@5: `0.218750`
- i2s_recall@10: `0.343750`
- s2i_recall@1: `0.031250`
- s2i_recall@5: `0.156250`
- s2i_recall@10: `0.343750`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.011621`
- spectrum mean std: `0.001829`
- pair cosine mean: `-0.034397`
- pair cosine std: `0.015549`
- pair distance p50: `1.039301`
- pair distance p95: `1.052059`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 32 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
