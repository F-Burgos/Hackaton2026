# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/normalized_medium_20260715_173724`
- export dir: `project/results/contrastive/normalized_medium_20260715_173724/export_val2048`
- checkpoint: `project/results/contrastive/normalized_medium_20260715_173724/best.pt`
- split: `val`
- objects exported: `2048`

## Training Summary

- epoch: `5`
- n_train: `8192`
- n_val: `2048`
- train_loss: `2.902746`
- val_loss: `3.153041`
- val_i2s_recall@1: `0.002441`
- val_i2s_recall@5: `0.007812`
- val_i2s_recall@10: `0.013672`
- val_s2i_recall@1: `0.000977`
- val_s2i_recall@5: `0.009277`
- val_s2i_recall@10: `0.021973`

## Export Retrieval

- i2s_recall@1: `0.002441`
- i2s_recall@5: `0.012695`
- i2s_recall@10: `0.023926`
- s2i_recall@1: `0.000488`
- s2i_recall@5: `0.005371`
- s2i_recall@10: `0.011230`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.032558`
- spectrum mean std: `0.038862`
- pair cosine mean: `0.553570`
- pair cosine std: `0.086059`
- pair distance p50: `0.445401`
- pair distance p95: `0.582171`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
