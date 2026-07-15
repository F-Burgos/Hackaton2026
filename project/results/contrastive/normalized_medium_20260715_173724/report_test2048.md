# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/normalized_medium_20260715_173724`
- export dir: `project/results/contrastive/normalized_medium_20260715_173724/export_test2048`
- checkpoint: `project/results/contrastive/normalized_medium_20260715_173724/best.pt`
- split: `test`
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

- i2s_recall@1: `0.000977`
- i2s_recall@5: `0.012207`
- i2s_recall@10: `0.023438`
- s2i_recall@1: `0.003418`
- s2i_recall@5: `0.011230`
- s2i_recall@10: `0.022461`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.032328`
- spectrum mean std: `0.038493`
- pair cosine mean: `0.554218`
- pair cosine std: `0.084436`
- pair distance p50: `0.447235`
- pair distance p95: `0.578350`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
