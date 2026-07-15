# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp010_b64`
- export dir: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp010_b64/export_val2048`
- checkpoint: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp010_b64/best.pt`
- split: `val`
- objects exported: `2048`

## Training Summary

### Final Epoch

- epoch: `8`
- n_train: `8192`
- n_val: `2048`
- train_loss: `2.570661`
- val_loss: `5.691262`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.008301`
- val_i2s_recall@10: `0.019531`
- val_s2i_recall@1: `0.001953`
- val_s2i_recall@5: `0.009277`
- val_s2i_recall@10: `0.015625`

### Best Validation Checkpoint

- best_epoch: `2`
- best_train_loss: `3.798786`
- best_val_loss: `3.828716`
- best_val_i2s_recall@1: `0.001465`
- best_val_i2s_recall@5: `0.012207`
- best_val_i2s_recall@10: `0.022461`
- best_val_s2i_recall@1: `0.000977`
- best_val_s2i_recall@5: `0.010254`
- best_val_s2i_recall@10: `0.018066`

## Export Retrieval

- i2s_recall@1: `0.001465`
- i2s_recall@5: `0.012207`
- i2s_recall@10: `0.022461`
- s2i_recall@1: `0.000977`
- s2i_recall@5: `0.010254`
- s2i_recall@10: `0.018066`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.033507`
- spectrum mean std: `0.040666`
- pair cosine mean: `0.428123`
- pair cosine std: `0.089738`
- pair distance p50: `0.569994`
- pair distance p95: `0.713946`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
