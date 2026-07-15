# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp003_b64`
- export dir: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp003_b64/export_val2048`
- checkpoint: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp003_b64/best.pt`
- split: `val`
- objects exported: `2048`

## Training Summary

### Final Epoch

- epoch: `8`
- n_train: `8192`
- n_val: `2048`
- train_loss: `2.371311`
- val_loss: `6.423723`
- val_i2s_recall@1: `0.002441`
- val_i2s_recall@5: `0.010742`
- val_i2s_recall@10: `0.020508`
- val_s2i_recall@1: `0.001465`
- val_s2i_recall@5: `0.003906`
- val_s2i_recall@10: `0.010254`

### Best Validation Checkpoint

- best_epoch: `2`
- best_train_loss: `3.804581`
- best_val_loss: `3.844588`
- best_val_i2s_recall@1: `0.001465`
- best_val_i2s_recall@5: `0.010254`
- best_val_i2s_recall@10: `0.020020`
- best_val_s2i_recall@1: `0.000977`
- best_val_s2i_recall@5: `0.006836`
- best_val_s2i_recall@10: `0.012695`

## Export Retrieval

- i2s_recall@1: `0.001465`
- i2s_recall@5: `0.010254`
- i2s_recall@10: `0.020020`
- s2i_recall@1: `0.000977`
- s2i_recall@5: `0.006836`
- s2i_recall@10: `0.012695`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.022449`
- spectrum mean std: `0.024828`
- pair cosine mean: `0.345685`
- pair cosine std: `0.030072`
- pair distance p50: `0.654944`
- pair distance p95: `0.698339`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
