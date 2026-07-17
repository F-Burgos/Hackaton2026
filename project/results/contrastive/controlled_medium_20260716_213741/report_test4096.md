# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/controlled_medium_20260716_213741`
- export dir: `project/results/contrastive/controlled_medium_20260716_213741/export_test4096`
- checkpoint: `project/results/contrastive/controlled_medium_20260716_213741/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `7`
- learning_rate: `0.000240`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_loss: `2.415702`
- train_grad_norm: `14.261627`
- val_loss: `3.729737`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.004150`
- val_i2s_recall@10: `0.008301`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.003662`
- val_s2i_recall@10: `0.006348`

### Best Validation Checkpoint

- best_epoch: `3`
- best_learning_rate: `0.000293`
- best_train_loss: `2.990470`
- best_train_grad_norm: `5.971214`
- best_val_loss: `3.116941`
- best_val_i2s_recall@1: `0.000244`
- best_val_i2s_recall@5: `0.004395`
- best_val_i2s_recall@10: `0.008057`
- best_val_s2i_recall@1: `0.000488`
- best_val_s2i_recall@5: `0.003906`
- best_val_s2i_recall@10: `0.006836`

## Export Retrieval

- i2s_recall@1: `0.000977`
- i2s_recall@5: `0.004395`
- i2s_recall@10: `0.009033`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.004395`
- s2i_recall@10: `0.010010`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.036228`
- spectrum mean std: `0.041784`
- positive cosine mean: `0.460088`
- negative cosine mean: `0.341108`
- positive-negative margin: `0.118980`
- pair cosine mean: `0.460088`
- pair cosine std: `0.078959`
- pair distance p50: `0.536665`
- pair distance p95: `0.670220`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
