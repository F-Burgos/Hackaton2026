# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/long_simple_20260717_134339`
- export dir: `project/results/contrastive/long_simple_20260717_134339/export_val8192`
- checkpoint: `project/results/contrastive/long_simple_20260717_134339/best.pt`
- split: `val`
- objects exported: `8192`

## Training Summary

### Final Epoch

- epoch: `11`
- learning_rate: `0.000289`
- stopped_early: `True`
- n_train: `46213`
- n_val: `8192`
- train_loss: `2.088264`
- train_grad_norm: `8.690929`
- val_loss: `3.610612`
- val_i2s_recall@1: `0.000244`
- val_i2s_recall@5: `0.002808`
- val_i2s_recall@10: `0.005371`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.002686`
- val_s2i_recall@10: `0.005493`

### Best Validation Checkpoint

- best_epoch: `3`
- best_learning_rate: `0.000300`
- best_train_loss: `2.543592`
- best_train_grad_norm: `3.931031`
- best_val_loss: `2.973322`
- best_val_i2s_recall@1: `0.000732`
- best_val_i2s_recall@5: `0.003052`
- best_val_i2s_recall@10: `0.005493`
- best_val_s2i_recall@1: `0.000244`
- best_val_s2i_recall@5: `0.002441`
- best_val_s2i_recall@10: `0.005127`

## Export Retrieval

- i2s_recall@1: `0.000732`
- i2s_recall@5: `0.003052`
- i2s_recall@10: `0.005493`
- s2i_recall@1: `0.000244`
- s2i_recall@5: `0.002441`
- s2i_recall@10: `0.005127`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.032812`
- spectrum mean std: `0.041039`
- positive cosine mean: `0.553389`
- negative cosine mean: `0.444415`
- positive-negative margin: `0.108973`
- pair cosine mean: `0.553389`
- pair cosine std: `0.063676`
- pair distance p50: `0.446708`
- pair distance p95: `0.545970`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 8192 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
