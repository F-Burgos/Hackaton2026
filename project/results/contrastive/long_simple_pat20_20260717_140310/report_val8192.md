# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/long_simple_pat20_20260717_140310`
- export dir: `project/results/contrastive/long_simple_pat20_20260717_140310/export_val8192`
- checkpoint: `project/results/contrastive/long_simple_pat20_20260717_140310/best.pt`
- split: `val`
- objects exported: `8192`

## Training Summary

### Final Epoch

- epoch: `23`
- learning_rate: `0.000299`
- stopped_early: `True`
- n_train: `46213`
- n_val: `8192`
- train_loss: `1.496612`
- train_grad_norm: `14.189085`
- val_loss: `4.473719`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.002930`
- val_i2s_recall@10: `0.006836`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.003540`
- val_s2i_recall@10: `0.006592`

### Best Validation Checkpoint

- best_epoch: `3`
- best_learning_rate: `0.000300`
- best_train_loss: `2.543665`
- best_train_grad_norm: `3.929289`
- best_val_loss: `2.973097`
- best_val_i2s_recall@1: `0.000854`
- best_val_i2s_recall@5: `0.002930`
- best_val_i2s_recall@10: `0.005615`
- best_val_s2i_recall@1: `0.000244`
- best_val_s2i_recall@5: `0.002441`
- best_val_s2i_recall@10: `0.005249`

## Export Retrieval

- i2s_recall@1: `0.000854`
- i2s_recall@5: `0.002930`
- i2s_recall@10: `0.005615`
- s2i_recall@1: `0.000244`
- s2i_recall@5: `0.002441`
- s2i_recall@10: `0.005249`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.032804`
- spectrum mean std: `0.041040`
- positive cosine mean: `0.553577`
- negative cosine mean: `0.444602`
- positive-negative margin: `0.108975`
- pair cosine mean: `0.553577`
- pair cosine std: `0.063695`
- pair distance p50: `0.446500`
- pair distance p95: `0.545894`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 8192 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
