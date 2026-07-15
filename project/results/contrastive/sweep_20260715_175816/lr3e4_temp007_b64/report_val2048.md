# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp007_b64`
- export dir: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp007_b64/export_val2048`
- checkpoint: `project/results/contrastive/sweep_20260715_175816/lr3e4_temp007_b64/best.pt`
- split: `val`
- objects exported: `2048`

## Training Summary

### Final Epoch

- epoch: `8`
- n_train: `8192`
- n_val: `2048`
- train_loss: `2.495303`
- val_loss: `5.899497`
- val_i2s_recall@1: `0.002930`
- val_i2s_recall@5: `0.008789`
- val_i2s_recall@10: `0.017578`
- val_s2i_recall@1: `0.001953`
- val_s2i_recall@5: `0.010742`
- val_s2i_recall@10: `0.017090`

### Best Validation Checkpoint

- best_epoch: `2`
- best_train_loss: `3.797258`
- best_val_loss: `3.835076`
- best_val_i2s_recall@1: `0.003906`
- best_val_i2s_recall@5: `0.010742`
- best_val_i2s_recall@10: `0.019531`
- best_val_s2i_recall@1: `0.001465`
- best_val_s2i_recall@5: `0.008789`
- best_val_s2i_recall@10: `0.016113`

## Export Retrieval

- i2s_recall@1: `0.003906`
- i2s_recall@5: `0.010742`
- i2s_recall@10: `0.019531`
- s2i_recall@1: `0.001465`
- s2i_recall@5: `0.008789`
- s2i_recall@10: `0.016113`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.029168`
- spectrum mean std: `0.036690`
- pair cosine mean: `0.399980`
- pair cosine std: `0.068536`
- pair distance p50: `0.601550`
- pair distance p95: `0.702893`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 2048 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
