# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/accum_medium_20260717_1515`
- export dir: `project/results/contrastive/accum_medium_20260717_1515/export_val4096`
- checkpoint: `project/results/contrastive/accum_medium_20260717_1515/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `8`
- learning_rate: `0.000295`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.680941`
- train_grad_norm: `13.720915`
- val_loss: `3.921543`
- val_i2s_recall@1: `0.001221`
- val_i2s_recall@5: `0.003174`
- val_i2s_recall@10: `0.007324`
- val_i2s_median_rank: `1091.000000`
- val_i2s_mrr: `0.005486`
- val_s2i_recall@1: `0.000488`
- val_s2i_recall@5: `0.003662`
- val_s2i_recall@10: `0.007080`
- val_s2i_median_rank: `1093.000000`
- val_s2i_mrr: `0.004999`

### Best Validation Checkpoint

- best_epoch: `2`
- best_learning_rate: `0.000300`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.475273`
- best_train_grad_norm: `3.414002`
- best_val_loss: `3.099186`
- best_val_i2s_recall@1: `0.001221`
- best_val_i2s_recall@5: `0.005615`
- best_val_i2s_recall@10: `0.009766`
- best_val_i2s_median_rank: `973.000000`
- best_val_i2s_mrr: `0.006885`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.003418`
- best_val_s2i_recall@10: `0.007568`
- best_val_s2i_median_rank: `980.000000`
- best_val_s2i_mrr: `0.005606`

## Export Retrieval

- i2s_recall@1: `0.001221`
- i2s_recall@5: `0.005615`
- i2s_recall@10: `0.009766`
- i2s_median_rank: `973.000000`
- i2s_mrr: `0.006885`
- i2s_rank_percentile_median: `0.762637`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.003418`
- s2i_recall@10: `0.007568`
- s2i_median_rank: `980.000000`
- s2i_mrr: `0.005606`
- s2i_rank_percentile_median: `0.760684`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.031265`
- spectrum mean std: `0.038787`
- positive cosine mean: `0.403339`
- negative cosine mean: `0.315249`
- positive-negative margin: `0.088090`
- pair cosine mean: `0.403339`
- pair cosine std: `0.071718`
- pair distance p50: `0.597449`
- pair distance p95: `0.702653`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
