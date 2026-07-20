# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_sym`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_sym/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_sym/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000291`
- contrastive_loss: `symmetric_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.398353`
- train_grad_norm: `17.162661`
- val_loss: `4.567915`
- val_i2s_recall@1: `0.000244`
- val_i2s_recall@5: `0.005127`
- val_i2s_recall@10: `0.007080`
- val_i2s_median_rank: `1223.000000`
- val_i2s_mrr: `0.005246`
- val_s2i_recall@1: `0.000488`
- val_s2i_recall@5: `0.003662`
- val_s2i_recall@10: `0.009766`
- val_s2i_median_rank: `1174.000000`
- val_s2i_mrr: `0.005188`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000299`
- best_contrastive_loss: `symmetric_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.329517`
- best_train_grad_norm: `4.601292`
- best_val_loss: `3.111043`
- best_val_i2s_recall@1: `0.000977`
- best_val_i2s_recall@5: `0.004639`
- best_val_i2s_recall@10: `0.008789`
- best_val_i2s_median_rank: `985.000000`
- best_val_i2s_mrr: `0.006330`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.003174`
- best_val_s2i_recall@10: `0.008301`
- best_val_s2i_median_rank: `1009.000000`
- best_val_s2i_mrr: `0.005658`

## Export Retrieval

- i2s_recall@1: `0.000977`
- i2s_recall@5: `0.004639`
- i2s_recall@10: `0.008789`
- i2s_median_rank: `985.000000`
- i2s_mrr: `0.006330`
- i2s_rank_percentile_median: `0.759707`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.003174`
- s2i_recall@10: `0.008301`
- s2i_median_rank: `1009.000000`
- s2i_mrr: `0.005658`
- s2i_rank_percentile_median: `0.753846`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.043342`
- spectrum mean std: `0.048630`
- positive cosine mean: `0.505483`
- negative cosine mean: `0.356453`
- positive-negative margin: `0.149029`
- pair cosine mean: `0.505483`
- pair cosine std: `0.103932`
- pair distance p50: `0.486010`
- pair distance p95: `0.666822`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
