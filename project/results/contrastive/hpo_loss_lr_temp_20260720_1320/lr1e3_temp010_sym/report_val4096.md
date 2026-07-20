# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_sym`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_sym/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_sym/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000969`
- contrastive_loss: `symmetric_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.641637`
- train_grad_norm: `6.678700`
- val_loss: `3.799410`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.003418`
- val_i2s_recall@10: `0.008057`
- val_i2s_median_rank: `1034.000000`
- val_i2s_mrr: `0.005591`
- val_s2i_recall@1: `0.000000`
- val_s2i_recall@5: `0.003662`
- val_s2i_recall@10: `0.009766`
- val_s2i_median_rank: `1036.000000`
- val_s2i_mrr: `0.005195`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000997`
- best_contrastive_loss: `symmetric_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.326861`
- best_train_grad_norm: `3.178727`
- best_val_loss: `3.091872`
- best_val_i2s_recall@1: `0.000488`
- best_val_i2s_recall@5: `0.005615`
- best_val_i2s_recall@10: `0.010498`
- best_val_i2s_median_rank: `942.000000`
- best_val_i2s_mrr: `0.006506`
- best_val_s2i_recall@1: `0.001221`
- best_val_s2i_recall@5: `0.005615`
- best_val_s2i_recall@10: `0.011230`
- best_val_s2i_median_rank: `929.000000`
- best_val_s2i_mrr: `0.006996`

## Export Retrieval

- i2s_recall@1: `0.000488`
- i2s_recall@5: `0.005615`
- i2s_recall@10: `0.010498`
- i2s_median_rank: `942.000000`
- i2s_mrr: `0.006506`
- i2s_rank_percentile_median: `0.770208`
- s2i_recall@1: `0.001221`
- s2i_recall@5: `0.005615`
- s2i_recall@10: `0.011230`
- s2i_median_rank: `929.000000`
- s2i_mrr: `0.006996`
- s2i_rank_percentile_median: `0.773138`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.041686`
- spectrum mean std: `0.047184`
- positive cosine mean: `0.507285`
- negative cosine mean: `0.351223`
- positive-negative margin: `0.156061`
- pair cosine mean: `0.507285`
- pair cosine std: `0.103118`
- pair distance p50: `0.478642`
- pair distance p95: `0.674509`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
