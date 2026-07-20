# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_sym`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_sym/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_sym/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `11`
- learning_rate: `0.000097`
- contrastive_loss: `symmetric_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.924016`
- train_grad_norm: `13.449324`
- val_loss: `3.466029`
- val_i2s_recall@1: `0.001221`
- val_i2s_recall@5: `0.005371`
- val_i2s_recall@10: `0.009033`
- val_i2s_median_rank: `1061.000000`
- val_i2s_mrr: `0.006405`
- val_s2i_recall@1: `0.000488`
- val_s2i_recall@5: `0.004883`
- val_s2i_recall@10: `0.008545`
- val_s2i_median_rank: `1109.000000`
- val_s2i_mrr: `0.005721`

### Best Validation Checkpoint

- best_epoch: `5`
- best_learning_rate: `0.000099`
- best_contrastive_loss: `symmetric_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.388020`
- best_train_grad_norm: `5.252109`
- best_val_loss: `3.104024`
- best_val_i2s_recall@1: `0.000977`
- best_val_i2s_recall@5: `0.006104`
- best_val_i2s_recall@10: `0.009277`
- best_val_i2s_median_rank: `1016.000000`
- best_val_i2s_mrr: `0.006464`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.004883`
- best_val_s2i_recall@10: `0.008301`
- best_val_s2i_median_rank: `1004.000000`
- best_val_s2i_mrr: `0.005886`

## Export Retrieval

- i2s_recall@1: `0.000977`
- i2s_recall@5: `0.006104`
- i2s_recall@10: `0.009277`
- i2s_median_rank: `1016.000000`
- i2s_mrr: `0.006464`
- i2s_rank_percentile_median: `0.752137`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.004883`
- s2i_recall@10: `0.008301`
- s2i_median_rank: `1004.000000`
- s2i_mrr: `0.005886`
- s2i_rank_percentile_median: `0.754823`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.043401`
- spectrum mean std: `0.047733`
- positive cosine mean: `0.489614`
- negative cosine mean: `0.362593`
- positive-negative margin: `0.127021`
- pair cosine mean: `0.489614`
- pair cosine std: `0.091442`
- pair distance p50: `0.507255`
- pair distance p95: `0.656731`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
