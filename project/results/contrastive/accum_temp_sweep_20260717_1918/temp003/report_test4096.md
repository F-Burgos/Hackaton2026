# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/accum_temp_sweep_20260717_1918/temp003`
- export dir: `project/results/contrastive/accum_temp_sweep_20260717_1918/temp003/export_test4096`
- checkpoint: `project/results/contrastive/accum_temp_sweep_20260717_1918/temp003/best.pt`
- split: `test`
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
- train_loss: `3.614803`
- train_grad_norm: `15.975970`
- val_loss: `4.019250`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.004150`
- val_i2s_recall@10: `0.008545`
- val_i2s_median_rank: `1154.000000`
- val_i2s_mrr: `0.005637`
- val_s2i_recall@1: `0.001221`
- val_s2i_recall@5: `0.004395`
- val_s2i_recall@10: `0.007568`
- val_s2i_median_rank: `1109.000000`
- val_s2i_mrr: `0.005772`

### Best Validation Checkpoint

- best_epoch: `2`
- best_learning_rate: `0.000300`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.474585`
- best_train_grad_norm: `3.717297`
- best_val_loss: `3.097617`
- best_val_i2s_recall@1: `0.001221`
- best_val_i2s_recall@5: `0.007324`
- best_val_i2s_recall@10: `0.012695`
- best_val_i2s_median_rank: `969.000000`
- best_val_i2s_mrr: `0.007437`
- best_val_s2i_recall@1: `0.001221`
- best_val_s2i_recall@5: `0.002930`
- best_val_s2i_recall@10: `0.006348`
- best_val_s2i_median_rank: `986.000000`
- best_val_s2i_mrr: `0.005970`

## Export Retrieval

- i2s_recall@1: `0.000244`
- i2s_recall@5: `0.004150`
- i2s_recall@10: `0.010254`
- i2s_median_rank: `967.000000`
- i2s_mrr: `0.005736`
- i2s_rank_percentile_median: `0.764103`
- s2i_recall@1: `0.001953`
- s2i_recall@5: `0.005859`
- s2i_recall@10: `0.011475`
- s2i_median_rank: `949.000000`
- s2i_mrr: `0.007376`
- s2i_rank_percentile_median: `0.768498`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.023113`
- spectrum mean std: `0.026506`
- positive cosine mean: `0.341832`
- negative cosine mean: `0.303179`
- positive-negative margin: `0.038653`
- pair cosine mean: `0.341832`
- pair cosine std: `0.031074`
- pair distance p50: `0.659248`
- pair distance p95: `0.702055`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
