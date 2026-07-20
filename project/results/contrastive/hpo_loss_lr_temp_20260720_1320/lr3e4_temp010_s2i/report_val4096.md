# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_s2i`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_s2i/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_s2i/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000291`
- contrastive_loss: `spectrum_to_image_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.353674`
- train_grad_norm: `15.100640`
- val_loss: `4.317669`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.004395`
- val_i2s_recall@10: `0.006836`
- val_i2s_median_rank: `1135.000000`
- val_i2s_mrr: `0.005294`
- val_s2i_recall@1: `0.000000`
- val_s2i_recall@5: `0.004150`
- val_s2i_recall@10: `0.009521`
- val_s2i_median_rank: `1154.000000`
- val_s2i_mrr: `0.005133`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000299`
- best_contrastive_loss: `spectrum_to_image_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.320082`
- best_train_grad_norm: `5.099603`
- best_val_loss: `3.124878`
- best_val_i2s_recall@1: `0.000244`
- best_val_i2s_recall@5: `0.003174`
- best_val_i2s_recall@10: `0.006592`
- best_val_i2s_median_rank: `1269.000000`
- best_val_i2s_mrr: `0.003986`
- best_val_s2i_recall@1: `0.000977`
- best_val_s2i_recall@5: `0.004395`
- best_val_s2i_recall@10: `0.009766`
- best_val_s2i_median_rank: `1014.000000`
- best_val_s2i_mrr: `0.006286`

## Export Retrieval

- i2s_recall@1: `0.000244`
- i2s_recall@5: `0.003174`
- i2s_recall@10: `0.006592`
- i2s_median_rank: `1269.000000`
- i2s_mrr: `0.003986`
- i2s_rank_percentile_median: `0.690354`
- s2i_recall@1: `0.000977`
- s2i_recall@5: `0.004395`
- s2i_recall@10: `0.009766`
- s2i_median_rank: `1014.000000`
- s2i_mrr: `0.006286`
- s2i_rank_percentile_median: `0.752381`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.041330`
- spectrum mean std: `0.058671`
- positive cosine mean: `0.565340`
- negative cosine mean: `0.405146`
- positive-negative margin: `0.160194`
- pair cosine mean: `0.565340`
- pair cosine std: `0.247632`
- pair distance p50: `0.369469`
- pair distance p95: `0.900484`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
