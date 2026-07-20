# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_s2i`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_s2i/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_s2i/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000291`
- contrastive_loss: `spectrum_to_image_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.439820`
- train_grad_norm: `12.913346`
- val_loss: `3.917237`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.005127`
- val_i2s_recall@10: `0.008301`
- val_i2s_median_rank: `1134.000000`
- val_i2s_mrr: `0.005444`
- val_s2i_recall@1: `0.001953`
- val_s2i_recall@5: `0.005127`
- val_s2i_recall@10: `0.009766`
- val_s2i_median_rank: `1139.000000`
- val_s2i_mrr: `0.006772`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000299`
- best_contrastive_loss: `spectrum_to_image_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.324227`
- best_train_grad_norm: `4.655573`
- best_val_loss: `3.110005`
- best_val_i2s_recall@1: `0.000244`
- best_val_i2s_recall@5: `0.003906`
- best_val_i2s_recall@10: `0.005859`
- best_val_i2s_median_rank: `1164.000000`
- best_val_i2s_mrr: `0.004304`
- best_val_s2i_recall@1: `0.001221`
- best_val_s2i_recall@5: `0.003906`
- best_val_s2i_recall@10: `0.008789`
- best_val_s2i_median_rank: `995.000000`
- best_val_s2i_mrr: `0.006067`

## Export Retrieval

- i2s_recall@1: `0.000244`
- i2s_recall@5: `0.003906`
- i2s_recall@10: `0.005859`
- i2s_median_rank: `1164.000000`
- i2s_mrr: `0.004304`
- i2s_rank_percentile_median: `0.715995`
- s2i_recall@1: `0.001221`
- s2i_recall@5: `0.003906`
- s2i_recall@10: `0.008789`
- s2i_median_rank: `995.000000`
- s2i_mrr: `0.006067`
- s2i_rank_percentile_median: `0.757265`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.048899`
- spectrum mean std: `0.064606`
- positive cosine mean: `0.549879`
- negative cosine mean: `0.328795`
- positive-negative margin: `0.221084`
- pair cosine mean: `0.549879`
- pair cosine std: `0.270701`
- pair distance p50: `0.380012`
- pair distance p95: `0.957647`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
