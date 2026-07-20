# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_sym`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_sym/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_sym/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000291`
- contrastive_loss: `symmetric_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.477874`
- train_grad_norm: `13.390786`
- val_loss: `4.116970`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.003174`
- val_i2s_recall@10: `0.007324`
- val_i2s_median_rank: `1124.000000`
- val_i2s_mrr: `0.005072`
- val_s2i_recall@1: `0.001221`
- val_s2i_recall@5: `0.005127`
- val_s2i_recall@10: `0.010254`
- val_s2i_median_rank: `1130.000000`
- val_s2i_mrr: `0.006275`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000299`
- best_contrastive_loss: `symmetric_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.336280`
- best_train_grad_norm: `4.268302`
- best_val_loss: `3.115570`
- best_val_i2s_recall@1: `0.001465`
- best_val_i2s_recall@5: `0.005371`
- best_val_i2s_recall@10: `0.009277`
- best_val_i2s_median_rank: `959.000000`
- best_val_i2s_mrr: `0.006786`
- best_val_s2i_recall@1: `0.000977`
- best_val_s2i_recall@5: `0.003662`
- best_val_s2i_recall@10: `0.007812`
- best_val_s2i_median_rank: `1010.000000`
- best_val_s2i_mrr: `0.005826`

## Export Retrieval

- i2s_recall@1: `0.001465`
- i2s_recall@5: `0.005371`
- i2s_recall@10: `0.009277`
- i2s_median_rank: `959.000000`
- i2s_mrr: `0.006786`
- i2s_rank_percentile_median: `0.765812`
- s2i_recall@1: `0.000977`
- s2i_recall@5: `0.003662`
- s2i_recall@10: `0.007812`
- s2i_median_rank: `1010.000000`
- s2i_mrr: `0.005826`
- s2i_rank_percentile_median: `0.753602`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.051559`
- spectrum mean std: `0.057053`
- positive cosine mean: `0.512615`
- negative cosine mean: `0.304208`
- positive-negative margin: `0.208407`
- pair cosine mean: `0.512615`
- pair cosine std: `0.154651`
- pair distance p50: `0.472046`
- pair distance p95: `0.753541`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
