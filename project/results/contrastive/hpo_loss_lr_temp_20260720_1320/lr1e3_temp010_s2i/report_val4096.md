# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_s2i`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_s2i/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_s2i/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000969`
- contrastive_loss: `spectrum_to_image_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.582748`
- train_grad_norm: `7.598254`
- val_loss: `3.778894`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.003174`
- val_i2s_recall@10: `0.006836`
- val_i2s_median_rank: `1080.000000`
- val_i2s_mrr: `0.004938`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.005127`
- val_s2i_recall@10: `0.009521`
- val_s2i_median_rank: `1073.000000`
- val_s2i_mrr: `0.005687`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000997`
- best_contrastive_loss: `spectrum_to_image_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.336388`
- best_train_grad_norm: `4.846764`
- best_val_loss: `3.053741`
- best_val_i2s_recall@1: `0.000244`
- best_val_i2s_recall@5: `0.002686`
- best_val_i2s_recall@10: `0.007080`
- best_val_i2s_median_rank: `1202.000000`
- best_val_i2s_mrr: `0.004210`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.005859`
- best_val_s2i_recall@10: `0.010254`
- best_val_s2i_median_rank: `928.000000`
- best_val_s2i_mrr: `0.006822`

## Export Retrieval

- i2s_recall@1: `0.000244`
- i2s_recall@5: `0.002686`
- i2s_recall@10: `0.007080`
- i2s_median_rank: `1202.000000`
- i2s_mrr: `0.004210`
- i2s_rank_percentile_median: `0.706471`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.005859`
- s2i_recall@10: `0.010254`
- s2i_median_rank: `928.000000`
- s2i_mrr: `0.006822`
- s2i_rank_percentile_median: `0.773382`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.036064`
- spectrum mean std: `0.050678`
- positive cosine mean: `0.571727`
- negative cosine mean: `0.435946`
- positive-negative margin: `0.135780`
- pair cosine mean: `0.571726`
- pair cosine std: `0.198670`
- pair distance p50: `0.380139`
- pair distance p95: `0.798375`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
