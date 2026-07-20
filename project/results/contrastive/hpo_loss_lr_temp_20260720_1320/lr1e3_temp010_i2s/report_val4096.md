# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_i2s`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_i2s/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp010_i2s/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `8`
- learning_rate: `0.000981`
- contrastive_loss: `image_to_spectrum_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.956311`
- train_grad_norm: `4.695368`
- val_loss: `3.363994`
- val_i2s_recall@1: `0.000732`
- val_i2s_recall@5: `0.004883`
- val_i2s_recall@10: `0.008301`
- val_i2s_median_rank: `1106.000000`
- val_i2s_mrr: `0.005619`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.002441`
- val_s2i_recall@10: `0.006348`
- val_s2i_median_rank: `1141.000000`
- val_s2i_mrr: `0.005021`

### Best Validation Checkpoint

- best_epoch: `2`
- best_learning_rate: `0.001000`
- best_contrastive_loss: `image_to_spectrum_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.462466`
- best_train_grad_norm: `2.429110`
- best_val_loss: `3.081479`
- best_val_i2s_recall@1: `0.001953`
- best_val_i2s_recall@5: `0.006104`
- best_val_i2s_recall@10: `0.010254`
- best_val_i2s_median_rank: `979.000000`
- best_val_i2s_mrr: `0.007207`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.002686`
- best_val_s2i_recall@10: `0.005371`
- best_val_s2i_median_rank: `1412.000000`
- best_val_s2i_mrr: `0.004432`

## Export Retrieval

- i2s_recall@1: `0.001953`
- i2s_recall@5: `0.006104`
- i2s_recall@10: `0.010254`
- i2s_median_rank: `979.000000`
- i2s_mrr: `0.007207`
- i2s_rank_percentile_median: `0.760928`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.002686`
- s2i_recall@10: `0.005371`
- s2i_median_rank: `1412.000000`
- s2i_mrr: `0.004432`
- s2i_rank_percentile_median: `0.655189`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.045477`
- spectrum mean std: `0.034182`
- positive cosine mean: `0.618335`
- negative cosine mean: `0.505779`
- positive-negative margin: `0.112556`
- pair cosine mean: `0.618335`
- pair cosine std: `0.134953`
- pair distance p50: `0.382518`
- pair distance p95: `0.589314`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
