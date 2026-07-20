# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_i2s`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_i2s/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp015_i2s/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000291`
- contrastive_loss: `image_to_spectrum_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.597251`
- train_grad_norm: `13.291564`
- val_loss: `3.825242`
- val_i2s_recall@1: `0.000732`
- val_i2s_recall@5: `0.004639`
- val_i2s_recall@10: `0.009521`
- val_i2s_median_rank: `1099.000000`
- val_i2s_mrr: `0.005789`
- val_s2i_recall@1: `0.001221`
- val_s2i_recall@5: `0.004150`
- val_s2i_recall@10: `0.008789`
- val_s2i_median_rank: `1090.000000`
- val_s2i_mrr: `0.006087`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000299`
- best_contrastive_loss: `image_to_spectrum_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.344040`
- best_train_grad_norm: `4.386152`
- best_val_loss: `3.107150`
- best_val_i2s_recall@1: `0.000977`
- best_val_i2s_recall@5: `0.004883`
- best_val_i2s_recall@10: `0.010742`
- best_val_i2s_median_rank: `952.000000`
- best_val_i2s_mrr: `0.006505`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.003662`
- best_val_s2i_recall@10: `0.008301`
- best_val_s2i_median_rank: `1074.000000`
- best_val_s2i_mrr: `0.005563`

## Export Retrieval

- i2s_recall@1: `0.000977`
- i2s_recall@5: `0.004883`
- i2s_recall@10: `0.010742`
- i2s_median_rank: `952.000000`
- i2s_mrr: `0.006505`
- i2s_rank_percentile_median: `0.767521`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.003662`
- s2i_recall@10: `0.008301`
- s2i_median_rank: `1074.000000`
- s2i_mrr: `0.005563`
- s2i_rank_percentile_median: `0.737729`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.057751`
- spectrum mean std: `0.051272`
- positive cosine mean: `0.624920`
- negative cosine mean: `0.416785`
- positive-negative margin: `0.208134`
- pair cosine mean: `0.624920`
- pair cosine std: `0.151229`
- pair distance p50: `0.355823`
- pair distance p95: `0.650975`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
