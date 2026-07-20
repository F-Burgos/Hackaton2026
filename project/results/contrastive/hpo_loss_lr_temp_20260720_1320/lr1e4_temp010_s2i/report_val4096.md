# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_s2i`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_s2i/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_s2i/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `11`
- learning_rate: `0.000097`
- contrastive_loss: `spectrum_to_image_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.907563`
- train_grad_norm: `12.054789`
- val_loss: `3.433628`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.004150`
- val_i2s_recall@10: `0.008057`
- val_i2s_median_rank: `1151.000000`
- val_i2s_mrr: `0.005378`
- val_s2i_recall@1: `0.001465`
- val_s2i_recall@5: `0.005859`
- val_s2i_recall@10: `0.009521`
- val_s2i_median_rank: `1093.000000`
- val_s2i_mrr: `0.006613`

### Best Validation Checkpoint

- best_epoch: `5`
- best_learning_rate: `0.000099`
- best_contrastive_loss: `spectrum_to_image_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.369738`
- best_train_grad_norm: `5.571376`
- best_val_loss: `3.098788`
- best_val_i2s_recall@1: `0.000244`
- best_val_i2s_recall@5: `0.000977`
- best_val_i2s_recall@10: `0.006104`
- best_val_i2s_median_rank: `1192.000000`
- best_val_i2s_mrr: `0.003776`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.002930`
- best_val_s2i_recall@10: `0.008057`
- best_val_s2i_median_rank: `1001.000000`
- best_val_s2i_mrr: `0.005686`

## Export Retrieval

- i2s_recall@1: `0.000244`
- i2s_recall@5: `0.000977`
- i2s_recall@10: `0.006104`
- i2s_median_rank: `1192.000000`
- i2s_mrr: `0.003776`
- i2s_rank_percentile_median: `0.708913`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.002930`
- s2i_recall@10: `0.008057`
- s2i_median_rank: `1001.000000`
- s2i_mrr: `0.005686`
- s2i_rank_percentile_median: `0.755556`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.044337`
- spectrum mean std: `0.059670`
- positive cosine mean: `0.562099`
- negative cosine mean: `0.418860`
- positive-negative margin: `0.143239`
- pair cosine mean: `0.562099`
- pair cosine std: `0.203504`
- pair distance p50: `0.391444`
- pair distance p95: `0.805708`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
