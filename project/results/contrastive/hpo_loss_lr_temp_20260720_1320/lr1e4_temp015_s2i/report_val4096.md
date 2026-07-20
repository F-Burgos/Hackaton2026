# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_s2i`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_s2i/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_s2i/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000097`
- contrastive_loss: `spectrum_to_image_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `4.019341`
- train_grad_norm: `9.973981`
- val_loss: `3.347132`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.004639`
- val_i2s_recall@10: `0.008301`
- val_i2s_median_rank: `1061.000000`
- val_i2s_mrr: `0.005912`
- val_s2i_recall@1: `0.001465`
- val_s2i_recall@5: `0.005615`
- val_s2i_recall@10: `0.010742`
- val_s2i_median_rank: `1075.000000`
- val_s2i_mrr: `0.006538`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000100`
- best_contrastive_loss: `spectrum_to_image_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.422723`
- best_train_grad_norm: `4.673829`
- best_val_loss: `3.099953`
- best_val_i2s_recall@1: `0.000732`
- best_val_i2s_recall@5: `0.003662`
- best_val_i2s_recall@10: `0.007568`
- best_val_i2s_median_rank: `1168.000000`
- best_val_i2s_mrr: `0.004894`
- best_val_s2i_recall@1: `0.000244`
- best_val_s2i_recall@5: `0.002441`
- best_val_s2i_recall@10: `0.006592`
- best_val_s2i_median_rank: `1020.000000`
- best_val_s2i_mrr: `0.004950`

## Export Retrieval

- i2s_recall@1: `0.000732`
- i2s_recall@5: `0.003662`
- i2s_recall@10: `0.007568`
- i2s_median_rank: `1168.000000`
- i2s_mrr: `0.004894`
- i2s_rank_percentile_median: `0.715018`
- s2i_recall@1: `0.000244`
- s2i_recall@5: `0.002441`
- s2i_recall@10: `0.006592`
- s2i_median_rank: `1020.000000`
- s2i_mrr: `0.004950`
- s2i_rank_percentile_median: `0.751160`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.048035`
- spectrum mean std: `0.062667`
- positive cosine mean: `0.572922`
- negative cosine mean: `0.394462`
- positive-negative margin: `0.178460`
- pair cosine mean: `0.572922`
- pair cosine std: `0.252277`
- pair distance p50: `0.358166`
- pair distance p95: `0.888859`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
