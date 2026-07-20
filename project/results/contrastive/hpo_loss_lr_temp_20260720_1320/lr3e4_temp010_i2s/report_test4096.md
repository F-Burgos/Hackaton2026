# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_i2s`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_i2s/export_test4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr3e4_temp010_i2s/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000291`
- contrastive_loss: `image_to_spectrum_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.514445`
- train_grad_norm: `18.046734`
- val_loss: `4.327557`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.003906`
- val_i2s_recall@10: `0.008789`
- val_i2s_median_rank: `1236.000000`
- val_i2s_mrr: `0.005575`
- val_s2i_recall@1: `0.000000`
- val_s2i_recall@5: `0.003174`
- val_s2i_recall@10: `0.006592`
- val_s2i_median_rank: `1189.000000`
- val_s2i_mrr: `0.004459`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000299`
- best_contrastive_loss: `image_to_spectrum_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.342494`
- best_train_grad_norm: `4.588710`
- best_val_loss: `3.114659`
- best_val_i2s_recall@1: `0.001221`
- best_val_i2s_recall@5: `0.005371`
- best_val_i2s_recall@10: `0.010254`
- best_val_i2s_median_rank: `964.000000`
- best_val_i2s_mrr: `0.006785`
- best_val_s2i_recall@1: `0.001221`
- best_val_s2i_recall@5: `0.005127`
- best_val_s2i_recall@10: `0.008301`
- best_val_s2i_median_rank: `1108.000000`
- best_val_s2i_mrr: `0.006059`

## Export Retrieval

- i2s_recall@1: `0.000732`
- i2s_recall@5: `0.004639`
- i2s_recall@10: `0.010498`
- i2s_median_rank: `919.000000`
- i2s_mrr: `0.006366`
- i2s_rank_percentile_median: `0.775824`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.003662`
- s2i_recall@10: `0.007568`
- s2i_median_rank: `1074.000000`
- s2i_mrr: `0.005767`
- s2i_rank_percentile_median: `0.737241`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.048230`
- spectrum mean std: `0.045767`
- positive cosine mean: `0.570962`
- negative cosine mean: `0.420331`
- positive-negative margin: `0.150631`
- pair cosine mean: `0.570962`
- pair cosine std: `0.110975`
- pair distance p50: `0.422069`
- pair distance p95: `0.630881`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
