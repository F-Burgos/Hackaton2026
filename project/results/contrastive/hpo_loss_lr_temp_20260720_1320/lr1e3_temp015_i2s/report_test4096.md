# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_i2s`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_i2s/export_test4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_i2s/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000969`
- contrastive_loss: `image_to_spectrum_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.766684`
- train_grad_norm: `5.593321`
- val_loss: `3.509423`
- val_i2s_recall@1: `0.002197`
- val_i2s_recall@5: `0.005127`
- val_i2s_recall@10: `0.011475`
- val_i2s_median_rank: `985.000000`
- val_i2s_mrr: `0.007389`
- val_s2i_recall@1: `0.000977`
- val_s2i_recall@5: `0.004883`
- val_s2i_recall@10: `0.008545`
- val_s2i_median_rank: `1029.000000`
- val_s2i_mrr: `0.005881`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000997`
- best_contrastive_loss: `image_to_spectrum_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.333046`
- best_train_grad_norm: `3.117153`
- best_val_loss: `3.058553`
- best_val_i2s_recall@1: `0.000732`
- best_val_i2s_recall@5: `0.005127`
- best_val_i2s_recall@10: `0.009277`
- best_val_i2s_median_rank: `894.000000`
- best_val_i2s_mrr: `0.006270`
- best_val_s2i_recall@1: `0.000977`
- best_val_s2i_recall@5: `0.003174`
- best_val_s2i_recall@10: `0.007568`
- best_val_s2i_median_rank: `1026.000000`
- best_val_s2i_mrr: `0.005352`

## Export Retrieval

- i2s_recall@1: `0.001221`
- i2s_recall@5: `0.005615`
- i2s_recall@10: `0.011475`
- i2s_median_rank: `881.000000`
- i2s_mrr: `0.007420`
- i2s_rank_percentile_median: `0.785104`
- s2i_recall@1: `0.000000`
- s2i_recall@5: `0.004150`
- s2i_recall@10: `0.008301`
- s2i_median_rank: `998.000000`
- s2i_mrr: `0.005189`
- s2i_rank_percentile_median: `0.756044`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.057159`
- spectrum mean std: `0.049970`
- positive cosine mean: `0.650955`
- negative cosine mean: `0.427929`
- positive-negative margin: `0.223026`
- pair cosine mean: `0.650955`
- pair cosine std: `0.151829`
- pair distance p50: `0.323344`
- pair distance p95: `0.638092`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
