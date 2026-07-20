# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_s2i`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_s2i/export_test4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_s2i/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000969`
- contrastive_loss: `spectrum_to_image_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.658424`
- train_grad_norm: `7.393826`
- val_loss: `3.667901`
- val_i2s_recall@1: `0.000488`
- val_i2s_recall@5: `0.005371`
- val_i2s_recall@10: `0.009521`
- val_i2s_median_rank: `1034.000000`
- val_i2s_mrr: `0.005701`
- val_s2i_recall@1: `0.000977`
- val_s2i_recall@5: `0.004639`
- val_s2i_recall@10: `0.011230`
- val_s2i_median_rank: `1048.000000`
- val_s2i_mrr: `0.006465`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000997`
- best_contrastive_loss: `spectrum_to_image_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.341402`
- best_train_grad_norm: `4.778952`
- best_val_loss: `3.063707`
- best_val_i2s_recall@1: `0.000488`
- best_val_i2s_recall@5: `0.002441`
- best_val_i2s_recall@10: `0.004639`
- best_val_i2s_median_rank: `1075.000000`
- best_val_i2s_mrr: `0.004689`
- best_val_s2i_recall@1: `0.000488`
- best_val_s2i_recall@5: `0.003906`
- best_val_s2i_recall@10: `0.009766`
- best_val_s2i_median_rank: `945.000000`
- best_val_s2i_mrr: `0.005786`

## Export Retrieval

- i2s_recall@1: `0.000488`
- i2s_recall@5: `0.002930`
- i2s_recall@10: `0.005615`
- i2s_median_rank: `1086.000000`
- i2s_mrr: `0.004967`
- i2s_rank_percentile_median: `0.734799`
- s2i_recall@1: `0.001221`
- s2i_recall@5: `0.005127`
- s2i_recall@10: `0.010986`
- s2i_median_rank: `930.000000`
- s2i_mrr: `0.007214`
- s2i_rank_percentile_median: `0.773138`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.046981`
- spectrum mean std: `0.059513`
- positive cosine mean: `0.573909`
- negative cosine mean: `0.344418`
- positive-negative margin: `0.229491`
- pair cosine mean: `0.573909`
- pair cosine std: `0.237101`
- pair distance p50: `0.363597`
- pair distance p95: `0.900605`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
