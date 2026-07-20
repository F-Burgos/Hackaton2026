# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_sym`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_sym/export_test4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_sym/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000097`
- contrastive_loss: `symmetric_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `4.036158`
- train_grad_norm: `10.167308`
- val_loss: `3.331918`
- val_i2s_recall@1: `0.000732`
- val_i2s_recall@5: `0.006104`
- val_i2s_recall@10: `0.010010`
- val_i2s_median_rank: `1078.000000`
- val_i2s_mrr: `0.006131`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.004395`
- val_s2i_recall@10: `0.008057`
- val_s2i_median_rank: `1092.000000`
- val_s2i_mrr: `0.005568`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000100`
- best_contrastive_loss: `symmetric_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.438343`
- best_train_grad_norm: `4.579034`
- best_val_loss: `3.106064`
- best_val_i2s_recall@1: `0.000977`
- best_val_i2s_recall@5: `0.006592`
- best_val_i2s_recall@10: `0.010498`
- best_val_i2s_median_rank: `990.000000`
- best_val_i2s_mrr: `0.006937`
- best_val_s2i_recall@1: `0.000488`
- best_val_s2i_recall@5: `0.003662`
- best_val_s2i_recall@10: `0.007812`
- best_val_s2i_median_rank: `1025.000000`
- best_val_s2i_mrr: `0.005405`

## Export Retrieval

- i2s_recall@1: `0.000488`
- i2s_recall@5: `0.002930`
- i2s_recall@10: `0.008301`
- i2s_median_rank: `1000.000000`
- i2s_mrr: `0.005563`
- i2s_rank_percentile_median: `0.755311`
- s2i_recall@1: `0.001221`
- s2i_recall@5: `0.003906`
- s2i_recall@10: `0.008057`
- s2i_median_rank: `1034.000000`
- s2i_mrr: `0.005983`
- s2i_rank_percentile_median: `0.747497`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.046482`
- spectrum mean std: `0.051660`
- positive cosine mean: `0.494363`
- negative cosine mean: `0.336278`
- positive-negative margin: `0.158086`
- pair cosine mean: `0.494363`
- pair cosine std: `0.130866`
- pair distance p50: `0.501664`
- pair distance p95: `0.727063`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
