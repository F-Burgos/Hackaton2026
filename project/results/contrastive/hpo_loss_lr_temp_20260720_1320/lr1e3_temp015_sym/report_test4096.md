# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_sym`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_sym/export_test4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e3_temp015_sym/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000969`
- contrastive_loss: `symmetric_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.706454`
- train_grad_norm: `6.153680`
- val_loss: `3.626689`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.004150`
- val_i2s_recall@10: `0.009277`
- val_i2s_median_rank: `1016.000000`
- val_i2s_mrr: `0.006133`
- val_s2i_recall@1: `0.000244`
- val_s2i_recall@5: `0.002930`
- val_s2i_recall@10: `0.006836`
- val_s2i_median_rank: `1025.000000`
- val_s2i_mrr: `0.005349`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000997`
- best_contrastive_loss: `symmetric_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.332421`
- best_train_grad_norm: `3.547243`
- best_val_loss: `3.055827`
- best_val_i2s_recall@1: `0.001221`
- best_val_i2s_recall@5: `0.006104`
- best_val_i2s_recall@10: `0.011963`
- best_val_i2s_median_rank: `891.000000`
- best_val_i2s_mrr: `0.007291`
- best_val_s2i_recall@1: `0.001465`
- best_val_s2i_recall@5: `0.005615`
- best_val_s2i_recall@10: `0.011719`
- best_val_s2i_median_rank: `940.000000`
- best_val_s2i_mrr: `0.007353`

## Export Retrieval

- i2s_recall@1: `0.001221`
- i2s_recall@5: `0.004395`
- i2s_recall@10: `0.008545`
- i2s_median_rank: `893.000000`
- i2s_mrr: `0.006728`
- i2s_rank_percentile_median: `0.781929`
- s2i_recall@1: `0.001953`
- s2i_recall@5: `0.005859`
- s2i_recall@10: `0.012207`
- s2i_median_rank: `909.000000`
- s2i_mrr: `0.007784`
- s2i_rank_percentile_median: `0.778022`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.047847`
- spectrum mean std: `0.054534`
- positive cosine mean: `0.481203`
- negative cosine mean: `0.272426`
- positive-negative margin: `0.208777`
- pair cosine mean: `0.481203`
- pair cosine std: `0.143777`
- pair distance p50: `0.510866`
- pair distance p95: `0.757499`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
