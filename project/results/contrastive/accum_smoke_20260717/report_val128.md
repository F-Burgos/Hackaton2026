# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/accum_smoke_20260717`
- export dir: `project/results/contrastive/accum_smoke_20260717/export_val128`
- checkpoint: `project/results/contrastive/accum_smoke_20260717/best.pt`
- split: `val`
- objects exported: `128`

## Training Summary

### Final Epoch

- epoch: `2`
- learning_rate: `0.000150`
- stopped_early: `False`
- n_train: `256`
- n_val: `128`
- train_contrastive_accumulation_steps: `2.000000`
- train_effective_contrastive_batch_size: `32.000000`
- train_optimizer_steps: `8.000000`
- train_loss: `3.319562`
- train_grad_norm: `1.401416`
- val_loss: `2.743415`
- val_i2s_recall@1: `0.015625`
- val_i2s_recall@5: `0.054688`
- val_i2s_recall@10: `0.093750`
- val_i2s_median_rank: `50.000000`
- val_i2s_mrr: `0.059449`
- val_s2i_recall@1: `0.015625`
- val_s2i_recall@5: `0.054688`
- val_s2i_recall@10: `0.109375`
- val_s2i_median_rank: `58.000000`
- val_s2i_mrr: `0.053421`

### Best Validation Checkpoint

- best_epoch: `2`
- best_learning_rate: `0.000150`
- best_train_contrastive_accumulation_steps: `2.000000`
- best_train_effective_contrastive_batch_size: `32.000000`
- best_train_optimizer_steps: `8.000000`
- best_train_loss: `3.319562`
- best_train_grad_norm: `1.401416`
- best_val_loss: `2.743415`
- best_val_i2s_recall@1: `0.015625`
- best_val_i2s_recall@5: `0.054688`
- best_val_i2s_recall@10: `0.093750`
- best_val_i2s_median_rank: `50.000000`
- best_val_i2s_mrr: `0.059449`
- best_val_s2i_recall@1: `0.015625`
- best_val_s2i_recall@5: `0.054688`
- best_val_s2i_recall@10: `0.109375`
- best_val_s2i_median_rank: `58.000000`
- best_val_s2i_mrr: `0.053421`

## Export Retrieval

- i2s_recall@1: `0.015625`
- i2s_recall@5: `0.054688`
- i2s_recall@10: `0.093750`
- i2s_median_rank: `50.000000`
- i2s_mrr: `0.059449`
- i2s_rank_percentile_median: `0.606299`
- s2i_recall@1: `0.015625`
- s2i_recall@5: `0.054688`
- s2i_recall@10: `0.109375`
- s2i_median_rank: `58.000000`
- s2i_mrr: `0.053421`
- s2i_rank_percentile_median: `0.519685`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.015728`
- spectrum mean std: `0.010407`
- positive cosine mean: `0.033596`
- negative cosine mean: `0.027264`
- positive-negative margin: `0.006331`
- pair cosine mean: `0.033596`
- pair cosine std: `0.034654`
- pair distance p50: `0.976940`
- pair distance p95: `1.005098`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 128 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
