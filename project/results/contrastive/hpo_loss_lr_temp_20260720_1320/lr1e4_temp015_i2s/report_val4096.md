# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_i2s`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_i2s/export_val4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp015_i2s/best.pt`
- split: `val`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `10`
- learning_rate: `0.000097`
- contrastive_loss: `image_to_spectrum_info_nce`
- temperature: `0.150000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `4.055449`
- train_grad_norm: `11.013673`
- val_loss: `3.299264`
- val_i2s_recall@1: `0.001221`
- val_i2s_recall@5: `0.005859`
- val_i2s_recall@10: `0.010254`
- val_i2s_median_rank: `1066.000000`
- val_i2s_mrr: `0.006385`
- val_s2i_recall@1: `0.000732`
- val_s2i_recall@5: `0.004883`
- val_s2i_recall@10: `0.008057`
- val_s2i_median_rank: `1120.000000`
- val_s2i_mrr: `0.005178`

### Best Validation Checkpoint

- best_epoch: `4`
- best_learning_rate: `0.000100`
- best_contrastive_loss: `image_to_spectrum_info_nce`
- best_temperature: `0.150000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.440136`
- best_train_grad_norm: `4.564436`
- best_val_loss: `3.102402`
- best_val_i2s_recall@1: `0.001221`
- best_val_i2s_recall@5: `0.006592`
- best_val_i2s_recall@10: `0.014648`
- best_val_i2s_median_rank: `995.000000`
- best_val_i2s_mrr: `0.007563`
- best_val_s2i_recall@1: `0.000732`
- best_val_s2i_recall@5: `0.002930`
- best_val_s2i_recall@10: `0.007080`
- best_val_s2i_median_rank: `1128.000000`
- best_val_s2i_mrr: `0.005030`

## Export Retrieval

- i2s_recall@1: `0.001221`
- i2s_recall@5: `0.006592`
- i2s_recall@10: `0.014648`
- i2s_median_rank: `995.000000`
- i2s_mrr: `0.007563`
- i2s_rank_percentile_median: `0.757265`
- s2i_recall@1: `0.000732`
- s2i_recall@5: `0.002930`
- s2i_recall@10: `0.007080`
- s2i_median_rank: `1128.000000`
- s2i_mrr: `0.005030`
- s2i_rank_percentile_median: `0.724786`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.051757`
- spectrum mean std: `0.047048`
- positive cosine mean: `0.564739`
- negative cosine mean: `0.412983`
- positive-negative margin: `0.151756`
- pair cosine mean: `0.564739`
- pair cosine std: `0.116963`
- pair distance p50: `0.427462`
- pair distance p95: `0.639768`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
