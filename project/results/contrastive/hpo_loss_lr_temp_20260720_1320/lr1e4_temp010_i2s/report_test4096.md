# Contrastive Run Report

## Inputs

- run dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_i2s`
- export dir: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_i2s/export_test4096`
- checkpoint: `project/results/contrastive/hpo_loss_lr_temp_20260720_1320/lr1e4_temp010_i2s/best.pt`
- split: `test`
- objects exported: `4096`

## Training Summary

### Final Epoch

- epoch: `11`
- learning_rate: `0.000097`
- contrastive_loss: `image_to_spectrum_info_nce`
- temperature: `0.100000`
- stopped_early: `True`
- n_train: `16384`
- n_val: `4096`
- train_contrastive_accumulation_steps: `4.000000`
- train_effective_contrastive_batch_size: `128.000000`
- train_optimizer_steps: `128.000000`
- train_loss: `3.937112`
- train_grad_norm: `15.659531`
- val_loss: `3.496449`
- val_i2s_recall@1: `0.000977`
- val_i2s_recall@5: `0.005127`
- val_i2s_recall@10: `0.010254`
- val_i2s_median_rank: `1058.000000`
- val_i2s_mrr: `0.006330`
- val_s2i_recall@1: `0.000244`
- val_s2i_recall@5: `0.003662`
- val_s2i_recall@10: `0.007812`
- val_s2i_median_rank: `1145.000000`
- val_s2i_mrr: `0.004902`

### Best Validation Checkpoint

- best_epoch: `5`
- best_learning_rate: `0.000099`
- best_contrastive_loss: `image_to_spectrum_info_nce`
- best_temperature: `0.100000`
- best_train_contrastive_accumulation_steps: `4.000000`
- best_train_effective_contrastive_batch_size: `128.000000`
- best_train_optimizer_steps: `128.000000`
- best_train_loss: `4.388905`
- best_train_grad_norm: `5.464561`
- best_val_loss: `3.101483`
- best_val_i2s_recall@1: `0.000977`
- best_val_i2s_recall@5: `0.003906`
- best_val_i2s_recall@10: `0.010742`
- best_val_i2s_median_rank: `986.000000`
- best_val_i2s_mrr: `0.006319`
- best_val_s2i_recall@1: `0.000244`
- best_val_s2i_recall@5: `0.002686`
- best_val_s2i_recall@10: `0.007080`
- best_val_s2i_median_rank: `1109.000000`
- best_val_s2i_mrr: `0.004629`

## Export Retrieval

- i2s_recall@1: `0.001465`
- i2s_recall@5: `0.006104`
- i2s_recall@10: `0.009521`
- i2s_median_rank: `988.000000`
- i2s_mrr: `0.006869`
- i2s_rank_percentile_median: `0.758730`
- s2i_recall@1: `0.000244`
- s2i_recall@5: `0.001953`
- s2i_recall@10: `0.004150`
- s2i_median_rank: `1109.000000`
- s2i_mrr: `0.004308`
- s2i_rank_percentile_median: `0.729182`

## Embedding Diagnostics

- image mean norm: `1.000000`
- spectrum mean norm: `1.000000`
- image mean std: `0.047368`
- spectrum mean std: `0.043922`
- positive cosine mean: `0.527006`
- negative cosine mean: `0.399408`
- positive-negative margin: `0.127597`
- pair cosine mean: `0.527006`
- pair cosine std: `0.093716`
- pair distance p50: `0.470941`
- pair distance p95: `0.630604`

## Export Prefix Counts

| Prefix | Objects |
|---|---:|
| `hst_3dhst` | 4096 |

## Notes

- This report describes representation/retrieval behavior only.
- It does not implement or evaluate the anomaly downstream.
- Low debug-run recall is expected for tiny one-epoch validation runs.
