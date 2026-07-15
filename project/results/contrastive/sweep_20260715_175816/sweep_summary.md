# Contrastive Mini-Sweep Summary

Run dir: `project/results/contrastive/sweep_20260715_175816`

Scope:

- train samples: `8192`
- validation samples: `2048`
- epochs: `8`
- batch size: `64`
- normalized per-sample image/spectrum inputs
- validation export: `2048` objects from `best.pt`

| Config | LR | Temp | Best Epoch | Best Val Loss | i2s R@1 | i2s R@5 | i2s R@10 | s2i R@1 | s2i R@5 | s2i R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `lr1e4_temp007_b64` | 0.0001 | 0.07 | 3 | 3.857881 | 0.000000 | 0.009766 | 0.018555 | 0.002930 | 0.009277 | 0.016602 |
| `lr3e4_temp003_b64` | 0.0003 | 0.03 | 2 | 3.844588 | 0.001465 | 0.010254 | 0.020020 | 0.000977 | 0.006836 | 0.012695 |
| `lr3e4_temp007_b64` | 0.0003 | 0.07 | 2 | 3.835076 | 0.003906 | 0.010742 | 0.019531 | 0.001465 | 0.008789 | 0.016113 |
| `lr3e4_temp010_b64` | 0.0003 | 0.10 | 2 | 3.828716 | 0.001465 | 0.012207 | 0.022461 | 0.000977 | 0.010254 | 0.018066 |

Interpretation:

- Increasing batch size from `32` to `64` did not improve validation loss relative to the previous normalized medium run (`3.1530`).
- Within this sweep, `lr3e4_temp010_b64` has the lowest validation loss and best i2s R@10.
- `lr3e4_temp007_b64` has the best i2s R@1, but the absolute retrieval remains weak.
- The next improvement should likely come from model/training changes beyond this small LR/temperature sweep, not simply from this batch-size increase.
