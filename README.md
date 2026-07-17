# Hackaton2026

Multimodal representation learning for paired galaxy image stamps and spectra.

The current objective is to build a reliable image-spectrum contrastive pipeline and inspect the learned multimodal latent space. The anomaly-detection downstream is defined in the project reports, but is not implemented yet.

## Broader Goal

Beyond the immediate hackaton objective, this repository is also a working case study in how agents and language models can be integrated into scientific research. The project is meant to keep code, experiments, data reports, decisions, and progress traces in one reproducible workflow where human scientific judgment stays central and AI agents help with implementation, validation, documentation, and experiment management.

## Current Status

Implemented:

- lazy HDF5 data access;
- partition filtering against available HDF5 keys;
- safe multimodal datasets for paired image-spectrum objects;
- SSL and contrastive smoke tests;
- base contrastive training;
- retrieval metrics;
- embedding export from contrastive checkpoints.

Not implemented yet:

- long production training runs;
- HPO;
- final scientific reports;
- anomaly downstream.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the living status board.

## Data

Heavy data files are expected locally under:

```text
hackaton/
  images_reduced.h5
  spectra.h5
  dataset_metadata_reduced.json
  partitions/v1/ssl/
```

These files are intentionally ignored by Git.

Lightweight outputs under `project/results/` are versioned when they document a run:

- Markdown reports;
- JSON/JSONL metrics;
- CSV diagnostics;
- PNG figures.

Heavy or easily regenerated artifacts remain ignored, even when they live below `project/results/`:

- checkpoints (`*.pt`, `*.ckpt`, `*.pth`);
- exported embedding arrays (`*.npz`);
- raw logs.

Key dataset facts:

- paired image-spectrum objects: `64,365`;
- spectra total: `69,351`;
- spectrum-only objects: `4,986`;
- filtered test pairs: `6,586`;
- original partitions cover a larger universe and must be filtered against HDF5 keys.

Detailed data notes live in [data/reports/dataset_structure_report.md](data/reports/dataset_structure_report.md).

## Environment

The common environment follows what the remote GPU/HPC server can run:

- Python `3.10`;
- `uv`;
- repo-local `.venv`;
- `--system-site-packages` so the remote server can reuse system PyTorch/CUDA.

Set up:

```bash
bash project/scripts/sh/setup_env.sh
source .venv/bin/activate
```

Local machines without Torch for Python 3.10 can still run non-Torch tests. Full Torch/CUDA validation is expected on the remote GPU/HPC server.

## Common Commands

Inspect local/remote status:

```bash
bash project/scripts/sh/status.sh
bash project/scripts/sh/check_remote.sh
```

Validate data access:

```bash
bash project/scripts/sh/dry_run_data.sh --sample-size 2
```

Run tests:

```bash
pytest -q
```

Run contrastive training, small default config:

```bash
bash project/scripts/sh/run_contrastive.sh
```

Export embeddings from a checkpoint:

```bash
bash project/scripts/sh/export_contrastive_embeddings.sh \
  eval.checkpoint_path=project/results/contrastive/dry_run/best.pt \
  eval.split=val \
  eval.max_samples=128 \
  eval.output_dir=project/results/contrastive/dry_run_eval
```

More execution notes live in [pipeline_execution_guide.md](pipeline_execution_guide.md).

## Remote Workflow

Full experiments run on a private remote GPU/HPC server. The hostname and user are intentionally not versioned.

Rules:

- update remote code with `git pull`;
- do not edit code directly on the remote server;
- do not touch processes outside `~/Hackaton2026`;
- check GPU/process state before launching jobs.

## Reports

- [Dataset structure](data/reports/dataset_structure_report.md)
- [Anomaly research plan](data/reports/anomaly_research_plan.md)
- [Implementation roadmap](data/reports/implementation_roadmap.md)
- [Contrastive progress](project/reports/contrastive_progress_report.md)

`CLAUDE.md` is ignored and kept only as local reference. The versioned workflow instructions are in [Codex.md](Codex.md).
