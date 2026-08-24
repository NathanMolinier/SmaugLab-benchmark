# SmaugLab-benchmark

A reproducible benchmarking harness for training [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) v2 models — including [SmaugLab](https://github.com/neuropoly/SmaugLab) DAExt trainers — on git-annexed BIDS datasets.

The `smaugbench` package provides small utilities (image reorientation, resampling, split generation) and a set of end-to-end shell scripts that download data, prepare it for nnU-Net, and launch training from a single JSON config.

## Installation

Requirements:

- Python 3.10+
- [`git-annex`](https://git-annex.branchable.com/) and `jq`
- SSH access to `data.neuro.polymtl.ca` (for the bundled datasets)
- PyTorch with CUDA if you want to train on GPU

```bash
git clone https://github.com/NathanMolinier/SmaugLab-benchmark
cd SmaugLab-benchmark
pip install -e ".[nnunetv2]"
```

The `nnunetv2` extra is optional — leave it out if you only want the preprocessing utilities.

## Quick start

```bash
export SMAUGBENCH_DATA=/path/to/scratch/smaugbench

./scripts/run_pipeline.sh smaugbench/configs/amos22_nnUNetTrainer.json
```

`run_pipeline.sh` reads a JSON config and runs four stages in order:

```
download_datasets.sh  →  prepare_datasets.sh  →  train.sh  →  inference.sh
```

Each stage can be run individually and each is idempotent. See [`scripts/README.md`](scripts/README.md) for the full pipeline reference (env vars, dataset JSON schema, per-step arguments).

## Repository layout

```
smaugbench/
├── configs/      # example pipeline configs (e.g. amos22_nnUNetTrainer.json)
├── datasets/     # dataset JSONs (source repos + commits + train/val/test splits)
└── utils/        # reorient / resample / splits helpers exposed as console scripts
scripts/          # download → prepare → train → inference pipeline (bash)
```

Console scripts installed with the package:

- `smaugbench_reorient_image` — reorient every NIfTI in a folder to a given orientation
- `smaugbench_resample_image` — resample to a target voxel size
- `smaugbench_create_splits_nnunet` — generate nnU-Net's `splits_final.json` from a dataset JSON
- `smaugbench_compute_pairwise_measurements` — pairwise reference/prediction metrics (Dice, NSD, HD, …)
- `smaugbench_compare_results` — aggregate every `metrics*/metrics.csv` under `Dataset<ID>_*` into comparison tables

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `SMAUGBENCH_DATA` | yes | Root folder for `bids/` and `nnUNet/{raw,preprocessed,results}` |
| `SMAUGBENCH_JOBS` | no | CPU jobs (defaults to detected core count) |
| `SMAUGBENCH_JOBSNN` | no | nnU-Net worker count (defaults to `min(SMAUGBENCH_JOBS, RAM_GB / 8)`) |
| `SMAUGBENCH_DEVICE` | no | `cuda` if available, else `cpu` |

## Adding a new dataset

1. Write a dataset JSON under `smaugbench/datasets/<name>.json` following the schema in [`scripts/README.md`](scripts/README.md#dataset-json-schema).
2. Reference it from a pipeline config as `"data_json": "<name>"` (or an absolute path).
3. Run `./scripts/run_pipeline.sh <config.json>`.
