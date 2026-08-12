# Scripts

End-to-end pipeline for training [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) v2 models with [SmaugLab](https://github.com/neuropoly/SmaugLab) trainers on git-annexed BIDS datasets in a reproducible way. The three scripts run in order:

```
download_datasets.sh  →  prepare_datasets.sh  →  train.sh
```

Adapted from [`totalspineseg/scripts`](https://github.com/neuropoly/totalspineseg/tree/4502d41bcb4a12e44f4be666411461ca81b02d89/scripts).

---

## Requirements

- Bash 4+
- `jq`
- `git` and [`git-annex`](https://git-annex.branchable.com/) with SSH access to `data.neuro.polymtl.ca`
- Python environment with the `smbench` package and `nnunetv2` installed (see Installation instructions in the main README)
- For GPU training: PyTorch with CUDA

## Environment variables

| Variable | Required | Default | Used by |
| --- | --- | --- | --- |
| `SMBENCH_DATA` | **yes** | — | all scripts. Root folder where `bids/` and `nnUNet/{raw,preprocessed,results}` are created. |
| `SMBENCH_JOBS` | no | detected CPU count (`lscpu` on Linux or `sysctl hw.ncpu` on macOS) | `prepare_datasets.sh`, `train.sh` |
| `SMBENCH_JOBSNN` | no | `min(SMBENCH_JOBS, RAM_GB / 8)` clamped to ≥ 1 | `prepare_datasets.sh`, `train.sh` |
| `SMBENCH_DEVICE` | no | `cuda` if `torch.cuda.is_available()` else `cpu` | `train.sh` |

Set once before running the scripts or inside your shell configuration file (e.g., `.bashrc`, `.zshrc`), for example:

```bash
export SMBENCH_DATA=/path/to/scratch/smbench
```

## Dataset JSON schema

Each dataset is described by a JSON file (see `smbench/datasets/amos22.json` for a full example):

```json
{
  "TYPE": "LABEL",
  "CONTRASTS": "CT_MR",
  "SOURCE":  ["git@data.neuro.polymtl.ca:datasets/abdominal-amos22.git"],
  "COMMIT":  ["dfff51b6fbb46dcbe2465739fb52f1ca3e3bb767"],
  "TRAINING":   [{ "IMAGE": "abdominal-amos22/sub-amos0056/anat/sub-amos0056_CT.nii.gz",
                   "LABEL": "abdominal-amos22/derivatives/labels/sub-amos0056/anat/sub-amos0056_CT_label-abdominal_dlabel.nii.gz" }],
  "VALIDATION": [{ "IMAGE": "…", "LABEL": "…" }],
  "TESTING":    [{ "IMAGE": "…", "LABEL": "…" }]
}
```

- `SOURCE[i]` and `COMMIT[i]` are paired: each source repo is checked out at the matching commit.
- `IMAGE` / `LABEL` paths are **relative to `$SMBENCH_DATA/bids/`** and their first path segment is the repo name that owns them (used by `git annex get`).
- All image/label files must be `.nii.gz`.
- For every image, there must be exactly **one** label file with the same base name plus a `_<suffix>` marker (e.g. `sub-amos0056_CT.nii.gz` ↔ `sub-amos0056_CT_label-abdominal_dlabel.nii.gz`). `prepare_datasets.sh` will strip the suffix and rename it to match.

The dataset argument accepted by scripts is either an **absolute path to a JSON file** or a JSON **shortcut name** already defined in `smbench/datasets/` that resolves to `smbench/datasets/<name>.json` (so `amos22` → `smbench/datasets/amos22.json`).

---

## 1. `download_datasets.sh`

Clone the BIDS repos listed in a dataset JSON and pull the git-annex payload for every file referenced in `TRAINING`, `VALIDATION`, and `TESTING`.

**Usage**

```bash
./scripts/download_datasets.sh <dataset.json>
```

**Example**

```bash
./scripts/download_datasets.sh amos22.json
```

**Result**

```
$SMBENCH_DATA/
└── bids/
    └── abdominal-amos22/
        ├── sub-amos0056/…
        └── derivatives/labels/…
```

Re-running the script is safe: repos that already exist are skipped, and `git annex get` is a no-op for files already present.

---

## 2. `prepare_datasets.sh`

Convert the BIDS layout into nnU-Net's `Dataset<ID>_<NAME>` layout, reorient to RPI, resample to 1×1×1 mm, then run `nnUNetv2_plan_and_preprocess` and write a `splits_final.json` that matches the `TRAINING` / `VALIDATION` split from the JSON.

**Usage**

```bash
./scripts/prepare_datasets.sh <dataset.json> <DATASET_ID> <DATASET_NAME> [nnUNetPlanner] [nnUNetPlans] [configuration]
```

| Position | Arg | Default | Description |
| --- | --- | --- | --- |
| 1 | dataset JSON | — | Absolute path or shortcut name (e.g., `amos22`) |
| 2 | `DATASET_ID` | — | nnUNet dataset ID, e.g. `501` |
| 3 | `DATASET_NAME` | — | nnUNet dataset name to form `Dataset<ID>_<NAME>` |
| 4 | `nnUNetPlanner` | `nnUNetPlannerResEncL` | The nnUNet planner to use |
| 5 | `nnUNetPlans` | `nnUNetPlans` | The nnUNet plans to use |
| 6 | `configuration` | `3d_fullres` | The nnUNet configuration to use |

**Example**

```bash
./scripts/prepare_datasets.sh amos22 501 AMOS22
```

**Result**

```
$SMBENCH_DATA/nnUNet/
├── raw/Dataset501_AMOS22/{imagesTr,labelsTr,imagesTs,labelsTs}
└── preprocessed/Dataset501_AMOS22/splits_final.json
```

Requires that `download_datasets.sh` has already populated `$SMBENCH_DATA/bids/`.

---

## 3. `train.sh`

Launch `nnUNetv2_train` on fold 0 of a prepared dataset. Currently hard-coded to fold 0.

**Usage**

Default nnUNet trainers (e.g., `nnUNetTrainer`, `nnUNetTrainerDA5`, etc.) take the following arguments:

```bash
./scripts/train.sh <DATASET_ID> <nnUNetTrainer> [nnUNetPlanner] [nnUNetPlans] [configuration]
```

DAExt trainers (`nnUNetTrainerDAExt…`) from [SmaugLab](https://github.com/neuropoly/SmaugLab) take an extra config path as arg 3:

```bash
./scripts/train.sh <DATASET_ID> <nnUNetTrainerDAExt...> <config.json> [nnUNetPlanner] [nnUNetPlans] [configuration]
```

| Position (std / DAExt) | Arg | Default | Description |
| --- | --- | --- | --- |
| 1 / 1 | `DATASET_ID` | — | nnUNet dataset ID, e.g. `501` |
| 2 / 2 | `nnUNetTrainer` | — | The nnUNet trainer to use |
| — / 3 | DAExt config JSON | — | The DAExt config file to use |
| 3 / 4 | `nnUNetPlanner` | `nnUNetPlannerResEncL` | The nnUNet planner to use |
| 4 / 5 | `nnUNetPlans` | `nnUNetPlans` | The nnUNet plans to use |
| 5 / 6 | `configuration` | `3d_fullres` | The nnUNet configuration to use |

**Examples**

```bash
# Standard trainer
./scripts/train.sh 501 nnUNetTrainer

# Standard trainer with a custom configuration
./scripts/train.sh 501 nnUNetTrainer nnUNetPlannerResEncL nnUNetPlans 3d_fullres

# DAExt trainer with its GPU-params JSON
./scripts/train.sh 501 nnUNetTrainerDAExtGPU ./daext_config.json
```

Training runs with `--c` (continue), so re-launching resumes from the last checkpoint. Outputs land under `$SMBENCH_DATA/nnUNet/results/Dataset<ID>_<NAME>/…`.

---

## End-to-end example

```bash
export SMBENCH_DATA=/scratch/smbench

./scripts/download_datasets.sh amos22
./scripts/prepare_datasets.sh  amos22 501 AMOS22
./scripts/train.sh             501    nnUNetTrainer
```

## Troubleshooting

- **`Please set the SMBENCH_DATA environment variable.`** — export it and retry.
- **`Could not find data JSON file.`** — pass an absolute path, or make sure `smbench/datasets/<name>.json` exists.
- **`Could not find BIDS data folder.`** — run `download_datasets.sh` before `prepare_datasets.sh`.
- **`Expected exactly one label file for image …`** — an image in the JSON has zero or more than one matching label file in `labelsTr` / `labelsTs`. Check the naming convention (`<image-basename>_<suffix>.nii.gz`).
- **`git annex get` asks for credentials or fails** — confirm SSH access to `data.neuro.polymtl.ca` and that `git-annex` is installed.
