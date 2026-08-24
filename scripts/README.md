# Scripts

End-to-end pipeline for training [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) v2 models with [SmaugLab](https://github.com/neuropoly/SmaugLab) trainers on git-annexed BIDS datasets in a reproducible way. The four scripts run in order:

```
download_datasets.sh  →  prepare_datasets.sh  →  train.sh  →  inference.sh
```

Each script can be invoked individually, or all four can be driven from a single JSON config through `run_pipeline.sh` (see [§5](#5-run_pipelinesh)).

Adapted from [`totalspineseg/scripts`](https://github.com/neuropoly/totalspineseg/tree/4502d41bcb4a12e44f4be666411461ca81b02d89/scripts).

---

## Requirements

- Bash 4+
- `jq`
- `git` and [`git-annex`](https://git-annex.branchable.com/) with SSH access to `data.neuro.polymtl.ca`
- Python environment with the `smaugbench` package and `nnunetv2` installed (see Installation instructions in the main README)
- For GPU training: PyTorch with CUDA

## Environment variables

| Variable | Required | Default | Used by |
| --- | --- | --- | --- |
| `SMAUGBENCH_DATA` | **yes** | — | all scripts. Root folder where `bids/` and `nnUNet/{raw,preprocessed,results}` are created. |
| `SMAUGBENCH_JOBS` | no | detected CPU count (`lscpu` on Linux or `sysctl hw.ncpu` on macOS) | `prepare_datasets.sh`, `train.sh` |
| `SMAUGBENCH_JOBSNN` | no | `min(SMAUGBENCH_JOBS, RAM_GB / 8)` clamped to ≥ 1 | `prepare_datasets.sh`, `train.sh` |
| `SMAUGBENCH_DEVICE` | no | `cuda` if `torch.cuda.is_available()` else `cpu` | `train.sh` |

Set once before running the scripts or inside your shell configuration file (e.g., `.bashrc`, `.zshrc`), for example:

```bash
export SMAUGBENCH_DATA=/path/to/scratch/smaugbench
```

## Dataset JSON schema

Each dataset is described by a JSON file (see `smaugbench/datasets/amos22.json` for a full example):

```json
{
  "TYPE": "LABEL",
  "CONTRASTS": "CT_MR",
  "SOURCE":  ["git@data.neuro.polymtl.ca:datasets/abdominal-amos22.git"],
  "COMMIT":  ["dfff51b6fbb46dcbe2465739fb52f1ca3e3bb767"],
  "LABELS": {"0": "background","1": "spleen"},
  "TRAINING":   [{ "IMAGE": "abdominal-amos22/sub-amos0056/anat/sub-amos0056_CT.nii.gz",
                   "LABEL": "abdominal-amos22/derivatives/labels/sub-amos0056/anat/sub-amos0056_CT_label-abdominal_dlabel.nii.gz" }],
  "VALIDATION": [{ "IMAGE": "…", "LABEL": "…" }],
  "TESTING":    [{ "IMAGE": "…", "LABEL": "…" }]
}
```

- `SOURCE[i]` and `COMMIT[i]` are paired: each source repo is checked out at the matching commit.
- `IMAGE` / `LABEL` paths are **relative to `$SMAUGBENCH_DATA/bids/`** and their first path segment is the repo name that owns them (used by `git annex get`).
- All image/label files must be `.nii.gz`.
- For every image, there must be exactly **one** label file with the same base name plus a `_<suffix>` marker (e.g. `sub-amos0056_CT.nii.gz` ↔ `sub-amos0056_CT_label-abdominal_dlabel.nii.gz`). `prepare_datasets.sh` will strip the suffix and rename it to match.

The dataset argument accepted by scripts is either an **absolute path to a JSON file** or a JSON **shortcut name** already defined in `smaugbench/datasets/` that resolves to `smaugbench/datasets/<name>.json` (so `amos22` → `smaugbench/datasets/amos22.json`).

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
$SMAUGBENCH_DATA/
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
$SMAUGBENCH_DATA/nnUNet/
├── raw/Dataset501_AMOS22/{imagesTr,labelsTr,imagesTs,labelsTs}
└── preprocessed/Dataset501_AMOS22/splits_final.json
```

Requires that `download_datasets.sh` has already populated `$SMAUGBENCH_DATA/bids/`.

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

Training runs with `--c` (continue), so re-launching resumes from the last checkpoint. Outputs land under `$SMAUGBENCH_DATA/nnUNet/results/Dataset<ID>_<NAME>/…`.

---

## 4. `inference.sh`

Preprocess the `TESTING` split, run `nnUNetv2_predict` on the trained weights, and compute pairwise reference/prediction metrics (Dice, NSD, HD).

**Usage**

```bash
./scripts/inference.sh <pipeline-config.json> [--images-only]
```

The config is the same JSON consumed by `run_pipeline.sh` — required keys are `data_json`, `dataset_id`, `dataset_name`, `nnunet_planner`, `nnunet_plans`, `configuration`, `nnunet_trainer`, and `nnunet_folder_name`. Optional keys:

| Key | Default | Description |
| --- | --- | --- |
| `inference_do_preprocessing` | `true` | If `false` (and the inference dir already exists), skip copying and reorient/resample and reuse the previously preprocessed images/labels. |
| `inference_all_weights` | `false` | If `true`, iterate over every `*.pth` checkpoint in the fold folder, writing one prediction dir (and one metrics dir when labels are available) per checkpoint. Otherwise only the default `checkpoint_best.pth` is used. |

**`--images-only` mode**

Passing `--images-only` (or having no `LABEL` on the first `TESTING` entry) tells the script to:

- skip copying / renaming / reorienting / resampling labels
- skip the pairwise metrics computation

Only the images are preprocessed and fed through `nnUNetv2_predict`. Use this when running the trained model on a dataset that has no ground truth.

**Result**

```
$SMAUGBENCH_DATA/inference/Dataset<ID>_<NAME>/
├── images/                        # preprocessed test images (nnUNet _0000 suffix)
├── labels/                        # preprocessed reference labels (skipped in --images-only)
├── prediction_<nnunet_folder>/    # nnUNetv2_predict output (one dir per checkpoint when inference_all_weights=true, suffixed with the checkpoint name)
└── metrics_<nnunet_folder>/       # metrics.csv + mapping.json (skipped in --images-only; one dir per checkpoint when inference_all_weights=true)
```

**Example**

```bash
./scripts/inference.sh smaugbench/configs/amos22_nnUNetTrainer.json
./scripts/inference.sh smaugbench/configs/amos22_nnUNetTrainer.json --images-only
```

---

## 5. `run_pipeline.sh`

Drives all four scripts from a single JSON config, then archives that config next to the trained weights.

**Usage**

```bash
./scripts/run_pipeline.sh <pipeline-config.json>
```

**Config schema** (all `steps.*` default to `true`; setting any of them to `false` skips that step):

```json
{
  "steps": {
    "download": true,
    "prepare": true,
    "train": true,
    "inference": true
  },

  "data_json": "amos22",

  "dataset_id": 501,
  "dataset_name": "AMOS22",

  "nnunet_planner": "nnUNetPlannerResEncL",
  "nnunet_plans":   "nnUNetPlans",
  "configuration":  "3d_fullres",

  "nnunet_trainer": "nnUNetTrainer",
  "nnunet_trainer_config": null,
  "nnunet_folder_name": "amos22_nnUNetTrainer",

  "inference_do_preprocessing": true,
  "inference_images_only": false
}
```

| Key | Required for | Default | Description |
| --- | --- | --- | --- |
| `steps.download` / `steps.prepare` / `steps.train` / `steps.inference` | — | `true` | Toggle each pipeline step. |
| `data_json` | all steps | — | Absolute path or shortcut name for the dataset JSON (see [§Dataset JSON schema](#dataset-json-schema)). |
| `dataset_id` | prepare, train, inference | — | nnUNet dataset ID (integer). |
| `dataset_name` | prepare, inference | — | nnUNet dataset name (also used to locate the archive folder after training). |
| `nnunet_planner` | prepare, inference | `nnUNetPlannerResEncL` | |
| `nnunet_plans` | prepare, train, inference | `nnUNetPlans` | |
| `configuration` | prepare, train, inference | `3d_fullres` | |
| `nnunet_trainer` | train, inference | — | e.g. `nnUNetTrainer`, `nnUNetTrainerDA5`, `nnUNetTrainerDAExt…` |
| `nnunet_trainer_config` | train (DAExt only) | — | Path to the DAExt GPU-params JSON. |
| `nnunet_folder_name` | train, inference | — | Sub-folder under `$SMAUGBENCH_DATA/nnUNet/results/` where weights live and where predictions/metrics are written. |
| `inference_do_preprocessing` | inference | `true` | Reuse an existing preprocessed inference folder when `false`. |
| `inference_images_only` | inference | `false` | Skip label handling and metrics — passes `--images-only` through to `inference.sh`. Auto-enabled when the first `TESTING` entry has no `LABEL`. |

**Config archival**

After the `train` step finishes, the config JSON is copied to:

```
$SMAUGBENCH_DATA/nnUNet/results/Dataset<ID>_<NAME>/<trainer>__<plans>__<configuration>/fold_0/config.json
```

so each checkpoint carries the exact config used to produce it. Runs that skip the `train` step do not archive.

**Example**

```bash
cat > amos22_run.json <<'EOF'
{
  "data_json": "amos22",
  "dataset_id": 501,
  "dataset_name": "AMOS22",
  "nnunet_trainer": "nnUNetTrainer",
}
EOF

./scripts/run_pipeline.sh amos22_run.json
```

To re-train against already-downloaded and already-prepared data, set both `steps.download` and `steps.prepare` to `false`. To only run inference against existing weights, set `steps.download`, `steps.prepare`, and `steps.train` to `false`.

---

## End-to-end example

Run the scripts individually:

```bash
export SMAUGBENCH_DATA=/scratch/smaugbench

./scripts/download_datasets.sh amos22
./scripts/prepare_datasets.sh  amos22 501 AMOS22
./scripts/train.sh             501    nnUNetTrainer
./scripts/inference.sh         smaugbench/configs/amos22_nnUNetTrainer.json
```

Or drive them from a config:

```bash
export SMAUGBENCH_DATA=/scratch/smaugbench

./scripts/run_pipeline.sh amos22_run.json
```

## Troubleshooting

- **`Please set the SMAUGBENCH_DATA environment variable.`** — export it and retry.
- **`Could not find data JSON file.`** — pass an absolute path, or make sure `smaugbench/datasets/<name>.json` exists.
- **`Could not find BIDS data folder.`** — run `download_datasets.sh` before `prepare_datasets.sh`.
- **`Expected exactly one label file for image …`** — an image in the JSON has zero or more than one matching label file in `labelsTr` / `labelsTs`. Check the naming convention (`<image-basename>_<suffix>.nii.gz`).
- **`git annex get` asks for credentials or fails** — confirm SSH access to `data.neuro.polymtl.ca` and that `git-annex` is installed.
