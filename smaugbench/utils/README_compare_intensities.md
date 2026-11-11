# `smaugbench_compare_intensities` — what makes a structure easy or hard to segment?

`smaugbench/utils/compare_intensities.py` measures every labelled structure of the test
images (intensity, size, contrast with its surroundings, frequency), on the raw images
and on images augmented exactly like during training, and correlates these measurements
with the scores of the model (`metrics.csv`). Everything is done separately for the MR
and the CT images.

The setting it is built for: the model is **trained on CT only, with augmentation, and
tested on CT and MR**. Only the CT images are therefore augmented, and the central
question is whether the difference between what the model was trained on (augmented CT)
and what it is tested on (raw MR, or raw CT) explains the scores:

| Comparison | Feature columns | Correlated with |
| --- | --- | --- |
| augmented CT − raw MR | `augdelta_*`, `augabsdelta_*`, `wasserstein_to_CT_aug`, … of the MR rows | MR scores, and MR − CT score difference |
| augmented CT − raw CT | same columns, CT rows | CT scores |
| raw CT − raw MR | `rawdelta_*` | MR − CT score difference |

- [Usage](#usage)
- [Inputs](#inputs)
- [Vocabulary](#vocabulary)
- [What the code does, step by step](#what-the-code-does-step-by-step)
- [Features](#features)
- [Outputs](#outputs)
- [How to read the results](#how-to-read-the-results)
- [Options](#options)

## Usage

```bash
export SMAUGBENCH_DATA=/path/to/smaugbench_data

# Everything is derived from the pipeline config (same file as run_pipeline.sh / inference.sh)
smaugbench_compare_intensities smaugbench/configs/amos22_nnUNetTrainerDAExtGPU_OSSTA.json

# Another metrics file, more augmented draws, explicit output folder
smaugbench_compare_intensities config.json --metrics /path/to/metrics.csv --n-aug 10 -o /path/to/output

# Quick test on 4 images, on CPU
smaugbench_compare_intensities config.json --max-images 4 --n-aug 2 --device cpu
```

Requirements: `smauglab`, `nibabel`, `numpy`, `pandas`, `scipy`, `torch`, `tqdm`, and
`matplotlib` for the figures. `pyradiomics` is **not** needed (see [step 4](#4-measure-every-structure)).

## Inputs

The script works on the folder prepared by `scripts/inference.sh`:

```
${SMAUGBENCH_DATA}/inference/Dataset<dataset_id>_<dataset_name><inference_supplementary_key>/
├── images/<stem>_0000.nii.gz                        # <stem> ends with _MR or _CT
├── labels/<stem>.nii.gz                             # one integer value per structure
└── metrics_<nnunet_folder_name><key>/metrics.csv    # scores of the model
```

| Input | Where it comes from |
| --- | --- |
| Images / labels | `images/` and `labels/` of the inference folder. The modality is the suffix of the file name. |
| Structure names | `LABELS` of the dataset JSON (`data_json` of the config). Value `0` is ignored. |
| Scores | `metrics.csv`: one row per (image, structure) with the columns `reference`, `label` and one column per metric (`DiceSimilarityCoefficient`, `NormalizedSurfaceDistance`, `HausdorffDistance95`, …). Override with `--metrics`. |
| Augmentation | `GPU` section of `nnunet_trainer_config` (override with `--trainer-config`). |
| Normalization | `plans.json` of the trained model in `${SMAUGBENCH_DATA}/nnUNet/results/...` (override with `--plans` or `--norm`). |

## Vocabulary

| Term | Meaning |
| --- | --- |
| modality | `MR` or `CT`, from the file name. |
| source | Where intensities come from: `MR`, `CT` (raw images) or `CT_aug` (augmented CT images; `MR_aug` only if `--aug-source MR` is asked). |
| draw | `0` for the raw image, `1..n` for the augmented versions of that image. |
| instance | One structure in one image (one row of `metrics.csv`). |

## The code

### 1. Read the configuration

`load_pipeline_config` reads the pipeline config and derives the inference folder, the
dataset JSON (list of structures), the `plans.json` of the trained model, the trainer
config (augmentations) and the default metrics folder.

### 2. Normalize the images

`build_normalizer` reproduces the normalization the network sees:

- `--norm nnunet` (default): scheme and statistics of `plans.json`. `CTNormalization`
  clips to the 0.5–99.5 percentiles of the training foreground and standardizes with its
  mean and standard deviation; `ZScoreNormalization` standardizes each image.
- `--norm zscore`: each image is standardized on its own (also the fallback when
  `plans.json` is not found).
- `--norm none`: raw intensities.

All the intensity measurements are therefore expressed **in the units the model works
with**, which is what makes MR, CT and augmented images comparable. The image is then
cropped to the bounding box of its labels plus a 16-voxel margin.

### 3. Augment the images

`build_augmenter` builds the GPU pipeline exactly like `nnUNetTrainerDAExtGPU`, and
`augment` applies it `--n-aug` times to each image of the training modality
(`--aug-source`, CT by default). MR images are not augmented: the model never sees
augmented MR. As during training:

- the label map is passed as the mask, so the in-segmentation transforms behave the same;
- the volume is split into tiles of the training patch size, each tile drawing its own
  random parameters.

The label map returned by the augmentation is the one used to measure the augmented
image (a flip moves the structures). The CPU (spatial) section of the trainer config is
not applied: it does not change intensities.

### 4. Measure every structure

`measure_structures` is called on the raw image (draw 0) and on every augmented draw.
For each structure present in the label map it works on the bounding box of the
structure and computes:

1. **First-order radiomics** (`first_order`) of the voxels inside the structure. They
   follow the definitions of the `firstorder` class of pyradiomics / IBSI and are
   implemented with numpy so that no extra dependency is needed. Entropy and uniformity
   use a histogram with a fixed bin width (`--bin-width`, 0.1 normalized unit by
   default, 25 with `--norm none`, the pyradiomics default).
2. **Contrast with the surroundings** (`surroundings`). The surroundings are the *shell*:
   all the voxels outside the structure that are at most `--shell-mm` (5 mm) away from
   it, found with a Euclidean distance transform that uses the voxel size. Voxels of
   neighbouring structures are part of the shell: they are what the model has to tell
   the structure apart from.
3. **Size** (raw image only).
4. **512 evenly spaced quantiles** (`--samples`) of the structure intensities. This is a
   compact, fixed-size description of the whole distribution, used in steps 5 and 6
   instead of keeping all the voxels.

Each (image, source, draw, structure) is one row of `intensity_stats.csv`.

### 5. Histogram PCA

`add_pca_features`:

1. turns the quantiles of every row into a normalized histogram of `--hist-bins` (64)
   bins on a **common intensity range** (the 0.5–99.5 percentile range of everything
   that was measured; values outside go to the first/last bin);
2. fits **one PCA** on all these histograms (all structures, raw and augmented, MR and
   CT), so that every structure of every source lives in the same space; the first
   `--pca-components` (5) are kept. The loadings (`histogram_pca_components.csv`) tell
   which part of the histogram each component describes;
3. computes, for each row, the distance to the centroid of its structure in its source
   (`pc_centroid_dist`: how variable the appearance of the structure is).

### 6. Compare the sources

Still in `add_pca_features`, each raw instance is compared with the distribution of the
**same structure in every other source** (for an MR image: `CT` and `CT_aug`; for a CT
image: `MR` and `CT_aug`):

- `wasserstein_to_<source>`: 1-Wasserstein distance between the intensities of the
  instance and the pooled intensities of the other source;
- `coverage_by_<source>`: fraction of the instance intensities inside the 1–99
  percentile range of the other source;
- `pc_dist_to_<source>`: distance in the PCA space to the centroid of the other source.

`wasserstein_to_CT_aug` / `coverage_by_CT_aug` of an MR structure answer "did the
augmentation show the model intensities like these?". The same columns on the CT rows
compare raw CT with augmented CT.

The pooled version of these distances (per structure and pair of sources) is written to
`intensity_distances.csv`, and pooled statistics to `intensity_summary.csv`.

### 7. Build the feature tables

- `build_instance_table` → `instance_features.csv`, one row per **(image, structure)**:
  - size features;
  - `raw_<feature>`: measured on the raw image;
  - `aug_<feature>`: the feature on augmented CT. For a CT image it is the mean over
    the augmented draws of that same image; for an MR image (not augmented) it is the
    mean of the structure over all the augmented CT images;
  - `augdelta_<feature>`: `aug − raw`, i.e. **augmented CT − raw CT** on the CT rows and
    **augmented CT − raw MR** on the MR rows;
  - `augabsdelta_<feature>`: its absolute value (size of the mismatch, whatever its
    direction);
  - `aug_to_raw_wasserstein`, `aug_to_raw_pc_dist` (CT rows only): how far a draw moves
    the structure away from the raw image (strength of the augmentation);
  - the cross-source features of step 6.
- `build_structure_table` → `structure_features.csv`, one row per **(modality,
  structure)**: mean of the instance features over the images, plus
  - **frequency**: `n_images`, `frequency` (share of the images of the modality that
    contain the structure) and `voxel_share` (share of the labelled voxels);
  - **separability**: `raw_pc_nearest_structure_dist` (distance to the closest other
    structure in the PCA space) and `raw_pc_separability` (that distance divided by the
    spread of the structure). Same with the `aug_` prefix on augmented images.
- `build_gap_table` → `structure_features_gap.csv`, one row per **structure**, for the
  difference of performance between the test and the training modality:
  - the metric columns contain **MR score − CT score** (mean per structure);
  - the features are those of the MR row (among which `augdelta_*` = augmented CT − raw
    MR), plus `rawdelta_<feature>` = raw CT − raw MR.

### 8. Join the scores

`load_metrics` reads `metrics.csv`, takes the image name from the `reference` column and
the structure from the `label` column, drops the rows with `EmptyRef` (no ground truth)
and replaces infinite values by NaN. The scores are joined to the instance table on
(image, structure) and averaged per (modality, structure) in the structure table. Every
numeric column other than `reference`, `prediction`, `label`, `EmptyRef`, `EmptyPred` is
treated as a metric.

### 9. Correlate features and scores

`correlate` computes the **Spearman** rank correlation between every feature and every
metric, separately for MR and CT, at two levels. At the structure level it is also
computed for the MR − CT score difference (`modality = MR_minus_CT`, from the gap table):

| File | One point is | Question |
| --- | --- | --- |
| `correlations_structure_level.csv`, `modality = MR` or `CT` | a structure (mean feature, mean score) | Which properties make a *structure* well or badly segmented? |
| `correlations_structure_level.csv`, `modality = MR_minus_CT` | a structure (mean feature, MR score − CT score) | Which properties explain how much is lost (or gained) from CT to MR? |
| `correlations_instance_level.csv`, `structure = all` | an (image, structure) pair | Same, using every image (dominated by the differences between structures). |
| `correlations_instance_level.csv`, `structure = <name>` | an image, for that structure only | For a given structure, which *images* are well or badly segmented? |
| `correlations_instance_level.csv`, `structure = mean_within` | — | Mean of the per-structure coefficients: the effect of a feature once the differences between structures are removed. |

A correlation needs at least 5 points with a non-constant feature and score. `p_value`
is the Spearman p-value and `q_value` its Benjamini–Hochberg correction over all the
features tested for the same modality, metric and structure. `group` is the family of
the feature (`radiomics`, `histogram_pca`, `surroundings`, `size`, `frequency`,
`cross_source`, `augmentation_shift`); the prefix of the feature tells which images it
compares (`raw_`, `aug_`, `augdelta_`, `augabsdelta_`, `rawdelta_`).

### 10. Figures

- `intensity_boxplot_<i>.png`: median intensity of each structure, per source.
- `histogram_pca.png`: PC1/PC2, one panel per source, with the structure centroids.
- `correlation_heatmap_<modality>.png`: the 30 features most correlated with the scores
  at the structure level (`*`: q < 0.05). `<modality>` is `MR`, `CT` or `MR_minus_CT`.
- `correlation_scatter_<modality>_<metric>.png`: the 6 most correlated features, one dot
  per structure.

## Features

### First-order radiomics (prefix `raw_`, `aug_`, `augdelta_`, `augabsdelta_`, `rawdelta_`)

| Column | Definition |
| --- | --- |
| `mean`, `std`, `variance` | Mean, standard deviation, variance |
| `minimum`, `maximum`, `range` | Extremes and their difference |
| `p01` … `p99` | 1, 5, 10, 25, 50 (median), 75, 90, 95, 99th percentiles |
| `iqr` | Interquartile range, `p75 − p25` |
| `mad` | Mean absolute deviation from the mean |
| `rmad` | Robust MAD: same, restricted to the voxels between `p10` and `p90` |
| `rms` | Root mean square |
| `energy`, `total_energy` | Sum of squared intensities, and the same times the voxel volume |
| `entropy`, `uniformity` | Shannon entropy (bits) and sum of squared probabilities of the discretized histogram |
| `skewness`, `kurtosis` | Third and fourth standardized moments (kurtosis is 3 for a Gaussian, as in pyradiomics) |

`energy` and `total_energy` grow with the number of voxels: they mostly measure the
size of the structure, read them as such.

### Surroundings (same prefixes)

| Column | Definition |
| --- | --- |
| `surround_mean`, `surround_std` | Mean and standard deviation of the shell |
| `contrast`, `abs_contrast` | Mean inside − mean of the shell, and its absolute value |
| `cnr` | `abs_contrast / sqrt((var_inside + var_shell) / 2)` |
| `surround_overlap` | Histogram intersection of inside and shell: 1 = same distribution, 0 = separable from the intensity alone |
| `surround_wasserstein` | 1-Wasserstein distance between inside and shell |

### Size (raw image)

| Column | Definition |
| --- | --- |
| `n_voxels`, `volume_mm3` | Number of voxels and volume |
| `equivalent_diameter_mm` | Diameter of the sphere of the same volume |
| `bbox_max_extent_mm` | Longest side of the bounding box |
| `bbox_fill` | Volume / bounding-box volume (low for thin, tortuous structures) |
| `boundary_fraction` | Share of the voxels of the structure that touch the outside (close to 1 for small or thin structures) |

### Histogram PCA

`raw_pc1…`, `aug_pc1…`, `augdelta_pc1…`, `augabsdelta_pc1…`, `rawdelta_pc1…`, `*_pc_centroid_dist`,
`*_pc_nearest_structure_dist`, `*_pc_separability` (steps 5 and 7).

### Frequency, cross-source and augmentation shift

`frequency`, `voxel_share`; `wasserstein_to_<source>`, `coverage_by_<source>`,
`pc_dist_to_<source>` (step 6); `aug_to_raw_wasserstein`, `aug_to_raw_pc_dist` (step 7).

## Outputs

Written to `<inference folder>/intensities_<nnunet_folder_name><key>/` (or `-o`):

| File | Content |
| --- | --- |
| `intensity_stats.csv` | One row per (image, source, draw, structure): every measurement of steps 4–6 |
| `intensity_summary.csv` | One row per (source, structure), pooled over the images |
| `intensity_distances.csv` | One row per (structure, pair of sources): Wasserstein distance and coverage |
| `histogram_pca.csv` | PCA scores of every row |
| `histogram_pca_components.csv` | Bin edges, mean histogram and loadings of each component |
| `histogram_pca_variance.csv` | Explained variance ratio of each component |
| `instance_features.csv` | One row per (image, structure): features and scores |
| `structure_features.csv` | One row per (modality, structure): mean features, frequency and mean scores (`<metric>_n` = number of scored images) |
| `structure_features_gap.csv` | One row per structure: MR features, `rawdelta_*` and MR score − CT score |
| `correlations_structure_level.csv` | Step 9, across structures |
| `correlations_instance_level.csv` | Step 9, across and inside structures |
| `*.png` | Figures of step 10 |
| `config.json` | Copy of the input config |

## How to read the results

- **Deltas.** `augdelta_*` is signed (augmented CT − raw): use it to see a direction
  ("MR is brighter than anything the augmentation produced"). `augabsdelta_*` is the size
  of the mismatch: if the augmentation gap explains the performance, it should correlate
  negatively with Dice/NSD (and with the MR − CT difference), positively with HD95.
- **Score difference.** In the `MR_minus_CT` rows, a negative Dice difference means the
  structure is worse on MR than on CT. The difference compares the means of two
  different sets of images (the MR and the CT test images are not the same subjects).
- **Sign.** A positive coefficient means "the larger the feature, the larger the
  metric". Dice and NSD are better when high, HD95 is better when low: a feature that
  helps has a positive correlation with Dice and a negative one with HD95.
- **Sample size.** The structure-level analysis has one point per structure (15 for
  AMOS22): only strong correlations are detectable and a single structure can drive a
  coefficient. Look at the scatter plots before trusting a number.
- **Many features are tested** (around 170) and many are redundant (percentiles,
  `energy` vs. volume, …). Use `q_value`, not `p_value`, and treat the results as
  hypotheses to check, not as proofs.
- **Confounding by size.** Small structures have a lower Dice for geometric reasons
  alone, and size is correlated with several intensity features. The `mean_within` and
  per-structure rows of `correlations_instance_level.csv` are not affected by the
  differences between structures.
- **Correlation is not causation**: the features are not independent from each other.
- **Frequency** is measured on the test images of the inference folder, not on the
  training set. It only has a structure-level correlation.
- **Augmented images** are drawn at random: results change slightly with `--seed` and
  get more stable with a larger `--n-aug`.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `-o`, `--output` | `<inference folder>/intensities_<name><key>` | Output folder |
| `--metrics` | `<inference folder>/metrics_<name><key>` | `metrics.csv` or the folder containing it |
| `--n-aug` | 5 | Augmented draws per image (0: no augmentation) |
| `--aug-source` | `CT` | Training modality: its images are augmented and the others are compared with them |
| `--trainer-config` | `nnunet_trainer_config` of the config | SmaugLab augmentation config |
| `--norm` | `nnunet` | `nnunet`, `zscore` or `none` |
| `--plans` | from the nnUNet results folder | `plans.json` used by `--norm nnunet` |
| `--patch-size` | patch size of the plans | Size of the augmented tiles |
| `--batch-size` | 2 | Tiles augmented together |
| `--samples` | 512 | Quantiles kept per (image, draw, structure) |
| `--shell-mm` | 5 | Thickness of the shell used as surroundings |
| `--bin-width` | 0.1 (25 with `--norm none`) | Bin width for entropy, uniformity and overlap |
| `--hist-bins` | 64 | Bins of the histograms given to the PCA |
| `--pca-components` | 5 | Principal components kept |
| `--device` | `$SMAUGBENCH_DEVICE`, else cuda/cpu | Device of the augmentations |
| `--seed` | 42 | Random seed |
| `--max-images` | all | Only process the first N images |
