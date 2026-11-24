# `smaugbench_compare_intensities`: what makes a structure easy or hard to segment?

`smaugbench/utils/compare_intensities.py` measures every labelled structure of the test
images (how often it appears, its size, its intensities, its contrast with the
surroundings) and correlates these measurements with the scores of the model
(`metrics.csv`), separately for MR and CT.

It is built for a model **trained on augmented CT and tested on CT and MR**. The CT
images are therefore also augmented with the training augmentations, which lets us ask:
*is a structure badly segmented on MR because its MR intensities look like nothing the
model saw on (augmented) CT?*

All the examples below come from one run: AMOS22, model `nnUNetTrainerDAExtGPU_OSSTA`,
30 CT and 60 MR test images, 15 structures, 5 augmented draws per CT image.

- [Usage](#usage)
- [Vocabulary](#vocabulary)
- [Features](#features)
- [Correlations with the scores](#correlations-with-the-scores)
- [Outputs](#outputs)
- [Pitfalls](#pitfalls)
- [Options](#options)

## Usage

```bash
export SMAUGBENCH_DATA=/path/to/smaugbench_data

# Same config as run_pipeline.sh / inference.sh: everything else is derived from it
smaugbench_compare_intensities smaugbench/configs/amos22_nnUNetTrainerDAExtGPU_OSSTA.json

# Other metrics file, more augmented draws, explicit output folder
smaugbench_compare_intensities config.json --metrics /path/to/metrics.csv --n-aug 10 -o /path/to/output

# Quick test on CPU
smaugbench_compare_intensities config.json --max-images 4 --n-aug 2 --device cpu

# CT and MR test sets in two inference folders (e.g. TotalSegmentator)
smaugbench_compare_intensities smaugbench/configs/totalsegmentator-ct_3mm_nnUNetTrainerDAExtGPU_OSSTA.json \
    --extra-config smaugbench/configs/totalsegmentator-ct_3mm_nnUNetTrainerDAExtGPU_OSSTA_testMRI.json
```

It reads the folder prepared by `scripts/inference.sh`:

```
${SMAUGBENCH_DATA}/inference/Dataset<id>_<name><key>/
├── images/<stem>_0000.nii.gz                        # <stem> ends with _MR or _CT
├── labels/<stem>.nii.gz
└── metrics_<nnunet_folder_name><key>/metrics.csv    # scores per (image, structure)
```

**Two inference folders.** When the MR test set is run with its own config (own
`data_json` and `inference_supplementary_key`, e.g. `Dataset301_TOTALSEGMENTATOR-CT-3mm`
and `Dataset301_TOTALSEGMENTATOR-CT-3mm_MRIDATA`), pass the MR config with
`--extra-config`. Its images, labels and `metrics.csv` are added to those of the main
config. Structures are matched **by name**, so the two dataset JSONs can use different
label values (`autochthon_left` is 5 in `totalsegmentator-ct` and 4 in
`totalsegmentator-mri`) and different subsets: the 50 MR structures are compared with
the same structures among the 117 CT ones, and MR-only structures (`lung_left`,
`vertebrae`, …) are added with no CT counterpart. Model, normalization, augmentation and
output folder always come from the main config.

Structure names come from `LABELS` in the dataset JSON, augmentations from the `GPU`
section of `nnunet_trainer_config`, and normalization from the `plans.json` of the
trained model. Requirements: `smauglab`, `nibabel`, `numpy`, `pandas`, `scipy`, `torch`,
`tqdm` (+ `matplotlib` for the figures).

## Vocabulary

| Term | Meaning |
| --- | --- |
| source | Where the intensities come from: `CT`, `MR` (raw test images) or `CT_aug` (CT images augmented like during training). |
| draw | `0` = raw image, `1..n` = augmented versions of that image. |
| instance | One structure in one image, e.g. the liver of `sub-amos0021_CT`. |

**Intensities are in normalized units**, the ones the network sees (nnU-Net
normalization from `plans.json`, here `CTNormalization`). MR, CT and augmented CT are
thus directly comparable.

Most intensity features come with a **prefix** telling which images they describe.
Example with the mean intensity of the liver (`structure_features.csv`):

| Prefix | Meaning | Liver, CT row | Liver, MR row |
| --- | --- | --- | --- |
| `raw_` | measured on the raw test image | `raw_mean` = 1.38 | `raw_mean` = 2.08 |
| `aug_` | measured on augmented CT (for MR rows: mean over all augmented CT) | `aug_mean` = 0.97 | `aug_mean` = 0.97 |
| `augdelta_` | `aug_ − raw_`: augmented CT − this raw image | `augdelta_mean` = −0.40 | `augdelta_mean` = −1.11 |
| `augabsdelta_` | `abs(augdelta_)`: size of the mismatch, whatever its direction | 0.40 | 1.11 |
| `rawdelta_` | raw CT − raw MR (only in `structure_features_gap.csv`) | — | `rawdelta_mean` = −0.70 |

Read: the augmentation darkens the CT liver (1.38 → 0.97), but the MR liver is even
brighter (2.08), so the model was trained on livers ~1.1 units darker than the MR
livers it is tested on.

## Features

### 1. Frequency

How often the structure is seen. Structure level only.

| Column | Meaning | Example |
| --- | --- | --- |
| `n_images` | Images of the modality that contain the structure | bladder on MR: 2 of 60 |
| `frequency` | `n_images` / number of images | bladder: 0.97 on CT, 0.03 on MR |
| `voxel_share` | Share of all the labelled voxels that belong to the structure | liver: 0.49 (CT), adrenal glands: 0.001 |

### 2. Size

Measured on the raw image.

| Column | Meaning | Example (CT) |
| --- | --- | --- |
| `n_voxels`, `volume_mm3` | Number of voxels, volume | liver 1.3 L, right adrenal gland 3.7 mL |
| `equivalent_diameter_mm` | Diameter of a sphere of the same volume | liver 134 mm, right adrenal 19 mm |
| `bbox_max_extent_mm` | Longest side of the bounding box | long for the aorta and esophagus |
| `bbox_fill` | Volume / bounding-box volume: low for thin, curved structures | prostate/uterus 0.46, duodenum 0.11 |
| `boundary_fraction` | Share of the voxels touching the outside: close to 1 for small or thin structures | liver 0.07, esophagus 0.34, adrenal glands 0.55 |

### 3. Intensity distribution (first-order radiomics)

Computed on the voxels inside the structure, with the pyradiomics / IBSI definitions.

| Column | Meaning | Example |
| --- | --- | --- |
| `mean`, `p50` (median) | Typical intensity | kidneys 1.40 on CT, 2.77 on MR |
| `std`, `variance`, `iqr`, `mad`, `rmad` | Spread (`rmad`: MAD of the voxels between p10 and p90, robust to outliers) | stomach `std` 0.46 on CT (air + content), liver 0.04 |
| `minimum`, `maximum`, `range`, `p01` … `p99` | Extremes and percentiles | |
| `rms`, `energy`, `total_energy` | Root mean square; sum of squares (grows with the size) | |
| `entropy`, `uniformity` | How spread the discretized histogram is (bins of `--bin-width`) | liver entropy 0.73 on CT vs 4.33 on MR: MR livers are much less homogeneous |
| `skewness`, `kurtosis` | Asymmetry and tails of the histogram | liver skewness −5.3 on CT: a long tail of dark voxels |

### 4. Contrast with the surroundings

The surroundings are a **shell** of `--shell-mm` (5 mm) around the structure, whatever
the labels in it: what the model has to separate the structure from.

| Column | Meaning | Example (MR) |
| --- | --- | --- |
| `surround_mean`, `surround_std` | Intensity of the shell | |
| `contrast`, `abs_contrast` | Mean inside − mean of the shell | right kidney 2.07, esophagus 0.13 |
| `cnr` | Contrast-to-noise ratio: `abs_contrast` / pooled std | right kidney 2.52, esophagus 0.37 |
| `surround_overlap` | Overlap of the inside and shell histograms: 1 = identical, 0 = separable from intensity alone | right kidney 0.23, duodenum 0.70 |
| `surround_wasserstein` | Distance between the inside and shell distributions | right kidney 2.12, esophagus 0.40 |

### 5. Histogram shape (PCA)

The intensity histogram of every (image, draw, structure) is computed on a common
intensity range and **one PCA** is fitted on all of them (all structures, all sources),
so every instance becomes a point in the same small space (`histogram_pca.png`).

| Column | Meaning | Example |
| --- | --- | --- |
| `pc1` … `pc5` | Coordinates of the histogram on the first 5 components (`histogram_pca_components.csv` shows which part of the histogram each one describes) | PC1 and PC2 explain 14 % and 13 % of the variance |
| `pc_centroid_dist` | Distance to the centroid of the same structure in the same source: how variable the structure looks across images | |
| `pc_nearest_structure_dist` | Distance between the structure centroid and the closest other structure (structure level) | |
| `pc_separability` | `pc_nearest_structure_dist` / `pc_centroid_dist`: > 1 = the structure has its own histogram, < 1 = it looks like its neighbours | MR prostate/uterus 3.65, MR spleen 0.07 |

### 6. Distance to the other sources

Each raw instance is compared with the pooled distribution of the **same structure in
the other sources** (an MR liver vs. all CT livers, and vs. all augmented CT livers).
This is the core of the question "did the model see intensities like these?".

| Column | Meaning | Example (MR rows) |
| --- | --- | --- |
| `wasserstein_to_<source>` | Distance between the intensities of the instance and those of the other source | `wasserstein_to_CT_aug`: kidneys 1.7, esophagus 0.5 |
| `coverage_by_<source>` | Share of the instance intensities inside the [p1, p99] range of the other source: 1 = everything was seen | `coverage_by_CT_aug`: left kidney 0.38, esophagus 1.00 |
| `pc_dist_to_<source>` | Same idea in the PCA space | |

Pooled per structure, the same comparisons are in `intensity_distances.csv`. Averaged
over structures:

| Pair | Wasserstein | Share of A inside B | Share of B inside A |
| --- | --- | --- | --- |
| MR vs CT | 1.00 | 0.37 | 0.95 |
| MR vs CT_aug | 1.12 | 0.82 | 0.88 |
| CT vs CT_aug | 0.43 | 1.00 | 0.64 |

Read: only 37 % of MR intensities fall in the raw CT range, but 82 % fall in the
augmented CT range: the augmentation widens what the model sees (all raw CT is covered
by augmented CT, and augmented CT goes well beyond raw CT).

### 7. Strength of the augmentation

CT rows only: how far the augmented draws move a structure away from its raw image.

| Column | Meaning | Example |
| --- | --- | --- |
| `aug_to_raw_wasserstein` | Distance between raw and augmented intensities | spleen 0.81, duodenum 0.37 |
| `aug_to_raw_pc_dist` | Same in the PCA space | |

## Correlations with the scores

Every feature is correlated (Spearman) with every metric of `metrics.csv`
(Dice, NSD, HD95, …), separately for MR and CT.

| File / rows | One point is | Question |
| --- | --- | --- |
| `correlations_structure_level.csv`, `modality` = `CT` / `MR` | a structure (mean feature, mean score) | Which structures are hard, and why? |
| `correlations_structure_level.csv`, `modality` = `MR_minus_CT` | a structure (MR features, MR score − CT score) | Which structures lose the most from CT to MR? |
| `correlations_instance_level.csv`, `structure` = `all` | an (image, structure) pair | Same as structure level, with every image |
| `correlations_instance_level.csv`, `structure` = `<name>` | an image, for that structure | For the liver, which images are badly segmented? |
| `correlations_instance_level.csv`, `structure` = `mean_within` | mean of the per-structure coefficients | Effect of a feature once the differences between structures are removed |

Columns: `n` (points), `spearman_r`, `p_value`, `q_value` (Benjamini–Hochberg over the
features tested together) and `group` (feature family).

Example (Dice), from the run above:

- **Structure level**: size dominates. CT: `boundary_fraction` r = −0.91,
  `volume_mm3` r = +0.89. MR: `voxel_share` r = +0.98.
- **MR − CT**: the structures losing the most Dice on MR are bladder (−0.57),
  prostate/uterus (−0.45), right adrenal gland (−0.29) and esophagus (−0.23); the liver
  loses only 0.02.
- **Within a structure** (`mean_within`): `boundary_fraction` still matters
  (r = −0.44 on CT, −0.57 on MR), followed by `raw_surround_overlap`
  (r = −0.37 on CT): images where the structure blends into its surroundings are
  segmented worse.

Sign: a positive r means "larger feature, larger metric". Dice and NSD are better when
high, HD95 when low.

## Outputs

Written to `<inference folder>/intensities_<nnunet_folder_name><key>/` (or `-o`).

| File | One row per | Example size |
| --- | --- | --- |
| `intensity_stats.csv` | (image, source, draw, structure): every raw measurement | 3428 rows |
| `intensity_summary.csv` | (source, structure): statistics pooled over images | |
| `intensity_distances.csv` | (structure, pair of sources): Wasserstein and coverage | |
| `histogram_pca.csv`, `histogram_pca_components.csv`, `histogram_pca_variance.csv` | PCA scores, loadings, explained variance | |
| `instance_features.csv` | (image, structure): all features + scores | 1220 rows |
| `structure_features.csv` | (modality, structure): mean features + mean scores | 30 rows |
| `structure_features_gap.csv` | structure: MR features, `rawdelta_*`, MR − CT scores | 15 rows |
| `correlations_structure_level.csv`, `correlations_instance_level.csv` | (modality, metric, [structure], feature) | |
| `config.json` | copy of the input config | |

Figures:

| Figure | Content |
| --- | --- |
| `intensity_boxplot_<i>.png` | Median intensity of each structure, per source |
| `histogram_pca.png` | PC1 vs PC2, one panel per source, structure centroids labelled |
| `correlation_heatmap_<modality>.png` | 30 features most correlated with the scores (`*`: q < 0.05); `<modality>` = `CT`, `MR` or `MR_minus_CT` |
| `correlation_scatter_<modality>_<metric>.png` | 6 most correlated features, one dot per structure |

## Pitfalls

- **Few points at the structure level** (15 for AMOS22): one structure can drive a
  coefficient. Bladder and prostate/uterus appear in only 2 MR images and
  sit at the low end of both `voxel_share` and Dice on MR. Look at the scatter plots.
- **Size confounds everything**: small structures have a low Dice for geometric reasons,
  and `energy`, `total_energy`, `n_voxels` are size in disguise. Use the `mean_within`
  rows to remove differences between structures.
- **Many redundant features** (~170 tested): use `q_value`, and treat results as
  hypotheses.
- **MR − CT compares different subjects** (MR and CT test images are not paired).
- **Frequency is measured on the test images**, not on the training set.
- **Augmentation is random**: results move slightly with `--seed`; raise `--n-aug` for
  stability.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--extra-config` | none | Configs of other inference folders analysed with the main one (structures matched by name) |
| `-o`, `--output` | `<inference folder>/intensities_<name><key>` | Output folder |
| `--metrics` | `<inference folder>/metrics_<name><key>` | `metrics.csv` or its folder |
| `--n-aug` | 5 | Augmented draws per image (0: none) |
| `--aug-source` | `CT` | Training modality: its images are augmented, the others are compared with them |
| `--trainer-config` | `nnunet_trainer_config` of the config | SmaugLab augmentation config |
| `--norm` | `nnunet` | `nnunet` (from `plans.json`), `zscore` (per image) or `none` |
| `--plans` | from the nnUNet results folder | `plans.json` used by `--norm nnunet` |
| `--patch-size` | patch size of the plans | Tile size for the GPU augmentation |
| `--batch-size` | 2 | Tiles augmented together (lower it on GPU OOM) |
| `--samples` | 512 | Quantiles kept per (image, draw, structure) |
| `--shell-mm` | 5 | Thickness of the surroundings shell |
| `--bin-width` | 0.1 (25 with `--norm none`) | Bin width for entropy, uniformity, overlap |
| `--hist-bins` | 64 | Histogram bins given to the PCA |
| `--pca-components` | 5 | Principal components kept |
| `--device` | `$SMAUGBENCH_DEVICE`, else cuda/cpu | Device of the augmentations |
| `--seed` | 42 | Random seed |
| `--max-images` | all | Only process the first N images of each inference folder |
