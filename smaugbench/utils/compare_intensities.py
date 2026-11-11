"""
Relate the intensity, size and frequency of each labelled structure to the segmentation
performance, on MR images, CT images and GPU-augmented images generated with the
SmaugLab trainer config.

The script reads the same pipeline config as `scripts/inference.sh` and works on the
inference folder that script already prepared:

    ${SMAUGBENCH_DATA}/inference/Dataset${dataset_id}_${dataset_name}${inference_supplementary_key}/
        images/<stem>_0000.nii.gz                     (<stem> ends with _MR or _CT)
        labels/<stem>.nii.gz
        metrics_<nnunet_folder_name><key>/metrics.csv (scores per image and structure)

For every image/label pair, each structure listed in the `LABELS` field of the dataset
JSON (`data_json`) is measured on:

    - the image itself (source `MR` or `CT`, from the filename suffix)
    - `--n-aug` augmented versions of the image (source `MR_aug` / `CT_aug`), generated
      with the GPU section of `nnunet_trainer_config`, built exactly like
      `nnUNetTrainerDAExtGPU` builds it, and with the label map passed as the mask so
      that the in-segmentation transforms behave as during training.

The measurements are first-order radiomics, a PCA of the intensity histograms, the
contrast with the surroundings, the size and the frequency of the structure. They are
then correlated (Spearman) with the scores of metrics.csv, separately for MR and CT.

Images are normalized before the measurements (and before augmentation, as in training).
By default the normalization stored in the trained model's plans.json is used
(CTNormalization or ZScoreNormalization), i.e. the intensities the network actually
sees. The CPU section of the trainer config (spatial transforms) is not applied: it only
moves voxels around and does not change intensities.

Outputs (in `<inference folder>/intensities_<nnunet_folder_name><supplementary_key>/` by default):
    intensity_stats.csv              one row per (image, source, draw, structure): all measurements
    intensity_summary.csv            one row per (source, structure), pooled over images
    intensity_distances.csv          one row per (structure, source pair): Wasserstein distance
                                     and coverage of one distribution by the other
    histogram_pca*.csv               PCA scores, loadings and explained variance
    instance_features.csv            one row per (image, structure): features + scores
    structure_features.csv           one row per (modality, structure): mean features + mean scores
    correlations_structure_level.csv Spearman correlation feature/score across structures
    correlations_instance_level.csv  same across (image, structure) pairs and inside each structure
    *.png                            boxplots, PCA scatter, correlation heatmaps and scatter plots
    config.json                      copy of the input config

Every step is described in smaugbench/utils/README_compare_intensities.md.
"""

import argparse
import itertools
import json
import os
import shutil
import textwrap
import warnings
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SMAUGBENCH_ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ('MR', 'CT')
PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 95, 99)
# Voxels kept around the labelled structures when cropping (context for blur/convolution-like augmentations)
CROP_MARGIN = 16
# Quantiles describing the pooled distribution of a structure in a source
POOLED_QUANTILES = 2048
# Guard against a --bin-width far too small for the intensity range
MAX_BINS = 1_000_000
# Points needed to compute a correlation
MIN_CORRELATION_POINTS = 5

# Feature families
FREQUENCY_FEATURES = ['frequency', 'voxel_share']
SIZE_FEATURES = ['n_voxels', 'volume_mm3', 'equivalent_diameter_mm', 'bbox_max_extent_mm', 'bbox_fill', 'boundary_fraction']
SURROUND_FEATURES = ['surround_mean', 'surround_std', 'contrast', 'abs_contrast', 'cnr', 'surround_overlap', 'surround_wasserstein']
RADIOMICS_FEATURES = (['mean', 'std', 'variance', 'minimum', 'maximum', 'range'] + [f'p{p:02d}' for p in PERCENTILES] +
                      ['iqr', 'mad', 'rmad', 'rms', 'energy', 'total_energy', 'entropy', 'uniformity', 'skewness', 'kurtosis'])
INTENSITY_FEATURES = RADIOMICS_FEATURES + SURROUND_FEATURES
METRICS_META = ('reference', 'prediction', 'label', 'EmptyRef', 'EmptyPred')

# Figure colours: one hue per modality, its lighter step for the augmented images
SOURCE_COLORS = {'MR': '#2a78d6', 'MR_aug': '#9ec5f4', 'CT': '#eb6834', 'CT_aug': '#f5b59a'}
DIVERGING = ('#1c5cab', '#c23a3a')
SURFACE, INK, INK_MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        epilog=textwrap.dedent('''
            Examples:
            smaugbench_compare_intensities smaugbench/configs/amos22_nnUNetTrainerDAExtGPU_OSSTA.json
            smaugbench_compare_intensities config.json --n-aug 10 --aug-source CT --device cuda:0
            smaugbench_compare_intensities config.json --metrics /path/to/metrics.csv -o /path/to/output
        '''),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('config', type=str, help='Pipeline config JSON (same as the one given to run_pipeline.sh / inference.sh).')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output folder (default: <inference folder>/intensities_<nnunet_folder_name><supplementary_key>).')
    parser.add_argument('--metrics', type=str, default=None,
                        help='metrics.csv (or the folder containing it) of the model to relate the features to '
                             '(default: <inference folder>/metrics_<nnunet_folder_name><supplementary_key>).')
    parser.add_argument('--n-aug', type=int, default=5, help='Number of augmented draws per image (default: 5). 0 disables augmentation.')
    parser.add_argument('--aug-source', nargs='+', choices=MODALITIES, default=list(MODALITIES),
                        help='Modalities of the images that are augmented (default: MR CT). '
                             'Falls back to the available modalities if none of them is present.')
    parser.add_argument('--trainer-config', type=str, default=None,
                        help='Override the SmaugLab config (default: nnunet_trainer_config of the input config).')
    parser.add_argument('--norm', choices=['nnunet', 'zscore', 'none'], default='nnunet',
                        help='Intensity normalization applied to every image before extraction and augmentation. '
                             '"nnunet" uses the scheme and statistics of the trained model plans.json (falls back to '
                             '"zscore" if the plans cannot be found), "zscore" normalizes each image on its own, '
                             '"none" keeps raw intensities (default: nnunet).')
    parser.add_argument('--plans', type=str, default=None,
                        help='Path to the plans.json used by --norm nnunet (default: the one in the nnUNet results folder).')
    parser.add_argument('--patch-size', type=int, nargs=3, default=None,
                        help='Size of the tiles augmented on the GPU (default: patch_size of the plans.json, else 128 128 128). '
                             'Each volume is cropped around its labels and split into tiles of this size, as the '
                             'trainer only ever augments patches; augmenting whole volumes does not fit in GPU memory.')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Number of tiles augmented together; lower it if the GPU runs out of memory (default: 2).')
    parser.add_argument('--samples', type=int, default=512,
                        help='Evenly spaced quantiles kept per (image, draw, structure) to describe its intensity '
                             'distribution (histograms, PCA and distances) (default: 512).')
    parser.add_argument('--shell-mm', type=float, default=5.0,
                        help='Thickness in mm of the shell around a structure used as its surroundings (default: 5).')
    parser.add_argument('--bin-width', type=float, default=None,
                        help='Bin width of the discretized intensities (entropy, uniformity, overlap with the surroundings) '
                             '(default: 0.1 for normalized intensities, 25 with --norm none).')
    parser.add_argument('--hist-bins', type=int, default=64, help='Bins of the histograms given to the PCA (default: 64).')
    parser.add_argument('--pca-components', type=int, default=5, help='Principal components kept (default: 5).')
    parser.add_argument('--device', type=str, default=os.environ.get('SMAUGBENCH_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'),
                        help='Device used for the GPU augmentations (default: $SMAUGBENCH_DEVICE, else cuda if available, else cpu).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42).')
    parser.add_argument('--max-images', type=int, default=None, help='Only process the first N images (for quick tests).')
    args = parser.parse_args()

    compare_intensities(**vars(args))


# --- config -------------------------------------------------------------------------


def _resolve_json(path, fallback_dir):
    """Return `path` if it exists, else `fallback_dir/<basename>.json` (same rule as inference.sh)."""
    if path and os.path.isfile(path):
        return Path(path)
    name = Path(path).name
    if not name.endswith('.json'):
        name += '.json'
    candidate = fallback_dir / name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f'Could not find {path} (also tried {candidate})')


def load_pipeline_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    for key in ('data_json', 'dataset_id', 'dataset_name', 'nnunet_trainer', 'nnunet_plans', 'configuration', 'nnunet_folder_name'):
        if config.get(key) in (None, ''):
            raise ValueError(f'config: .{key} is required')
    config.setdefault('inference_supplementary_key', '')
    if config['inference_supplementary_key'] is None:
        config['inference_supplementary_key'] = ''
    return config


# --- normalization ------------------------------------------------------------------


def _load_plans_configuration(plans_path, configuration):
    """(plans, configuration dict with inheritance resolved) from a nnUNet plans.json."""
    with open(plans_path) as f:
        plans = json.load(f)
    configs = plans['configurations']
    conf = dict(configs[configuration])
    # nnUNet configurations can inherit from each other
    while 'inherits_from' in conf:
        parent = dict(configs[conf.pop('inherits_from')])
        parent.update(conf)
        conf = parent
    return plans, conf


def load_plans_patch_size(plans_path, configuration):
    """patch_size of the configuration, or None if the plans cannot be read."""
    if plans_path is None or not Path(plans_path).is_file():
        return None
    _, conf = _load_plans_configuration(plans_path, configuration)
    patch_size = conf.get('patch_size')
    return [int(p) for p in patch_size] if patch_size and len(patch_size) == 3 else None


def _load_plans_normalization(plans_path, configuration):
    """(scheme, use_mask_for_norm, intensity properties) of channel 0 from a nnUNet plans.json."""
    plans, conf = _load_plans_configuration(plans_path, configuration)
    scheme = conf['normalization_schemes'][0]
    use_mask = bool(conf.get('use_mask_for_norm', [False])[0])
    props = plans.get('foreground_intensity_properties_per_channel', {}).get('0')
    return scheme, use_mask, props


def build_normalizer(norm, plans_path, configuration):
    """Return (description, function image -> normalized image)."""
    if norm == 'none':
        return 'none', lambda x: x

    def zscore(x, use_mask=False):
        values = x[x > 0] if use_mask and (x > 0).any() else x
        mean, std = values.mean(), values.std()
        out = (x - mean) / max(std, 1e-8)
        if use_mask:
            out[x <= 0] = 0
        return out

    if norm == 'nnunet':
        if plans_path is None or not Path(plans_path).is_file():
            warnings.warn(f'plans.json not found ({plans_path}): falling back to --norm zscore')
            norm = 'zscore'
        else:
            scheme, use_mask, props = _load_plans_normalization(plans_path, configuration)
            if scheme == 'CTNormalization':
                if props is None:
                    raise ValueError(f'{plans_path} uses CTNormalization but has no foreground intensity properties')
                lo, hi = props['percentile_00_5'], props['percentile_99_5']
                mean, std = props['mean'], max(props['std'], 1e-8)
                return (f'nnunet CTNormalization (clip [{lo:.1f}, {hi:.1f}], mean {mean:.1f}, std {std:.1f})',
                        lambda x: (np.clip(x, lo, hi) - mean) / std)
            if scheme == 'ZScoreNormalization':
                return f'nnunet ZScoreNormalization (use_mask_for_norm={use_mask})', lambda x: zscore(x, use_mask)
            if scheme in ('NoNormalization', 'RescaleTo01Normalization', 'RGBTo01Normalization'):
                warnings.warn(f'{scheme} is not reimplemented here: falling back to --norm zscore')
                norm = 'zscore'
            else:
                warnings.warn(f'Unknown normalization scheme {scheme}: falling back to --norm zscore')
                norm = 'zscore'

    return 'per-image zscore', lambda x: zscore(x)


# --- augmentation -------------------------------------------------------------------


def build_augmenter(trainer_config_path, device):
    """GPU pipeline built exactly like nnUNetTrainerDAExtGPU.__init__ builds it."""
    from smauglab.config import load_config
    from smauglab.registry import Backend
    from smauglab.transforms.build import build_gpu_pipeline
    from smauglab.transforms.gpu.base import AugmentationSequentialCustom

    config = load_config(str(trainer_config_path))
    if not config.names(Backend.GPU):
        warnings.warn(f'{trainer_config_path} has no GPU augmentation: augmented sources are skipped')
        return None
    return AugmentationSequentialCustom(
        *build_gpu_pipeline(
            config.section(Backend.GPU),
            mode=config.pipeline_mode(),
            options=config.pipeline_options('random_choose'),
            source=config.source,
            order_source=config.order_source(),
        ),
        data_keys=['input', 'mask'],
        same_on_batch=True if config.same_on_batch() else None,
    ).to(device)


def crop_to_labels(image, label, margin):
    """Crop image and label to the bounding box of the labelled voxels, plus `margin` voxels."""
    coords = np.argwhere(label > 0)
    if coords.size == 0:
        return image, label
    lo = np.maximum(coords.min(axis=0) - margin, 0)
    hi = np.minimum(coords.max(axis=0) + margin + 1, label.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    return image[box], label[box]


def _tile_starts(size, patch):
    """Start indices of non-overlapping tiles covering `size` (the volume is padded to a multiple of `patch`)."""
    return list(range(0, max(size, 1), patch))


@torch.no_grad()
def augment(augmenter, image, label, device, patch_size, batch_size):
    """Augment one normalized image [D,H,W] with its label map as mask, like train_step does.

    The trainer never augments a whole volume, only patches of `patch_size` (batched by
    the dataloader), and a whole volume does not fit in GPU memory anyway. The volume is
    therefore zero-padded to a multiple of `patch_size`, split into non-overlapping tiles
    that are augmented independently (each tile draws its own random parameters, like
    each sample of a training batch), and reassembled.
    """
    patch = [min(p, s) for p, s in zip(patch_size, image.shape)]
    padded_shape = [int(np.ceil(s / p)) * p for s, p in zip(image.shape, patch)]
    pad = [(0, ps - s) for ps, s in zip(padded_shape, image.shape)]
    image_p = np.pad(image.astype(np.float32), pad, mode='constant', constant_values=0)
    label_p = np.pad(label.astype(np.float32), pad, mode='constant', constant_values=0)

    out_image = np.empty_like(image_p)
    out_label = np.empty(label_p.shape, dtype=np.int32)
    boxes = [tuple(slice(a, a + p) for a, p in zip(starts, patch))
             for starts in itertools.product(*(_tile_starts(s, p) for s, p in zip(padded_shape, patch)))]

    device_type = torch.device(device).type
    for i in range(0, len(boxes), batch_size):
        batch = boxes[i:i + batch_size]
        data = torch.from_numpy(np.ascontiguousarray(np.stack([image_p[b] for b in batch])))[:, None].to(device)
        target = torch.from_numpy(np.ascontiguousarray(np.stack([label_p[b] for b in batch])))[:, None].to(device)
        if device_type == 'cuda':
            with torch.autocast(device_type, enabled=True):
                data, target = augmenter(data, target)
        else:
            data, target = augmenter(data, target)
        data = data[:, 0].float().cpu().numpy()
        target = torch.round(target[:, 0].float()).cpu().numpy().astype(np.int32)
        for j, b in enumerate(batch):
            out_image[b] = data[j]
            out_label[b] = target[j]
        del data, target

    if np.isnan(out_image).any():
        warnings.warn('NaNs in an augmented image: they are ignored')
    crop = tuple(slice(0, s) for s in image.shape)
    return out_image[crop], out_label[crop]


# --- per-structure measurements -----------------------------------------------------


def _sorted_quantiles(sorted_values, probs):
    """Linearly interpolated quantiles of an already sorted 1D array."""
    pos = np.asarray(probs, dtype=np.float64) * (sorted_values.size - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, sorted_values.size - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _bin_index(values, bin_width):
    """Index of the fixed-width bin of each value (bin edges are multiples of `bin_width`, as in pyradiomics)."""
    index = np.floor(values / bin_width).astype(np.int64)
    if index.max() - index.min() > MAX_BINS:
        raise ValueError(f'--bin-width {bin_width} gives more than {MAX_BINS} bins for intensities in '
                         f'[{values.min():.4g}, {values.max():.4g}]: use a larger --bin-width.')
    return index


def first_order(sorted_values, bin_width, voxel_volume):
    """First-order radiomics of a structure (definitions of pyradiomics `firstorder` / IBSI).

    `sorted_values` are the intensities of the structure, sorted, in float64.
    """
    v = sorted_values
    n = v.size
    mean = v.mean()
    centered = v - mean
    variance = float(np.mean(centered ** 2))
    std = np.sqrt(variance)
    pct = dict(zip(PERCENTILES, _sorted_quantiles(v, np.array(PERCENTILES) / 100)))
    # Values between the 10th and 90th percentiles (robust mean absolute deviation)
    inner = v[np.searchsorted(v, pct[10], 'left'):np.searchsorted(v, pct[90], 'right')]
    # Discretized histogram (entropy, uniformity)
    index = _bin_index(v, bin_width)
    prob = np.bincount(index - index[0])
    prob = prob[prob > 0] / n
    energy = float(np.sum(v ** 2))

    stats = {'mean': float(mean), 'std': float(std), 'variance': variance,
             'minimum': float(v[0]), 'maximum': float(v[-1]), 'range': float(v[-1] - v[0])}
    stats.update({f'p{p:02d}': float(q) for p, q in pct.items()})
    stats.update({
        'iqr': float(pct[75] - pct[25]),
        'mad': float(np.mean(np.abs(centered))),
        'rmad': float(np.mean(np.abs(inner - inner.mean()))) if inner.size else np.nan,
        'rms': float(np.sqrt(energy / n)),
        'energy': energy,
        'total_energy': energy * voxel_volume,
        'entropy': float(-np.sum(prob * np.log2(prob))),
        'uniformity': float(np.sum(prob ** 2)),
        'skewness': float(np.mean(centered ** 3) / std ** 3) if std > 0 else 0.0,
        'kurtosis': float(np.mean(centered ** 4) / variance ** 2) if std > 0 else 0.0,
    })
    return stats


def surroundings(sorted_values, shell_values, bin_width, probs):
    """Difference between the intensities of a structure and those of the shell around it."""
    if shell_values.size == 0:
        return dict.fromkeys(SURROUND_FEATURES, np.nan)
    shell = np.sort(shell_values.astype(np.float64))
    mean_in, mean_out = sorted_values.mean(), shell.mean()
    var_in, var_out = sorted_values.var(), shell.var()
    contrast = mean_in - mean_out
    pooled_std = np.sqrt((var_in + var_out) / 2)
    # Histogram intersection of the two distributions (1: identical, 0: separable from the intensity alone)
    idx_in, idx_out = _bin_index(sorted_values, bin_width), _bin_index(shell, bin_width)
    low = min(idx_in[0], idx_out[0])
    n_bins = max(idx_in[-1], idx_out[-1]) - low + 1
    if n_bins > MAX_BINS:
        raise ValueError(f'--bin-width {bin_width} gives more than {MAX_BINS} bins: use a larger --bin-width.')
    hist_in = np.bincount(idx_in - low, minlength=n_bins) / idx_in.size
    hist_out = np.bincount(idx_out - low, minlength=n_bins) / idx_out.size
    return {
        'surround_mean': float(mean_out),
        'surround_std': float(np.sqrt(var_out)),
        'contrast': float(contrast),
        'abs_contrast': float(abs(contrast)),
        'cnr': float(abs(contrast) / pooled_std) if pooled_std > 0 else np.nan,
        'surround_overlap': float(np.minimum(hist_in, hist_out).sum()),
        # 1-Wasserstein distance = mean absolute difference between the two quantile functions
        'surround_wasserstein': float(np.mean(np.abs(_sorted_quantiles(sorted_values, probs) - _sorted_quantiles(shell, probs)))),
    }


def measure_structures(image, label, label_values, spacing, shell_mm, bin_width, n_quantiles, with_size):
    """Yield (label value, features, quantiles) for every structure of `label_values` present in `label`.

    features: first-order radiomics, contrast with the surroundings and, if `with_size`, size.
    quantiles: `n_quantiles` evenly spaced quantiles of the structure intensities, the
    compact description of its distribution used for the histograms and the distances.
    """
    from scipy import ndimage

    spacing = np.asarray(spacing, dtype=np.float64)
    voxel_volume = float(np.prod(spacing))
    pad = np.ceil(shell_mm / spacing).astype(int) + 1
    probs = np.linspace(0, 1, n_quantiles)
    objects = ndimage.find_objects(label, max_label=int(label_values.max()))
    for value in label_values:
        bbox = objects[value - 1]
        if bbox is None:
            continue
        # Work on the bounding box of the structure, enlarged to contain the shell
        box = tuple(slice(max(s.start - p, 0), min(s.stop + p, n)) for s, p, n in zip(bbox, pad, label.shape))
        img = image[box]
        mask = label[box] == value
        finite = np.isfinite(img)
        values = np.sort(img[mask & finite].astype(np.float64))
        if values.size == 0:
            continue
        # Shell: voxels outside the structure at most `shell_mm` away from it (whatever their label)
        distance = ndimage.distance_transform_edt(~mask, sampling=spacing)
        shell_values = img[(distance > 0) & (distance <= shell_mm) & finite]

        features = {'n_voxels': int(values.size)}
        features.update(first_order(values, bin_width, voxel_volume))
        features.update(surroundings(values, shell_values, bin_width, probs))
        if with_size:
            n = int(mask.sum())
            extent = np.array([s.stop - s.start for s in bbox])
            features.update({
                'volume_mm3': n * voxel_volume,
                'equivalent_diameter_mm': float((6 * n * voxel_volume / np.pi) ** (1 / 3)),
                'bbox_max_extent_mm': float((extent * spacing).max()),
                'bbox_fill': float(n / np.prod(extent)),
                # Share of the structure voxels touching the outside (thin/small structures are close to 1)
                'boundary_fraction': float(1 - ndimage.binary_erosion(mask, border_value=1).sum() / n),
            })
        yield int(value), features, _sorted_quantiles(values, probs).astype(np.float32)


def wasserstein(a, b):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


def coverage(target, reference):
    """Fraction of `target` voxels falling inside the [p1, p99] range of `reference`."""
    lo, hi = np.percentile(reference, [1, 99])
    return float(((target >= lo) & (target <= hi)).mean())


# --- histogram PCA ------------------------------------------------------------------


def histograms_from_quantiles(quantiles, n_bins):
    """Normalized histograms [n, n_bins] on a common intensity range, and the bin edges.

    The range is the [0.5, 99.5] percentile range of all the stored quantiles; what falls
    outside is counted in the first/last bin.
    """
    lo, hi = np.percentile(quantiles, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, n_bins + 1)
    hists = np.empty((len(quantiles), n_bins), dtype=np.float64)
    for i, q in enumerate(quantiles):
        cdf = np.searchsorted(q, edges, side='right') / q.size
        cdf[0], cdf[-1] = 0, 1
        hists[i] = np.diff(cdf)
    return hists, edges


def pca(x, n_components):
    """PCA of the rows of `x`: (scores, components, explained variance ratio, mean)."""
    mean = x.mean(axis=0)
    centered = x - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / max(len(x) - 1, 1))
    order = np.argsort(eigenvalues)[::-1][:n_components]
    components = eigenvectors[:, order].T
    # Deterministic sign: the largest loading of each component is positive
    components *= np.sign(components[np.arange(len(components)), np.abs(components).argmax(axis=1)])[:, None]
    ratio = eigenvalues[order] / max(eigenvalues.sum(), 1e-12)
    return centered @ components.T, components, ratio, mean


# --- metrics and correlations -------------------------------------------------------


def load_metrics(path):
    """(metrics table with `subject` and `structure` columns, metric names) from a metrics.csv or its folder."""
    path = Path(path)
    if path.is_dir():
        path = path / 'metrics.csv'
    if not path.is_file():
        return None, []
    df = pd.read_csv(path)
    missing = {'reference', 'label'} - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing column(s) {sorted(missing)}')
    if 'EmptyRef' in df.columns:
        # No ground truth for this structure: nothing to relate the score to
        df = df[~df['EmptyRef'].astype(str).str.lower().eq('true')]
    metrics = [c for c in df.columns if c not in METRICS_META and pd.api.types.is_numeric_dtype(df[c])]
    df = df.assign(subject=df['reference'].map(lambda r: Path(str(r)).name.split('.nii')[0]), structure=df['label'])
    df[metrics] = df[metrics].replace([np.inf, -np.inf], np.nan)
    return df.groupby(['subject', 'structure'], as_index=False)[metrics].mean(), metrics


def feature_group(name):
    """Family of a feature column (used to read the correlation tables)."""
    if name in FREQUENCY_FEATURES:
        return 'frequency'
    if name in SIZE_FEATURES:
        return 'size'
    if '_to_' in name or name.startswith('coverage_by_'):
        return 'augmentation_shift' if name.startswith('aug_') else 'cross_source'
    base = name.split('_', 1)[1] if name.split('_', 1)[0] in ('raw', 'aug', 'augdelta') else name
    if base in SURROUND_FEATURES:
        return 'surroundings'
    if base.startswith('pc'):
        return 'histogram_pca'
    return 'radiomics'


def benjamini_hochberg(p):
    """Benjamini-Hochberg false discovery rate q-values (NaNs are left untouched)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.shape, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    if ok.size:
        order = ok[np.argsort(p[ok])]
        scaled = p[order] * ok.size / np.arange(1, ok.size + 1)
        q[order] = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1)
    return q


def _spearman(x, y):
    from scipy.stats import spearmanr
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < MIN_CORRELATION_POINTS or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return None
    r, p = spearmanr(x[ok], y[ok])
    return int(ok.sum()), float(r), float(p)


def correlate(table, features, metrics, per_structure):
    """Spearman correlation of every feature with every metric, for each modality.

    If `per_structure` (instance level), the correlation is computed over all the
    (image, structure) pairs (`structure` = "all"), inside each structure, and the
    per-structure coefficients are averaged (`structure` = "mean_within").
    """
    rows = []
    for modality, sub in table.groupby('modality'):
        groups = [('all', sub)]
        if per_structure:
            groups += [(name, g) for name, g in sub.groupby('structure', sort=False)]
        for metric in metrics:
            for feature in features:
                within = []
                for name, g in groups:
                    res = _spearman(g[feature].to_numpy(dtype=np.float64), g[metric].to_numpy(dtype=np.float64))
                    if res is None:
                        continue
                    rows.append({'modality': modality, 'metric': metric, 'structure': name, 'feature': feature,
                                 'group': feature_group(feature), 'n': res[0], 'spearman_r': res[1], 'p_value': res[2]})
                    if name != 'all':
                        within.append(res[1])
                if within:
                    rows.append({'modality': modality, 'metric': metric, 'structure': 'mean_within', 'feature': feature,
                                 'group': feature_group(feature), 'n': len(within),
                                 'spearman_r': float(np.mean(within)), 'p_value': np.nan})
    out = pd.DataFrame(rows, columns=['modality', 'metric', 'structure', 'feature', 'group', 'n', 'spearman_r', 'p_value'])
    if not out.empty:
        out['q_value'] = out.groupby(['modality', 'metric', 'structure'])['p_value'].transform(benjamini_hochberg)
        out = out.sort_values(['modality', 'metric', 'structure', 'spearman_r'], key=lambda c: -c.abs() if c.name == 'spearman_r' else c)
    if not per_structure:
        out = out.drop(columns='structure')
    return out


# --- tables -------------------------------------------------------------------------


def add_pca_features(stats, quantiles, sources, n_bins, n_components, output):
    """Histogram PCA of every (image, draw, structure) + distances between sources. Returns the PC column names."""
    hists, edges = histograms_from_quantiles(quantiles, n_bins)
    n_components = max(1, min(n_components, n_bins, len(hists) - 1))
    scores, components, ratio, mean = pca(hists, n_components)
    pcs = [f'pc{i + 1}' for i in range(n_components)]
    stats[pcs] = scores

    loadings = pd.DataFrame({'bin_low': edges[:-1], 'bin_high': edges[1:], 'mean_histogram': mean})
    for name, component in zip(pcs, components):
        loadings[name] = component
    loadings.to_csv(output / 'histogram_pca_components.csv', index=False)
    pd.DataFrame({'component': pcs, 'explained_variance_ratio': ratio}).to_csv(output / 'histogram_pca_variance.csv', index=False)
    stats[['subject', 'modality', 'source', 'draw', 'label', 'structure'] + pcs].to_csv(output / 'histogram_pca.csv', index=False)

    # Distance of every instance to the centroid of its structure in its own source
    centroids = stats.groupby(['source', 'label'])[pcs].mean()
    own = centroids.reindex(pd.MultiIndex.from_frame(stats[['source', 'label']])).to_numpy()
    stats['pc_centroid_dist'] = np.linalg.norm(scores - own, axis=1)

    # Raw instances against the distribution of the same structure in every other source
    pooled = {}
    for (source, label), idx in stats.groupby(['source', 'label']).indices.items():
        values = np.sort(quantiles[idx].ravel())
        pooled[(source, label)] = (_sorted_quantiles(values, np.linspace(0, 1, POOLED_QUANTILES)), *np.percentile(values, [1, 99]))
    cross = {f'{kind}_{other}': np.full(len(stats), np.nan) for other in sources
             for kind in ('wasserstein_to', 'coverage_by', 'pc_dist_to')}
    for i in np.flatnonzero((stats['draw'] == 0).to_numpy()):
        source, label = stats['source'].iat[i], stats['label'].iat[i]
        for other in sources:
            if other == source or (other, label) not in pooled:
                continue
            reference, lo, hi = pooled[(other, label)]
            cross[f'wasserstein_to_{other}'][i] = wasserstein(quantiles[i], reference)
            cross[f'coverage_by_{other}'][i] = np.mean((quantiles[i] >= lo) & (quantiles[i] <= hi))
            cross[f'pc_dist_to_{other}'][i] = np.linalg.norm(scores[i] - centroids.loc[(other, label)].to_numpy())
    for name, column in cross.items():
        stats[name] = column
    return pcs, ratio


def build_instance_table(stats, quantiles, pcs, sources):
    """One row per (image, structure): size, raw_*, aug_* (mean over the draws), augdelta_* and cross-source features."""
    keys = ['subject', 'modality', 'label', 'structure']
    intensity = [c for c in INTENSITY_FEATURES if c in stats.columns] + pcs + ['pc_centroid_dist']
    cross = [c for c in stats.columns if any(c == f'{kind}_{s}' for s in sources for kind in ('wasserstein_to', 'coverage_by', 'pc_dist_to'))]
    cross = [c for c in cross if stats[c].notna().any()]

    raw = stats[stats['draw'] == 0]
    parts = [raw[keys + SIZE_FEATURES + cross], raw[intensity].add_prefix('raw_')]

    aug = stats[stats['draw'] > 0]
    if not aug.empty:
        # How far each draw moves the structure away from the raw image
        raw_row = pd.Series(raw.index, index=pd.MultiIndex.from_frame(raw[['subject', 'label']]))
        match = raw_row.reindex(pd.MultiIndex.from_frame(aug[['subject', 'label']])).to_numpy()
        ok = ~np.isnan(match)
        aug, match = aug[ok].copy(), match[ok].astype(int)
        aug['to_raw_wasserstein'] = np.mean(np.abs(quantiles[aug.index] - quantiles[match]), axis=1)
        aug['to_raw_pc_dist'] = np.linalg.norm(aug[pcs].to_numpy() - stats.loc[match, pcs].to_numpy(), axis=1)
        mean_aug = aug.groupby(['subject', 'label'])[intensity + ['to_raw_wasserstein', 'to_raw_pc_dist']].mean()
        mean_aug = mean_aug.reindex(pd.MultiIndex.from_frame(raw[['subject', 'label']])).set_axis(raw.index)
        parts += [mean_aug.add_prefix('aug_'), (mean_aug[intensity] - raw[intensity]).add_prefix('augdelta_')]
    return pd.concat(parts, axis=1).reset_index(drop=True)


def build_structure_table(instances, stats, pcs, n_images, metrics):
    """One row per (modality, structure): mean of the instance features, frequency and separability in the PCA space."""
    keys = ['modality', 'label', 'structure']
    features = [c for c in instances.columns if c not in keys + ['subject']]
    table = instances.groupby(keys, as_index=False)[features].mean()
    count = instances.groupby(keys)['subject'].nunique().reindex(pd.MultiIndex.from_frame(table[keys])).to_numpy()
    total = table['modality'].map(n_images).to_numpy()
    voxels = instances.groupby(keys)['n_voxels'].sum().reindex(pd.MultiIndex.from_frame(table[keys])).to_numpy()
    table.insert(3, 'n_images', count)
    table.insert(4, 'frequency', count / total)
    table.insert(5, 'voxel_share', voxels / table.assign(v=voxels).groupby('modality')['v'].transform('sum').to_numpy())
    if metrics:
        scored = instances.groupby(keys)[metrics].count().reindex(pd.MultiIndex.from_frame(table[keys])).to_numpy()
        for i, metric in enumerate(metrics):
            table[f'{metric}_n'] = scored[:, i]

    # Separability: distance to the closest other structure, relative to the spread of the structure itself
    centroids = stats.groupby(['source', 'label'])[pcs].mean()
    for prefix, suffix in (('raw', ''), ('aug', '_aug')):
        if f'{prefix}_pc_centroid_dist' not in table.columns:
            continue
        nearest = np.full(len(table), np.nan)
        for i, (modality, label) in enumerate(zip(table['modality'], table['label'])):
            source = f'{modality}{suffix}'
            if source not in centroids.index.get_level_values(0) or (source, label) not in centroids.index:
                continue
            others = centroids.loc[source].drop(index=label)
            if len(others):
                nearest[i] = np.linalg.norm(others.to_numpy() - centroids.loc[(source, label)].to_numpy(), axis=1).min()
        table[f'{prefix}_pc_nearest_structure_dist'] = nearest
        table[f'{prefix}_pc_separability'] = nearest / table[f'{prefix}_pc_centroid_dist'].replace(0, np.nan)
    return table


# --- main ---------------------------------------------------------------------------


def compare_intensities(config, output=None, metrics=None, n_aug=5, aug_source=MODALITIES, trainer_config=None, norm='nnunet',
                        plans=None, patch_size=None, batch_size=2, samples=512, shell_mm=5.0, bin_width=None, hist_bins=64,
                        pca_components=5, device='cpu', seed=42, max_images=None):
    data_root = os.environ.get('SMAUGBENCH_DATA')
    if not data_root:
        raise EnvironmentError('Please set the SMAUGBENCH_DATA environment variable.')

    config_path = Path(config).resolve()
    cfg = load_pipeline_config(config_path)
    key = cfg['inference_supplementary_key']
    src_dataset = f"Dataset{cfg['dataset_id']}_{cfg['dataset_name']}"
    inference_dir = Path(data_root) / 'inference' / f'{src_dataset}{key}'
    images_dir, labels_dir = inference_dir / 'images', inference_dir / 'labels'
    for d in (images_dir, labels_dir):
        if not d.is_dir():
            raise FileNotFoundError(f'{d} does not exist: run scripts/inference.sh with this config first.')

    # Structures of the dataset
    data_json = _resolve_json(cfg['data_json'], SMAUGBENCH_ROOT / 'datasets')
    with open(data_json) as f:
        structures = {int(k): v for k, v in json.load(f)['LABELS'].items() if int(k) != 0}
    label_values = np.array(sorted(structures))

    # Normalization (from the trained model's plans by default)
    if plans is None:
        plans = (Path(data_root) / 'nnUNet' / 'results' / cfg['nnunet_folder_name'] / src_dataset /
                 f"{cfg['nnunet_trainer']}__{cfg['nnunet_plans']}__{cfg['configuration']}" / 'plans.json')
    norm_desc, normalize = build_normalizer(norm, plans, cfg['configuration'])
    if patch_size is None:
        patch_size = load_plans_patch_size(plans, cfg['configuration']) or [128, 128, 128]
    if bin_width is None:
        # Normalized intensities are in standard deviations; raw ones use the pyradiomics default
        bin_width = 25.0 if norm_desc == 'none' else 0.1

    # Image/label pairs
    pairs = []
    for img in sorted(images_dir.glob('*_0000.nii.gz')):
        stem = img.name[:-len('_0000.nii.gz')]
        modality = stem.rsplit('_', 1)[-1]
        lab = labels_dir / f'{stem}.nii.gz'
        if modality not in MODALITIES:
            warnings.warn(f'Skipping {img.name}: name does not end with _MR or _CT')
            continue
        if not lab.is_file():
            warnings.warn(f'Skipping {img.name}: missing label {lab}')
            continue
        pairs.append((stem, modality, img, lab))
    if max_images is not None:
        pairs = pairs[:max_images]
    if not pairs:
        raise FileNotFoundError(f'No <stem>_MR/_CT image/label pairs found in {inference_dir}')

    present = sorted({m for _, m, _, _ in pairs})
    aug_source = [m for m in aug_source if m in present]
    if not aug_source:
        aug_source = present
        warnings.warn(f'None of the requested --aug-source modalities are present: augmenting {aug_source} instead')

    # Augmentation pipeline
    augmenter = None
    if n_aug > 0:
        trainer_config = trainer_config or cfg.get('nnunet_trainer_config')
        if not trainer_config:
            warnings.warn('No nnunet_trainer_config in the config and no --trainer-config: augmented sources are skipped')
        else:
            trainer_config = _resolve_json(trainer_config, SMAUGBENCH_ROOT / 'smauglab_configs')
            torch.manual_seed(seed)
            augmenter = build_augmenter(trainer_config, device)

    # Performance metrics of the model described by the config
    metrics_path = Path(metrics) if metrics else inference_dir / f"metrics_{cfg['nnunet_folder_name']}{key}"
    scores, metric_names = load_metrics(metrics_path)
    if scores is None:
        warnings.warn(f'No metrics.csv found in {metrics_path}: the correlations with the performance are skipped')

    output = Path(output) if output else inference_dir / f"intensities_{cfg['nnunet_folder_name']}{key}"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output / 'config.json')

    print(f'Inference folder: {inference_dir}')
    print(f'Dataset JSON: {data_json} ({len(structures)} structures)')
    print(f'Images: {len(pairs)} ({", ".join(f"{m}: {sum(p[1] == m for p in pairs)}" for m in present)})')
    print(f'Normalization: {norm_desc}')
    print(f'Augmentation: {f"{n_aug} draws of {aug_source} with {trainer_config} on {device}, tiles {patch_size} x {batch_size}" if augmenter else "none"}')
    print(f'Metrics: {f"{metrics_path} ({len(metric_names)} metrics)" if scores is not None else "none"}')
    print(f'Output: {output}')

    # 1. Measure every structure of every raw and augmented image
    rows, quantiles = [], []
    n_images = {m: 0 for m in present}

    def collect(stem, modality, source, draw, image, label, spacing):
        for value, features, q in measure_structures(image, label, label_values, spacing, shell_mm, bin_width,
                                                     samples, with_size=draw == 0):
            rows.append({'subject': stem, 'modality': modality, 'source': source, 'draw': draw, 'label': value,
                         'structure': structures[value], **features})
            quantiles.append(q)

    for stem, modality, img_path, lab_path in tqdm(pairs, desc='Measuring structures'):
        nii = nib.load(img_path)
        image = np.ascontiguousarray(nii.dataobj, dtype=np.float32)
        label = np.ascontiguousarray(np.round(np.asanyarray(nib.load(lab_path).dataobj)), dtype=np.int32)
        if image.shape != label.shape:
            warnings.warn(f'Skipping {stem}: image shape {image.shape} != label shape {label.shape}')
            continue
        spacing = nii.header.get_zooms()[:3]
        n_images[modality] += 1
        # Normalize on the whole image, then drop what is far from any structure
        image, label = crop_to_labels(normalize(image), label, margin=CROP_MARGIN)
        collect(stem, modality, modality, 0, image, label, spacing)

        if augmenter is not None and modality in aug_source:
            for draw in range(1, n_aug + 1):
                aug_image, aug_label = augment(augmenter, image, label, device, patch_size, batch_size)
                collect(stem, modality, f'{modality}_aug', draw, aug_image, aug_label, spacing)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError('No intensity extracted: check that the labels contain values listed in the dataset JSON.')

    stats = pd.DataFrame(rows)
    quantiles = np.stack(quantiles)
    sources = [s for s in ('MR', 'CT', 'MR_aug', 'CT_aug') if s in set(stats['source'])]

    # 2. Histogram PCA and distances between sources, per (image, draw, structure)
    pcs, ratio = add_pca_features(stats, quantiles, sources, hist_bins, pca_components, output)
    stats.to_csv(output / 'intensity_stats.csv', index=False)

    # 3. Per-structure summary and distances between the pooled distributions of each pair of sources
    pooled = {k: quantiles[idx].ravel() for k, idx in stats.groupby(['source', 'label']).indices.items()}
    summary, distances = [], []
    for value in label_values:
        for source in sources:
            sub = stats[(stats['label'] == value) & (stats['source'] == source)]
            if sub.empty:
                continue
            vals = pooled[(source, value)]
            row = {'label': int(value), 'structure': structures[value], 'source': source,
                   'n_images': sub['subject'].nunique(), 'n_samples': len(sub),
                   'mean_of_means': sub['mean'].mean(), 'std_of_means': sub['mean'].std(),
                   'mean_of_medians': sub['p50'].mean(), 'mean_of_stds': sub['std'].mean(),
                   'pooled_mean': float(vals.mean()), 'pooled_std': float(vals.std())}
            row.update({f'pooled_p{p:02d}': float(v) for p, v in zip(PERCENTILES, np.percentile(vals, PERCENTILES))})
            summary.append(row)
        for a, b in itertools.combinations(sources, 2):
            if (a, value) not in pooled or (b, value) not in pooled:
                continue
            va, vb = pooled[(a, value)], pooled[(b, value)]
            distances.append({'label': int(value), 'structure': structures[value], 'source_a': a, 'source_b': b,
                              'wasserstein': wasserstein(va, vb),
                              'coverage_a_by_b': coverage(va, vb), 'coverage_b_by_a': coverage(vb, va)})
    pd.DataFrame(summary).to_csv(output / 'intensity_summary.csv', index=False)
    distances = pd.DataFrame(distances)
    distances.to_csv(output / 'intensity_distances.csv', index=False)
    if not distances.empty:
        print('\nMean over structures:')
        print(distances.groupby(['source_a', 'source_b'])[['wasserstein', 'coverage_a_by_b', 'coverage_b_by_a']]
              .mean().round(3).to_string())

    # 4. Feature tables: one row per (image, structure) and one row per (modality, structure)
    instances = build_instance_table(stats, quantiles, pcs, sources)
    if scores is not None:
        instances = instances.merge(scores, on=['subject', 'structure'], how='left')
        matched = int(instances[metric_names].notna().any(axis=1).sum())
        print(f'\nMetrics matched for {matched}/{len(instances)} (image, structure) pairs')
        if matched == 0:
            warnings.warn('No row of the metrics matches the images/structures: the correlations are skipped')
            instances = instances.drop(columns=metric_names)
            metric_names = []
    structure_table = build_structure_table(instances, stats, pcs, n_images, metric_names)
    instances.to_csv(output / 'instance_features.csv', index=False)
    structure_table.to_csv(output / 'structure_features.csv', index=False)

    # 5. Correlations between the features and the performance metrics
    structure_corr = instance_corr = None
    if metric_names:
        not_features = {'subject', 'modality', 'label', 'structure', 'n_images', *metric_names, *(f'{m}_n' for m in metric_names)}
        structure_corr = correlate(structure_table, [c for c in structure_table.columns if c not in not_features],
                                   metric_names, per_structure=False)
        instance_corr = correlate(instances, [c for c in instances.columns if c not in not_features],
                                  metric_names, per_structure=True)
        structure_corr.to_csv(output / 'correlations_structure_level.csv', index=False)
        instance_corr.to_csv(output / 'correlations_instance_level.csv', index=False)
        for modality, sub in structure_corr.groupby('modality'):
            print(f'\nStrongest structure-level correlations, {modality} images (Spearman):')
            top = sub.reindex(sub['spearman_r'].abs().sort_values(ascending=False).index).groupby('metric').head(5)
            print(top.sort_values('metric', kind='stable')[['metric', 'feature', 'n', 'spearman_r', 'p_value', 'q_value']].round(3).to_string(index=False))

    # 6. Figures
    plot_boxplots(stats, structures, sources, output, norm_desc)
    plot_pca(stats, sources, ratio, output)
    if structure_corr is not None and not structure_corr.empty:
        plot_correlations(structure_corr, structure_table, metric_names, output)
    print(f'\nResults written to {output}')


# --- figures ------------------------------------------------------------------------


def _pyplot():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('matplotlib is not installed: no plot generated')
        return None
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK_MUTED,
                         'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'axes.labelcolor': INK,
                         'text.color': INK, 'grid.color': GRID, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
                         'savefig.facecolor': SURFACE})
    return plt


def plot_boxplots(stats, structures, sources, output, norm_desc, per_figure=20):
    plt = _pyplot()
    if plt is None:
        return
    present = [v for v in sorted(structures) if v in set(stats['label'])]
    width = 0.8 / len(sources)
    for i in range(0, len(present), per_figure):
        chunk = present[i:i + per_figure]
        fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(chunk) * len(sources)), 6))
        for j, source in enumerate(sources):
            data = [stats.loc[(stats['label'] == v) & (stats['source'] == source), 'p50'].to_numpy() for v in chunk]
            positions = [k + (j - (len(sources) - 1) / 2) * width for k in range(len(chunk))]
            keep = [(p, d) for p, d in zip(positions, data) if d.size]
            if not keep:
                continue
            bp = ax.boxplot([d for _, d in keep], positions=[p for p, _ in keep], widths=width * 0.85,
                            patch_artist=True, showfliers=False, medianprops={'color': INK},
                            whiskerprops={'color': INK_MUTED}, capprops={'color': INK_MUTED})
            for patch in bp['boxes']:
                patch.set(facecolor=SOURCE_COLORS[source], edgecolor=SURFACE)
            ax.plot([], [], color=SOURCE_COLORS[source], linewidth=8, label=source)
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([structures[v] for v in chunk], rotation=60, ha='right')
        ax.set_ylabel('Median intensity per image')
        ax.set_title(f'Median intensity of each structure (normalization: {norm_desc})', fontsize=10, loc='left')
        ax.legend(frameon=False)
        ax.grid(axis='y', linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output / f'intensity_boxplot_{i // per_figure}.png', dpi=150)
        plt.close(fig)


def plot_pca(stats, sources, ratio, output):
    """PC1/PC2 of the intensity histograms: one panel per source, structure centroids labelled."""
    plt = _pyplot()
    if plt is None or 'pc2' not in stats.columns:
        return
    cols = min(2, len(sources))
    rows = int(np.ceil(len(sources) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 6.5 * rows), sharex=True, sharey=True, squeeze=False)
    for ax in axes.ravel()[len(sources):]:
        ax.set_visible(False)
    for ax, source in zip(axes.ravel(), sources):
        sub = stats[stats['source'] == source]
        ax.scatter(sub['pc1'], sub['pc2'], s=6, color=SOURCE_COLORS[source], alpha=0.25, linewidths=0)
        centroids = sub.groupby('structure')[['pc1', 'pc2']].mean()
        ax.scatter(centroids['pc1'], centroids['pc2'], s=28, color=INK, edgecolors=SURFACE, linewidths=1, zorder=3)
        for name, (x, y) in centroids.iterrows():
            ax.annotate(name, (x, y), xytext=(4, 3), textcoords='offset points', fontsize=7, color=INK, zorder=4)
        ax.set_title(f'{source} ({sub["subject"].nunique()} images)', fontsize=10, loc='left')
        ax.set_xlabel(f'PC1 ({ratio[0]:.0%} of variance)')
        ax.set_ylabel(f'PC2 ({ratio[1]:.0%} of variance)')
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.grid(linewidth=0.5)
    fig.suptitle('PCA of the intensity histogram of each structure — coloured dots: one (image, draw, structure), '
                 'black dots: structure centroids', fontsize=10, x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(output / 'histogram_pca.png', dpi=150)
    plt.close(fig)


def plot_correlations(corr, structure_table, metrics, output, n_features=30, n_scatter=6):
    """Per modality: heatmap of the strongest structure-level correlations and scatter plots of the top features."""
    plt = _pyplot()
    if plt is None:
        return
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blue_red', [DIVERGING[0], '#f0efec', DIVERGING[1]])

    for modality, sub in corr.groupby('modality'):
        r = sub.pivot(index='feature', columns='metric', values='spearman_r').reindex(columns=metrics)
        q = sub.pivot(index='feature', columns='metric', values='q_value').reindex(columns=metrics)
        keep = r.abs().max(axis=1).sort_values(ascending=False).index[:n_features]
        r, q = r.loc[keep], q.loc[keep]
        n = int(sub['n'].max())

        fig, ax = plt.subplots(figsize=(3.2 + 1.5 * len(metrics), 1.2 + 0.28 * len(r)))
        im = ax.imshow(r.to_numpy(), cmap=cmap, vmin=-1, vmax=1, aspect='auto')
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                value = r.iat[i, j]
                if np.isfinite(value):
                    ax.text(j, i, f'{value:.2f}' + ('*' if q.iat[i, j] < 0.05 else ''), ha='center', va='center',
                            fontsize=7, color='white' if abs(value) > 0.6 else INK)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(metrics, fontsize=8)
        ax.set_yticks(range(len(r)))
        ax.set_yticklabels(r.index, fontsize=7)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'{modality} images: Spearman correlation between structure features and mean scores\n'
                     f'(up to {n} structures, {len(r)} strongest features, * q < 0.05)', fontsize=9, loc='left')
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).outline.set_visible(False)
        fig.tight_layout()
        fig.savefig(output / f'correlation_heatmap_{modality}.png', dpi=150)
        plt.close(fig)

        table = structure_table[structure_table['modality'] == modality]
        for metric in metrics:
            top = sub[sub['metric'] == metric].head(n_scatter)
            if top.empty:
                continue
            cols = min(3, len(top))
            rows = int(np.ceil(len(top) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4 * rows), squeeze=False)
            for ax in axes.ravel()[len(top):]:
                ax.set_visible(False)
            for ax, (_, row) in zip(axes.ravel(), top.iterrows()):
                ax.scatter(table[row['feature']], table[metric], s=30, color=SOURCE_COLORS[modality],
                           edgecolors=SURFACE, linewidths=1, zorder=3)
                for name, x, y in zip(table['structure'], table[row['feature']], table[metric]):
                    if np.isfinite(x) and np.isfinite(y):
                        ax.annotate(name, (x, y), xytext=(4, 3), textcoords='offset points', fontsize=6.5, color=INK_MUTED)
                ax.set_xlabel(row['feature'], fontsize=8)
                ax.set_ylabel(f'Mean {metric}', fontsize=8)
                ax.set_title(f'r = {row["spearman_r"]:.2f}, ' + ('p < 0.001' if row['p_value'] < 0.001 else f'p = {row["p_value"]:.3f}') + f', n = {row["n"]}', fontsize=9, loc='left')
                ax.grid(linewidth=0.5)
            fig.suptitle(f'{modality} images: structure features most correlated with {metric} (one dot per structure)',
                         fontsize=10, x=0.01, ha='left')
            fig.tight_layout()
            fig.savefig(output / f'correlation_scatter_{modality}_{metric}.png', dpi=150)
            plt.close(fig)


if __name__ == '__main__':
    main()
