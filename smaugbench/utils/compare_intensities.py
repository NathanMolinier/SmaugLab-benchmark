"""
Compare the intensity of each labelled structure between MR images, CT images and
GPU-augmented images generated with the SmaugLab trainer config.

The script reads the same pipeline config as `scripts/inference.sh` and works on the
inference folder that script already prepared:

    ${SMAUGBENCH_DATA}/inference/Dataset${dataset_id}_${dataset_name}${inference_supplementary_key}/
        images/<stem>_0000.nii.gz   (<stem> ends with _MR or _CT)
        labels/<stem>.nii.gz

For every image/label pair, the voxels of each structure listed in the `LABELS` field of
the dataset JSON (`data_json`) are extracted from:

    - the image itself (source `MR` or `CT`, from the filename suffix)
    - `--n-aug` augmented versions of the image (source `CT_aug` / `MR_aug`), generated
      with the GPU section of `nnunet_trainer_config`, built exactly like
      `nnUNetTrainerDAExtGPU` builds it, and with the label map passed as the mask so
      that the in-segmentation transforms behave as during training.

Images are normalized before extraction (and before augmentation, as in training).
By default the normalization stored in the trained model's plans.json is used
(CTNormalization or ZScoreNormalization), i.e. the intensities the network actually
sees. The CPU section of the trainer config (spatial transforms) is not applied: it only
moves voxels around and does not change intensities.

Outputs (in `<inference folder>/intensities_<nnunet_folder_name><supplementary_key>/` by default):
    intensity_stats.csv          one row per (subject, source, draw, structure)
    intensity_summary.csv        one row per (source, structure), pooled over subjects
    intensity_distances.csv      one row per (structure, source pair): Wasserstein distance
                                 and coverage of one distribution by the other
    intensity_boxplot_<i>.png    per-structure boxplots of the per-image medians
    config.json                  copy of the input config
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
PERCENTILES = (1, 5, 25, 50, 75, 95, 99)
# Voxels kept around the labelled structures when cropping (context for blur/convolution-like augmentations)
CROP_MARGIN = 16


def main():
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        epilog=textwrap.dedent('''
            Examples:
            smaugbench_compare_intensities smaugbench/configs/amos22_nnUNetTrainerDAExtGPU_OSSTA.json
            smaugbench_compare_intensities config.json --n-aug 10 --aug-source CT MR --device cuda:0
        '''),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument('config', type=str, help='Pipeline config JSON (same as the one given to run_pipeline.sh / inference.sh).')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output folder (default: <inference folder>/intensities_<nnunet_folder_name><supplementary_key>).')
    parser.add_argument('--n-aug', type=int, default=5, help='Number of augmented draws per image (default: 5). 0 disables augmentation.')
    parser.add_argument('--aug-source', nargs='+', choices=MODALITIES, default=['CT'],
                        help='Modalities of the images that are augmented (default: CT). '
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
    parser.add_argument('--samples', type=int, default=500,
                        help='Voxels randomly kept per (image, draw, structure) to compute the distribution distances (default: 500).')
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


# --- extraction ---------------------------------------------------------------------


def structure_values(image, label, label_values):
    """Yield (label value, intensities of that structure) for every present label of `label_values`."""
    fg = np.isin(label, label_values) & np.isfinite(image)
    lab = label[fg]
    val = image[fg]
    order = np.argsort(lab, kind='stable')
    lab, val = lab[order], val[order]
    uniques, starts = np.unique(lab, return_index=True)
    ends = np.append(starts[1:], len(lab))
    for value, start, end in zip(uniques, starts, ends):
        yield int(value), val[start:end]


def describe(values):
    pct = np.percentile(values, PERCENTILES)
    stats = {'n_voxels': int(values.size), 'mean': float(values.mean()), 'std': float(values.std())}
    stats.update({f'p{p:02d}': float(v) for p, v in zip(PERCENTILES, pct)})
    return stats


def wasserstein(a, b):
    from scipy.stats import wasserstein_distance
    return float(wasserstein_distance(a, b))


def coverage(target, reference):
    """Fraction of `target` voxels falling inside the [p1, p99] range of `reference`."""
    lo, hi = np.percentile(reference, [1, 99])
    return float(((target >= lo) & (target <= hi)).mean())


# --- main ---------------------------------------------------------------------------


def compare_intensities(config, output=None, n_aug=5, aug_source=('CT',), trainer_config=None, norm='nnunet',
                        plans=None, patch_size=None, batch_size=2, samples=500, device='cpu', seed=42, max_images=None):
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

    output = Path(output) if output else inference_dir / f"intensities_{cfg['nnunet_folder_name']}{key}"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output / 'config.json')

    print(f'Inference folder: {inference_dir}')
    print(f'Dataset JSON: {data_json} ({len(structures)} structures)')
    print(f'Images: {len(pairs)} ({", ".join(f"{m}: {sum(p[1] == m for p in pairs)}" for m in present)})')
    print(f'Normalization: {norm_desc}')
    print(f'Augmentation: {f"{n_aug} draws of {aug_source} with {trainer_config} on {device}, tiles {patch_size} x {batch_size}" if augmenter else "none"}')
    print(f'Output: {output}')

    rng = np.random.default_rng(seed)
    rows = []
    samples_store = {}  # (source, label value) -> list of sampled intensities

    def collect(stem, source, draw, image, label):
        for value, vals in structure_values(image, label, label_values):
            rows.append({'subject': stem, 'source': source, 'draw': draw, 'label': value,
                         'structure': structures[value], **describe(vals)})
            keep = vals if vals.size <= samples else rng.choice(vals, samples, replace=False)
            samples_store.setdefault((source, value), []).append(keep.astype(np.float32))

    for stem, modality, img_path, lab_path in tqdm(pairs, desc='Extracting intensities'):
        image = np.ascontiguousarray(nib.load(img_path).dataobj, dtype=np.float32)
        label = np.ascontiguousarray(np.round(np.asanyarray(nib.load(lab_path).dataobj)), dtype=np.int32)
        if image.shape != label.shape:
            warnings.warn(f'Skipping {stem}: image shape {image.shape} != label shape {label.shape}')
            continue
        # Normalize on the whole image, then drop what is far from any structure
        image, label = crop_to_labels(normalize(image), label, margin=CROP_MARGIN)
        collect(stem, modality, 0, image, label)

        if augmenter is not None and modality in aug_source:
            for draw in range(1, n_aug + 1):
                aug_image, aug_label = augment(augmenter, image, label, device, patch_size, batch_size)
                collect(stem, f'{modality}_aug', draw, aug_image, aug_label)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError('No intensity extracted: check that the labels contain values listed in the dataset JSON.')

    # Per-image statistics
    stats = pd.DataFrame(rows)
    stats.to_csv(output / 'intensity_stats.csv', index=False)

    sources = [s for s in ('MR', 'CT', 'MR_aug', 'CT_aug') if s in set(stats['source'])]
    pooled = {k: np.concatenate(v) for k, v in samples_store.items()}

    # Per-structure summary, pooled over subjects (from the sampled voxels) and averaged per image
    summary = []
    for value in label_values:
        for source in sources:
            sub = stats[(stats['label'] == value) & (stats['source'] == source)]
            if sub.empty:
                continue
            row = {'label': int(value), 'structure': structures[value], 'source': source,
                   'n_images': sub['subject'].nunique(), 'n_samples': len(sub),
                   'mean_of_means': sub['mean'].mean(), 'std_of_means': sub['mean'].std(),
                   'mean_of_medians': sub['p50'].mean(), 'mean_of_stds': sub['std'].mean()}
            vals = pooled[(source, value)]
            row.update({f'pooled_{k}': v for k, v in describe(vals).items() if k != 'n_voxels'})
            summary.append(row)
    pd.DataFrame(summary).to_csv(output / 'intensity_summary.csv', index=False)

    # Distances between the distributions of each pair of sources
    distances = []
    for value in label_values:
        for a, b in itertools.combinations(sources, 2):
            if (a, value) not in pooled or (b, value) not in pooled:
                continue
            va, vb = pooled[(a, value)], pooled[(b, value)]
            distances.append({'label': int(value), 'structure': structures[value], 'source_a': a, 'source_b': b,
                              'wasserstein': wasserstein(va, vb),
                              'coverage_a_by_b': coverage(va, vb), 'coverage_b_by_a': coverage(vb, va)})
    distances = pd.DataFrame(distances)
    distances.to_csv(output / 'intensity_distances.csv', index=False)
    if not distances.empty:
        print('\nMean over structures:')
        print(distances.groupby(['source_a', 'source_b'])[['wasserstein', 'coverage_a_by_b', 'coverage_b_by_a']]
              .mean().round(3).to_string())

    plot_boxplots(stats, structures, sources, output, norm_desc)
    print(f'\nResults written to {output}')


def plot_boxplots(stats, structures, sources, output, norm_desc, per_figure=20):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn('matplotlib is not installed: no plot generated')
        return

    colors = {'MR': '#1f77b4', 'CT': '#d62728', 'MR_aug': '#9ecae1', 'CT_aug': '#fc9272'}
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
            bp = ax.boxplot([d for _, d in keep], positions=[p for p, _ in keep], widths=width * 0.9,
                            patch_artist=True, showfliers=False, medianprops={'color': 'black'})
            for patch in bp['boxes']:
                patch.set_facecolor(colors.get(source, 'grey'))
            ax.plot([], [], color=colors.get(source, 'grey'), linewidth=8, label=source)
        ax.set_xticks(range(len(chunk)))
        ax.set_xticklabels([structures[v] for v in chunk], rotation=60, ha='right')
        ax.set_ylabel('Median intensity per image')
        ax.set_title(f'Normalization: {norm_desc}', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        path = output / f'intensity_boxplot_{i // per_figure}.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    main()
