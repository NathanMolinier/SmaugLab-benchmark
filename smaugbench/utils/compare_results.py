"""
Aggregate metrics.csv files across every `metrics*` folder produced by
inference.sh for a given nnUNet dataset ID, and generate comparison tables.

For each `$SMAUGBENCH_DATA/inference/Dataset<ID>_*/metrics*/metrics.csv` this
script produces:
  * comparison_long.csv    -- concatenated per-subject rows, tagged by model
  * comparison_summary.csv -- mean / std / count per (model, label, metric)
  * comparison_<metric>.csv -- pivot with rows=model, columns=label,
                               values="mean +/- std" (one file per metric)
"""

import argparse
import os
import textwrap
from pathlib import Path

import pandas as pd


META_COLUMNS = {'model', 'reference', 'prediction', 'label', 'EmptyRef', 'EmptyPred'}


def main():
    parser = argparse.ArgumentParser(
        description=' '.join('''
            Aggregate metrics.csv files produced by inference.sh across every
            `metrics*` folder inside $SMAUGBENCH_DATA/inference/Dataset<ID>_*/
            and generate comparison tables.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
              smaugbench_compare_results -d 501
              smaugbench_compare_results -d 501 -o /tmp/comparison
              smaugbench_compare_results -d 501 --metrics DiceSimilarityCoefficient NormalizedSurfaceDistance
        '''),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '-d', '--dataset-id', required=True, type=str,
        help='nnUNet dataset ID, e.g. 501.',
    )
    parser.add_argument(
        '-i', '--inference-dir', type=Path, default=None,
        help='Root inference folder. Defaults to $SMAUGBENCH_DATA/inference.',
    )
    parser.add_argument(
        '-o', '--output-dir', type=Path, default=None,
        help='Where to write the aggregated tables. Defaults to <dataset-folder>/comparison.',
    )
    parser.add_argument(
        '--metrics', nargs='+', default=None,
        help='Restrict the pivot tables to these metric columns (as they appear in metrics.csv). '
             'By default every numeric column is included.',
    )
    args = parser.parse_args()

    inference_dir = _resolve_inference_dir(args.inference_dir)

    dataset_folders = sorted(inference_dir.glob(f'Dataset{args.dataset_id}_*'))
    if not dataset_folders:
        raise FileNotFoundError(
            f'No dataset folder matching Dataset{args.dataset_id}_* under {inference_dir}.'
        )
    if len(dataset_folders) > 1:
        print(f'Warning: multiple dataset folders match Dataset{args.dataset_id}_*; '
              f'using {dataset_folders[0].name}.')
    dataset_dir = dataset_folders[0]

    output_dir = args.output_dir if args.output_dir else dataset_dir / 'comparison'
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df = _load_all_metrics(dataset_dir)
    metric_cols = _select_metric_columns(long_df, args.metrics)

    print(f'Loaded metrics from {long_df["model"].nunique()} model(s), '
          f'{len(long_df)} subject-label rows, {len(metric_cols)} metric column(s).')

    long_out = output_dir / 'comparison_long.csv'
    long_df.to_csv(long_out, index=False)
    print(f'Wrote {long_out}')

    summary = _summary_table(long_df, metric_cols)
    summary_out = output_dir / 'comparison_summary.csv'
    summary.to_csv(summary_out, index=False)
    print(f'Wrote {summary_out}')

    for metric in metric_cols:
        pivot = _pivot_metric(long_df, metric)
        pivot_out = output_dir / f'comparison_{metric}.csv'
        pivot.to_csv(pivot_out)
        print(f'Wrote {pivot_out}')


def _resolve_inference_dir(inference_dir_arg):
    if inference_dir_arg is not None:
        return Path(inference_dir_arg)
    data_root = os.environ.get('SMAUGBENCH_DATA')
    if not data_root:
        raise EnvironmentError(
            'SMAUGBENCH_DATA is not set and no -i/--inference-dir was provided.'
        )
    return Path(data_root) / 'inference'


def _load_all_metrics(dataset_dir):
    frames = []
    prefix = 'metrics_'
    for folder in sorted(p for p in dataset_dir.glob('metrics*') if p.is_dir()):
        csv_path = folder / 'metrics.csv'
        if not csv_path.is_file():
            print(f'Skipping {folder.name}: no metrics.csv')
            continue
        model_name = folder.name[len(prefix):] if folder.name.startswith(prefix) else folder.name
        df = pd.read_csv(csv_path)
        df.insert(0, 'model', model_name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f'No metrics.csv found in {dataset_dir}/metrics*.'
        )
    return pd.concat(frames, ignore_index=True)


def _select_metric_columns(df, requested):
    numeric_cols = [c for c in df.columns
                    if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if requested:
        missing = [m for m in requested if m not in df.columns]
        if missing:
            raise ValueError(f'Requested metric(s) not present in metrics.csv: {missing}')
        return list(requested)
    return numeric_cols


def _summary_table(df, metric_cols):
    agg = df.groupby(['model', 'label'])[metric_cols].agg(['mean', 'std', 'count']).reset_index()
    agg.columns = ['_'.join(col).rstrip('_') for col in agg.columns.values]
    return agg.round(3)


def _pivot_metric(df, metric):
    grouped = df.groupby(['model', 'label'])[metric].agg(['mean', 'std'])
    formatted = grouped.apply(
        lambda row: (f'{row["mean"]:.3f} +/- {row["std"]:.3f}'
                     if pd.notna(row['std']) else f'{row["mean"]:.3f}'),
        axis=1,
    )
    return formatted.unstack('label')


if __name__ == '__main__':
    main()
