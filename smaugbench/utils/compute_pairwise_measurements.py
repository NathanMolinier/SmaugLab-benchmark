"""
Compute MetricsReloaded metrics for segmentation tasks.

Based on https://github.com/ivadomed/MetricsReloaded
"""


import os
import argparse, textwrap
import numpy as np
import multiprocessing as mp
import pandas as pd
from tqdm.contrib.concurrent import process_map
from functools import partial
from pathlib import Path
import json
from smaugbench.utils.image import Image

from smaugbench.utils.measures.pairwise_measures import BinaryPairwiseMeasures as BPM

# This dictionary is used to rename the metric columns in the output CSV file
METRICS_TO_NAME = {
    'dsc': 'DiceSimilarityCoefficient',
    'hd': 'HausdorffDistance95',
    'fbeta': 'F1score',
    'nsd': 'NormalizedSurfaceDistance',
    'vol_diff': 'VolumeDifference',
    'rel_vol_error': 'RelativeVolumeError',
    'lesion_ppv': 'LesionWisePositivePredictiveValue',
    'lesion_sensitivity': 'LesionWiseSensitivity',
    'lesion_f1_score': 'LesionWiseF1Score',
    'ref_count': 'RefLesionsCount',
    'pred_count': 'PredLesionsCount',
    'lcwa': 'LesionCountWeightedByAssignment'
}

def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) segmentations and computes pairwise metrics.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            compute_pairwise_measurements -pred predictions -ref references -o metrics.csv
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '-pred', required=True, type=str,
        help='Path to the folder with nifti images of test predictions or path to a single nifti image of test prediction.'
    )
    parser.add_argument(
        '-ref', required=True, type=str,
        help='Path to the folder with nifti images of reference (ground truth) or path to a single nifti image of reference (ground truth).'
    )
    parser.add_argument(
        '-metrics', nargs='+', required=False,
        default=['dsc', 'fbeta', 'nsd', 'vol_diff', 'rel_vol_error',
                'lesion_ppv', 'lesion_sensitivity', 'lesion_f1_score',
                'ref_count', 'pred_count', 'lcwa'],
        help='List of metrics to compute. For details, see: https://metricsreloaded.readthedocs.io/en/latest/reference/metrics/metrics.html.'
    )
    parser.add_argument(
        '-output', '-o', type=str, default='metrics.csv', required=False,
        help='Path to the output CSV file to save the metrics. Default: metrics.csv'
    )
    parser.add_argument(
        '-pred-map', type=str, metavar='<json>', default=None, required=False,
        help='JSON file containing the prediction mapping between the imaged structure and the corresponding integer value in the image ~/<your_path>/<myjson>.json'
    )
    parser.add_argument(
        '-ref-map', type=str, metavar='<json>', default=None, required=False,
        help='JSON file containing the reference mapping between the imaged structure and the corresponding integer value in the image ~/<your_path>/<myjson>.json'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--max-workers', '-w', type=int, default=mp.cpu_count(),
        help='Max worker to run in parallel proccess, defaults to multiprocessing.cpu_count().'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    prediction_path = args.pred
    reference_path = args.ref
    metrics = args.metrics
    output_path = args.output
    pred_map_path = args.pred_map
    ref_map_path = args.ref_map
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Check if both -pred-map and -ref-map are referenced if at least one is specified
    if any((ref_map_path is None, pred_map_path is None)) and any((ref_map_path is not None, pred_map_path is not None)):
        raise ValueError(f'If used, both -ref-map and -pred-map must be provided.')

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            prediction_path = "{prediction_path}"
            reference_path = "{reference_path}"
            metrics = {metrics}
            output_path = "{output_path}"
            pred_map = "{pred_map_path}"
            ref_map = "{ref_map_path}"
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    compute_pairwise_metrics_mp(
        prediction_path=prediction_path,
        reference_path=reference_path,
        metrics=metrics,
        output_path=output_path,
        pred_map_path=pred_map_path,
        ref_map_path=ref_map_path,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def compute_pairwise_metrics_mp(
        prediction_path,
        reference_path,
        metrics,
        output_path,
        pred_map_path=None,
        ref_map_path=None,
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''

    # Load JSON mapping if provided
    if not any((ref_map_path is None, pred_map_path is None)):
        # Load JSON files and create a dictionary
        with open(ref_map_path, "r") as file:
            ref_map = json.load(file)
        # Load JSON files and create a dictionary
        with open(pred_map_path, "r") as file:
            pred_map = json.load(file)
    else:
        # Assign None value if not used
        ref_map = None
        pred_map = None

    # Print the metrics to be computed
    print(f'Computing metrics: {metrics}')
    print(f'Using {max_workers} CPU cores in parallel ...')
    
    prediction_path = Path(prediction_path)
    reference_path = Path(reference_path)
    output_path = Path(output_path)

    glob_pattern = '*.nii.gz'

    # Process the NIfTI image and segmentation files
    prediction_path_list = sorted(prediction_path.glob(glob_pattern))
    reference_path_list = sorted(reference_path.glob(glob_pattern))

    # Check if every predictions and references have the same names
    if len(prediction_path_list) != len(reference_path_list):
        raise ValueError('The number of files between references and predictions is different.')

    check_names = np.array([pred.name == ref.name for pred, ref in zip(prediction_path_list, reference_path_list)])
    if not check_names.all():
        raise ValueError('Predictions and references must be named the same way in the two folders.')

    # Create output folder
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create pairwise folder to keep individual predictions
    pairwise_path = output_path.parent / 'pairwise'
    pairwise_path.mkdir(parents=True, exist_ok=True)
    pairwise_path_list = [pairwise_path / f'{pred.name.replace(".nii.gz", "")}_metrics.json' for pred in prediction_path_list]

    process_map(
        partial(
            compute_pairwise_metrics,
            metrics=metrics,
            ref_map=ref_map,
            pred_map=pred_map,
            overwrite=overwrite,
        ),
        prediction_path_list,
        reference_path_list,
        pairwise_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

    # Save all metrics into a single CSV
    save_metrics(pairwise_path_list, output_path)


def compute_pairwise_metrics(
        prediction_path, 
        reference_path, 
        output_path, 
        metrics,
        ref_map,
        pred_map,
        overwrite
    ):
    """
    Compute the pairwise metrics between prediction and reference
    """
    # Check if pairwise metrics are already computed
    if not overwrite and output_path.exists():
        return

    # load nifti images
    prediction_data = Image(str(prediction_path)).change_orientation('RPI').data
    reference_data = Image(str(reference_path)).change_orientation('RPI').data

    # check whether the images have the same shape and orientation
    if prediction_data.shape != reference_data.shape:
        raise ValueError(f'The prediction and reference (ground truth) images must have the same shape. '
                        f'The prediction image has shape {prediction_data.shape} and the ground truth image has '
                        f'shape {reference_data.shape}.')

    if ref_map is None and pred_map is None:
        # get all unique labels (classes)
        # for example, for nnunet region-based segmentation, spinal cord has label 1, and lesions have label 2
        unique_labels = np.unique(reference_data)
        unique_labels = unique_labels[unique_labels != 0]  # remove background
    else:
        # Get the unique labels that are present in the reference
        unique_labels = list(ref_map.keys())

    # append entry into the output_list to store the metrics for the current subject
    metrics_dict = {'reference': str(reference_path), 'prediction': str(prediction_path)}

    # loop over all unique labels, e.g., voxels with values 1, 2, ...
    # by doing this, we can compute metrics for each label separately, e.g., separately for spinal cord and lesions
    for label in unique_labels:
        # create binary masks for the current label
        if not isinstance(label, str):
            prediction_data_label = (prediction_data == label).astype(np.uint8)
            reference_data_label = (reference_data == label).astype(np.uint8)
        else:
            prediction_data_label = (prediction_data == pred_map[label]).astype(np.uint8)
            reference_data_label = (reference_data == ref_map[label]).astype(np.uint8)
        if not np.any(reference_data_label):
            # If reference is empty for this label, skip computing metrics
            continue
        bpm = BPM(prediction_data_label, reference_data_label, measures=metrics)
        dict_seg = bpm.to_dict_meas()
        # Store info whether the reference or prediction is empty
        dict_seg['EmptyRef'] = bool(bpm.flag_empty_ref)
        dict_seg['EmptyPred'] = bool(bpm.flag_empty_pred)
        # add the metrics to the output dictionary; JSON keys must be strings
        label_key = label if isinstance(label, str) else str(label.item() if hasattr(label, 'item') else label)
        metrics_dict[label_key] = dict_seg

    # Save pairwise metrics
    with open(str(output_path), 'w') as f:
        json.dump(metrics_dict, f, indent=2)

def save_metrics(path_list, output_path):

    # Load all computed metrics
    dict_list = []
    for path in path_list:
        with open(path, 'r') as f:
            dict_list.append(json.load(f))

    # Convert JSON data to pandas DataFrame
    df = build_output_dataframe(dict_list)

    # keep only the rows where either pred or ref is non-empty or both are non-empty
    df = df[(df['EmptyRef'] == False) | (df['EmptyPred'] == False)]

    # Compute mean and standard deviation of metrics across all subjects
    df_mean = (df.drop(columns=['reference', 'prediction', 'EmptyRef', 'EmptyPred']).groupby('label').
               agg(['mean', 'std']).reset_index())

    # Convert multi-index to flat index
    df_mean.columns = ['_'.join(col).strip() for col in df_mean.columns.values]
    # Rename column `label_` back to `label`
    df_mean.rename(columns={'label_': 'label'}, inplace=True)

    # Rename columns
    df.rename(columns={metric: METRICS_TO_NAME[metric] for metric in METRICS_TO_NAME}, inplace=True)
    df_mean.rename(columns={metric: METRICS_TO_NAME[metric] for metric in METRICS_TO_NAME}, inplace=True)

    # format output up to 3 decimal places
    df = df.round(3)
    df_mean = df_mean.round(3)

    # save as CSV
    fname_output_csv = os.path.abspath(str(output_path))
    df.to_csv(fname_output_csv, index=False)
    print(f'Saved metrics to {fname_output_csv}.')

    # save as CSV
    fname_output_csv_mean = os.path.abspath(str(output_path).replace('.csv', '_mean.csv'))
    df_mean.to_csv(fname_output_csv_mean, index=False)
    print(f'Saved mean and standard deviation of metrics across all subjects to {fname_output_csv_mean}.')

def build_output_dataframe(output_list):
    """
    Convert JSON data to pandas DataFrame
    :param output_list: list of dictionaries with metrics
    :return: pandas DataFrame
    """
    rows = []
    for item in output_list:
        # Extract all keys except 'reference' and 'prediction' to get labels (e.g. 1.0, 2.0, etc.) dynamically
        labels = [key for key in item.keys() if key not in ['reference', 'prediction']]
        for label in labels:
            metrics = item[label]  # Get the dictionary of metrics
            # Dynamically add all metrics for the label
            row = {
                "reference": item["reference"],
                "prediction": item["prediction"],
                "label": label,
            }
            # Update row with all metrics dynamically
            row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)

    return df

if __name__ == '__main__':
    main()
