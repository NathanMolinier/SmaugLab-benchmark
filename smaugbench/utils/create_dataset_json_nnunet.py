import argparse, textwrap, os
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script generates a nnunet dataset.json based on LABELS in the data.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            python create_dataset_json_nnunet.py -i input.json -o output.json
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input-json', '-i', type=Path, required=True,
        help='The data JSON file with the training splits and files path (required).'
    )
    parser.add_argument(
        '--output-json', '-o', type=Path, required=True,
        help='The output JSON file for the nnunet dataset.json format (required).'
    )
    parser.add_argument(
        '--overwrite', '-r', action="store_true", default=False,
        help='Overwrite existing output files, defaults to false (Do not overwrite).'
    )
    parser.add_argument(
        '--quiet', '-q', action="store_true", default=False,
        help='Do not display inputs and progress bar, defaults to false (display).'
    )

    # Parse the command-line arguments
    args = parser.parse_args()

    # Get the command-line argument values
    input_json = args.input_json
    output_json = args.output_json
    overwrite = args.overwrite
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            input_json = "{input_json}"
            output_json = "{output_json}"
            overwrite = {overwrite}
            quiet = {quiet}
        '''))

    generate_nnunet_splits(
        input_json=input_json,
        output_json=output_json,
        overwrite=overwrite,
        quiet=quiet,
    )

def generate_nnunet_splits(
        input_json,
        output_json,
        overwrite=False,
        quiet=False
    ):
    '''
    Create a nnunet dataset.json from a custom data JSON file.
    '''
    # Load custom json file
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Fetch data from input JSON
    labels = data.get('LABELS', {})
    if not labels:
        raise ValueError(f'No LABELS found in input JSON {input_json}. Please provide LABELS for the dataset.')
    rev_labels = {v: int(k) for k, v in labels.items()}
    numTraining = len(data.get('TRAINING', [])) + len(data.get('VALIDATION', []))

    # Check if output file exists and overwrite is False
    if os.path.exists(output_json) and not overwrite:
        print(f'WARNING: Output file {output_json} already exists. Use --overwrite to overwrite it.')
        return

    # Create nnunet dataset.json structure
    rev_labels["background"] = 0  # add background label if not present
    nnunet_dataset = {
        "channel_names": {"0":"channel_0"},
        "labels": rev_labels,
        "numTraining": numTraining,
        "file_ending": ".nii.gz"
    }

    # Save the nnunet dataset to a JSON file
    with open(output_json, 'w') as f:
        json.dump(nnunet_dataset, f, indent=4)

if __name__ == '__main__':
    main()