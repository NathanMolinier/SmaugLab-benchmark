import argparse, textwrap, os
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def main():
    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script generates a nnunet splits_final.json from a custom data JSON file.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            python create_splits_nnunet.py -i input.json -o output.json
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input-json', '-i', type=Path, required=True,
        help='The data JSON file with the training splits and files path (required).'
    )
    parser.add_argument(
        '--output-json', '-o', type=Path, required=True,
        help='The output JSON file for the nnunet splits format (required).'
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
    Create a nnunet splits_final.json from a custom data JSON file.
    '''
    # Load custom json file
    with open(input_json, 'r') as f:
        data = json.load(f)

    # Check if output file exists and overwrite is False
    if os.path.exists(output_json) and not overwrite:
        print(f'WARNING: Output file {output_json} already exists. Use --overwrite to overwrite it.')
        return

    # Create nnunet splits_final.json structure
    nnunet_splits = {"train": [], "val": []}
    train_split = [os.path.basename(data['TRAINING'][i]['IMAGE']).replace('.nii.gz', '') for i in range(len(data['TRAINING']))]
    val_split = [os.path.basename(data['VALIDATION'][i]['IMAGE']).replace('.nii.gz', '') for i in range(len(data['VALIDATION']))]
    nnunet_splits["train"] = train_split
    nnunet_splits["val"] = val_split

    # Save the nnunet splits to a JSON file
    with open(output_json, 'w') as f:
        json.dump([nnunet_splits], f, indent=4)

if __name__ == '__main__':
    main()