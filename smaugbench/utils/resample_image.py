import argparse, textwrap
import multiprocessing as mp
from functools import partial
from tqdm.contrib.concurrent import process_map
from pathlib import Path
import warnings
from smaugbench.utils.image import Image, resample_nib

warnings.filterwarnings("ignore")


def main():

    # Description and arguments
    parser = argparse.ArgumentParser(
        description=' '.join(f'''
            This script processes NIfTI (Neuroimaging Informatics Technology Initiative) images and resample them to a specific resolution in mm.
        '''.split()),
        epilog=textwrap.dedent('''
            Examples:
            resample_image -i images -o images
        '''),
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--images-dir', '-i', type=Path, required=True,
        help='The folder where input NIfTI images files are located (required).'
    )
    parser.add_argument(
        '--output-images-dir', '-o', type=Path, required=True,
        help='The folder where output images will be saved (required).'
    )
    parser.add_argument(
        '--resolution', '-res', type=str, default='1x1x1',
        help='Output resolution for the images, defaults to "1x1x1".'
    )
    parser.add_argument(
        '--interpolation', '-int', type=str, default='linear',
        help='Interpolation method for resampling, defaults to "linear".'
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
    images_path = args.images_dir
    output_images_path = args.output_images_dir
    resolution = args.resolution
    interpolation = args.interpolation
    overwrite = args.overwrite
    max_workers = args.max_workers
    quiet = args.quiet

    # Print the argument values if not quiet
    if not quiet:
        print(textwrap.dedent(f'''
            Running {Path(__file__).stem} with the following params:
            images_path = "{images_path}"
            output_images_path = "{output_images_path}"
            resolution = "{resolution}"
            interpolation = "{interpolation}"
            overwrite = {overwrite}
            max_workers = {max_workers}
            quiet = {quiet}
        '''))

    resample_mp(
        images_path=images_path,
        output_images_path=output_images_path,
        resolution=resolution,
        interpolation=interpolation,
        overwrite=overwrite,
        max_workers=max_workers,
        quiet=quiet,
    )

def resample_mp(
        images_path,
        output_images_path,
        resolution='1x1x1',
        interpolation='linear',
        overwrite=False,
        max_workers=mp.cpu_count(),
        quiet=False,
    ):
    '''
    Wrapper function to handle multiprocessing.
    '''
    images_path = Path(images_path)
    output_images_path = Path(output_images_path)

    glob_pattern = '*.nii.gz'

    # Process the NIfTI image and segmentation files
    image_path_list = list(images_path.glob(glob_pattern))
    output_image_path_list = [output_images_path / _.relative_to(images_path).parent / _.name for _ in image_path_list]

    process_map(
        partial(
            _resample,
            overwrite=overwrite,
            resolution=resolution,
            interpolation=interpolation,
            quiet=quiet
        ),
        image_path_list,
        output_image_path_list,
        max_workers=max_workers,
        chunksize=1,
        disable=quiet,
    )

def _resample(
        image_path,
        output_image_path,
        overwrite=False,
        resolution='1x1x1',
        interpolation='linear',
        quiet=False
    ):
    '''
    Resample the image to the specified resolution.
    '''
    image_path = Path(image_path)
    output_image_path = Path(output_image_path)

    # Check if the output image already exists
    if not overwrite and output_image_path.exists():
        return

    image = Image(str(image_path))

    # Resample the image to the specified resolution
    resolution_tuple = tuple(map(float, resolution.split('x')))
    image = resample_nib(image, new_size=resolution_tuple, new_size_type='mm', interpolation=interpolation, verbose=not quiet)

    # Make sure output directory exists and save the image
    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_image_path))

if __name__ == '__main__':
    main()