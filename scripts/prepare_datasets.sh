#!/bin/bash

# This script prepares datasets for training. Input Niftiis are reoriented to a common orientation and resampled.
# The data is then organized using nnUNet's expected folder structure and running nnUNet's preprocessing steps.
# Based on https://github.com/neuropoly/totalspineseg/blob/main/scripts/prepare_datasets.sh

# BASH SETTINGS
# ======================================================================================================================

# Uncomment for full verbose
# set -v

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# SCRIPT STARTS HERE
# ======================================================================================================================

# Check if SMBENCH_DATA is set else stop script.
if [ -z "$SMBENCH_DATA" ]; then
    echo "Please set the SMBENCH_DATA environment variable. This folder will store all preprocessed datasets and training."
    exit 1
fi

# Get project path
SMBENCH="$(realpath "$(dirname "$0")/..")"
echo "SMBENCH project path: $SMBENCH"

# Load provided data_json file
data_json="$1"
if [ ! -f "$data_json" ]; then
    data_json="$SMBENCH/smbench/datasets/$(basename "$1").json"
    if [ ! -f "$data_json" ]; then
        echo "Error: Could not find data JSON file."
        exit 1
    fi
fi

# Set the paths to the BIDS data folders
bids="$SMBENCH_DATA"/bids

# Make sure $SMBENCH_DATA/bids exists and enter it
if [ ! -d "$bids" ]; then
    echo "Error: Could not find BIDS data folder. Please run the download_datasets.sh script first."
    exit 1
fi
CURR_DIR="$(realpath .)"
cd "$bids"

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Set the number of jobs
JOBS=${SMBENCH_JOBS:-$CORES}

# Set the number of jobs for the nnUNet
JOBSNN=$(( JOBS < $((MEMGB / 8)) ? JOBS : $((MEMGB / 8)) ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))
JOBSNN=${SMBENCH_JOBSNN:-$JOBSNN}

# Set nnunet params
nnUNet_raw="$SMBENCH_DATA"/nnUNet/raw
nnUNet_preprocessed="$SMBENCH_DATA"/nnUNet/preprocessed
nnUNet_results="$SMBENCH_DATA"/nnUNet/results

DATASET_ID="$2"
DATASET_NAME="$3"
nnUNetPlanner=${4:-nnUNetPlannerResEncL}
nnUNetPlans=${5:-nnUNetPlans}
configuration=${6:-3d_fullres}

ORIENTATION="RPI"
RESOLUTION="1x1x1"

SRC_DATASET=Dataset${DATASET_ID}_${DATASET_NAME}

echo ""
echo "Running with the following parameters:"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNetPlanner=${nnUNetPlanner}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "configuration=${configuration}"
echo "JOBSNN=${JOBSNN}"
echo "DEVICE=${DEVICE}"
echo "DATASETS=${DATASETS[@]}"
echo "FOLD=${FOLD}"
echo ""

### Prepare TRAIN set

IMAGES_TRAIN_DIR="$nnUNet_raw"/$SRC_DATASET/imagesTr
LABELS_TRAIN_DIR="$nnUNet_raw"/$SRC_DATASET/labelsTr

echo "Make nnUNet raw folders"
mkdir -p "$IMAGES_TRAIN_DIR"
mkdir -p "$LABELS_TRAIN_DIR"

# Copy label data in nnUNet_raw folder
cp $(jq -r ".TRAINING | .[].LABEL" "$data_json") "$LABELS_TRAIN_DIR"
cp $(jq -r ".VALIDATION | .[].LABEL" "$data_json") "$LABELS_TRAIN_DIR"

# Copy images and add nnUNet suffix _0000
for img in $(jq -r ".TRAINING | .[].IMAGE" "$data_json");do img_name=$(basename ${img/.nii.gz/_0000.nii.gz}); cp "$img" "$IMAGES_TRAIN_DIR/"$img_name";done
for img in $(jq -r ".VALIDATION | .[].IMAGE" "$data_json");do img_name=$(basename ${img/.nii.gz/_0000.nii.gz}); cp "$img" "$IMAGES_TRAIN_DIR/"$img_name";done

# Remove label suffix from filename
for img in "$IMAGES_TRAIN_DIR"/*_0000.nii.gz; do
    # Get the filename without the path
    filename=$(basename "$img")
    # Extract the prefix by removing the _0000.nii.gz extension
    prefix="${filename%_0000.nii.gz}"
    # Use nullglob so the array is empty if no files match (instead of containing the literal '*' string)
    shopt -s nullglob
    # Find all labels that start with the prefix and have an underscore afterward
    label_matches=("$LABELS_TRAIN_DIR"/"$prefix"_*.nii.gz)
    shopt -u nullglob
    match_count=${#label_matches[@]}
    if [ ! $match_count -eq 1 ]; then
        echo "Error: Expected exactly one label file for image $filename, but found $match_count."
        exit 1
    fi
    # Rename label file
    mv "${label_matches[0]}" "$LABELS_TRAIN_DIR"/"$prefix".nii.gz
done

# Reorient images to same orientation
echo "Transform training and validation images to $ORIENTATION"
smbench_reorient_image -i "$IMAGES_TRAIN_DIR" -o "$IMAGES_TRAIN_DIR" -ori $ORIENTATION -r -w $JOBS
smbench_reorient_image -i "$LABELS_TRAIN_DIR" -o "$LABELS_TRAIN_DIR" -ori $ORIENTATION -r -w $JOBS

# Resample images to a same resolution
echo "Resample training and validation images to $RESOLUTION"
smbench_resample_image -i "$IMAGES_TRAIN_DIR" -o "$IMAGES_TRAIN_DIR" -res $RESOLUTION -int linear -r -w $JOBS
smbench_resample_image -i "$LABELS_TRAIN_DIR" -o "$LABELS_TRAIN_DIR" -res $RESOLUTION -int nn -r -w $JOBS

### Prepare TEST set

IMAGES_TEST_DIR="$nnUNet_raw"/$SRC_DATASET/imagesTs
LABELS_TEST_DIR="$nnUNet_raw"/$SRC_DATASET/labelsTs

echo "Creating test folders"
mkdir -p "$IMAGES_TEST_DIR"
mkdir -p "$LABELS_TEST_DIR"

# Copy test data in nnUNet_raw folder
cp $(jq -r ".TESTING | .[].LABEL" "$data_json") "$LABELS_TEST_DIR"

# Copy images and add nnUNet suffix _0000
for img in $(jq -r ".TESTING | .[].IMAGE" "$data_json");do img_name=$(basename ${img/.nii.gz/_0000.nii.gz}); cp "$img" "$IMAGES_TEST_DIR/"$img_name";done

# Remove label suffix from filename
for img in "$IMAGES_TEST_DIR"/*_0000.nii.gz; do
    # Get the filename without the path
    filename=$(basename "$img")
    # Extract the prefix by removing the _0000.nii.gz extension
    prefix="${filename%_0000.nii.gz}"
    # Use nullglob so the array is empty if no files match (instead of containing the literal '*' string)
    shopt -s nullglob
    # Find all labels that start with the prefix and have an underscore afterward
    label_matches=("$LABELS_TEST_DIR"/"$prefix"_*.nii.gz)
    shopt -u nullglob
    match_count=${#label_matches[@]}
    if [ ! $match_count -eq 1 ]; then
        echo "Error: Expected exactly one label file for image $filename, but found $match_count."
        exit 1
    fi
    # Rename label file
    mv "${label_matches[0]}" "$LABELS_TEST_DIR"/"$prefix".nii.gz
done

# Reorient images to same orientation
echo "Transform test images to $ORIENTATION"
smbench_reorient_image -i "$IMAGES_TEST_DIR" -o "$IMAGES_TEST_DIR" -ori $ORIENTATION -r -w $JOBS
smbench_reorient_image -i "$LABELS_TEST_DIR" -o "$LABELS_TEST_DIR" -ori $ORIENTATION -r -w $JOBS

# Resample images to a same resolution
echo "Resample test images to $RESOLUTION"
smbench_resample_image -i "$IMAGES_TEST_DIR" -o "$IMAGES_TEST_DIR" -res $RESOLUTION -int linear -r -w $JOBS
smbench_resample_image -i "$LABELS_TEST_DIR" -o "$LABELS_TEST_DIR" -res $RESOLUTION -int nn -r -w $JOBS

### nnUNet Plan and Preprocess
export nnUNet_def_n_proc=$JOBSNN
export nnUNet_n_proc_DA=$JOBSNN
export nnUNet_raw="$SMBENCH_DATA"/nnUNet/raw
export nnUNet_preprocessed="$SMBENCH_DATA"/nnUNet/preprocessed
export nnUNet_results="$SMBENCH_DATA"/nnUNet/results

echo "Run nnUNet plan and preprocess"
nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl $nnUNetPlanner -overwrite_plans_name $nnUNetPlans -c $configuration -np $JOBSNN --verify_dataset_integrity

# Overwrite nnUNet splits_final.json
smbench_create_splits_nnunet -i "$data_json" -o "$nnUNet_preprocessed"/$SRC_DATASET/splits_final.json -r $JOBS

# Move back
cd "$CURR_DIR"