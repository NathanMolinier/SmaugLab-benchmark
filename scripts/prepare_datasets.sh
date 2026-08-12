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

# Check if SMAUGBENCH_DATA is set else stop script.
if [ -z "$SMAUGBENCH_DATA" ]; then
    echo "Please set the SMAUGBENCH_DATA environment variable. This folder will store all preprocessed datasets and training."
    exit 1
fi

# Get project path
SMAUGBENCH="$(realpath "$(dirname "$0")/..")"
echo "SMAUGBENCH project path: $SMAUGBENCH"

# Load provided data_json file
data_json="$1"
if [ ! -f "$data_json" ]; then
    data_json="$SMAUGBENCH/smaugbench/datasets/$(basename "$1").json"
    if [ ! -f "$data_json" ]; then
        echo "Error: Could not find data JSON file."
        exit 1
    fi
fi

# Set the paths to the BIDS data folders
bids="$SMAUGBENCH_DATA"/bids

# Make sure $SMAUGBENCH_DATA/bids exists and enter it
if [ ! -d "$bids" ]; then
    echo "Error: Could not find BIDS data folder. Please run the download_datasets.sh script first."
    exit 1
fi
CURR_DIR="$(realpath .)"
cd "$bids"

# Get memory in GB and number of CPUs
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: /proc/meminfo
    MEMGB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)
    CORES=$(lscpu -p | egrep -v '^#' | wc -l)
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # MacOS: sysctl -n hw.memsize
    MEMGB=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    CORES=$(sysctl -n hw.ncpu)
else
    echo "Error: Unsupported OS type: $OSTYPE"
    exit 1
fi

# Set the number of jobs
JOBS=${SMAUGBENCH_JOBS:-$CORES}

# Set the number of jobs for the nnUNet
JOBSNN=$(( JOBS < $((MEMGB / 8)) ? JOBS : $((MEMGB / 8)) ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))
JOBSNN=${SMAUGBENCH_JOBSNN:-$JOBSNN}

# Set nnunet params
nnUNet_raw="$SMAUGBENCH_DATA"/nnUNet/raw
nnUNet_preprocessed="$SMAUGBENCH_DATA"/nnUNet/preprocessed
nnUNet_results="$SMAUGBENCH_DATA"/nnUNet/results

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
echo "DATASET_ID=${DATASET_ID}"
echo "DATASET_NAME=${DATASET_NAME}"
echo ""

### Prepare TRAIN set

IMAGES_TRAIN_DIR="$nnUNet_raw"/$SRC_DATASET/imagesTr
LABELS_TRAIN_DIR="$nnUNet_raw"/$SRC_DATASET/labelsTr

echo "Make nnUNet raw folders"
mkdir -p "$IMAGES_TRAIN_DIR"
mkdir -p "$LABELS_TRAIN_DIR"

# Copy label data in nnUNet_raw folder
jq -r '.TRAINING[].LABEL' "$data_json" | xargs -I{} cp "{}" "$LABELS_TRAIN_DIR/"
jq -r '.VALIDATION[].LABEL' "$data_json" | xargs -I{} cp "{}" "$LABELS_TRAIN_DIR/"

# Copy images and add nnUNet suffix _0000
jq -r '.TRAINING[].IMAGE' "$data_json" | while IFS= read -r img; do
    cp "$img" "$IMAGES_TRAIN_DIR/$(basename "${img/.nii.gz/_0000.nii.gz}")"
done
jq -r '.VALIDATION[].IMAGE' "$data_json" | while IFS= read -r img; do
    cp "$img" "$IMAGES_TRAIN_DIR/$(basename "${img/.nii.gz/_0000.nii.gz}")"
done

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
    if [ "$match_count" -ne 1 ]; then
        echo "Error: Expected exactly one label file for image $filename, but found $match_count."
        exit 1
    fi
    # Rename label file
    mv "${label_matches[0]}" "$LABELS_TRAIN_DIR"/"$prefix".nii.gz
done

# Reorient images to same orientation
echo "Transform training and validation images to $ORIENTATION"
smaugbench_reorient_image -i "$IMAGES_TRAIN_DIR" -o "$IMAGES_TRAIN_DIR" -ori $ORIENTATION -r -w $JOBS
smaugbench_reorient_image -i "$LABELS_TRAIN_DIR" -o "$LABELS_TRAIN_DIR" -ori $ORIENTATION -r -w $JOBS

# Resample images to a same resolution
echo "Resample training and validation images to $RESOLUTION"
smaugbench_resample_image -i "$IMAGES_TRAIN_DIR" -o "$IMAGES_TRAIN_DIR" -res $RESOLUTION -int linear -r -w $JOBS
smaugbench_resample_image -i "$LABELS_TRAIN_DIR" -o "$LABELS_TRAIN_DIR" -res $RESOLUTION -int nn -r -w $JOBS

### Prepare TEST set

IMAGES_TEST_DIR="$nnUNet_raw"/$SRC_DATASET/imagesTs
LABELS_TEST_DIR="$nnUNet_raw"/$SRC_DATASET/labelsTs

echo "Creating test folders"
mkdir -p "$IMAGES_TEST_DIR"
mkdir -p "$LABELS_TEST_DIR"

# Copy test data in nnUNet_raw folder
jq -r '.TESTING[].LABEL' "$data_json" | xargs -I{} cp "{}" "$LABELS_TEST_DIR/"

# Copy images and add nnUNet suffix _0000
jq -r '.TESTING[].IMAGE' "$data_json" | while IFS= read -r img; do
    cp "$img" "$IMAGES_TEST_DIR/$(basename "${img/.nii.gz/_0000.nii.gz}")"
done

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
    if [ "$match_count" -ne 1 ]; then
        echo "Error: Expected exactly one label file for image $filename, but found $match_count."
        exit 1
    fi
    # Rename label file
    mv "${label_matches[0]}" "$LABELS_TEST_DIR"/"$prefix".nii.gz
done

# Reorient images to same orientation
echo "Transform test images to $ORIENTATION"
smaugbench_reorient_image -i "$IMAGES_TEST_DIR" -o "$IMAGES_TEST_DIR" -ori $ORIENTATION -r -w $JOBS
smaugbench_reorient_image -i "$LABELS_TEST_DIR" -o "$LABELS_TEST_DIR" -ori $ORIENTATION -r -w $JOBS

# Resample images to a same resolution
echo "Resample test images to $RESOLUTION"
smaugbench_resample_image -i "$IMAGES_TEST_DIR" -o "$IMAGES_TEST_DIR" -res $RESOLUTION -int linear -r -w $JOBS
smaugbench_resample_image -i "$LABELS_TEST_DIR" -o "$LABELS_TEST_DIR" -res $RESOLUTION -int nn -r -w $JOBS

# Create dataset.json file for nnUNet
echo "Create dataset.json file for nnUNet"
smaugbench_dataset_json_nnunet -i "$data_json" -o "$nnUNet_raw"/$SRC_DATASET/dataset.json -r

### nnUNet Plan and Preprocess
export nnUNet_def_n_proc=$JOBSNN
export nnUNet_n_proc_DA=$JOBSNN
export nnUNet_raw="$SMAUGBENCH_DATA"/nnUNet/raw
export nnUNet_preprocessed="$SMAUGBENCH_DATA"/nnUNet/preprocessed
export nnUNet_results="$SMAUGBENCH_DATA"/nnUNet/results

echo "Run nnUNet plan and preprocess"
nnUNetv2_plan_and_preprocess -d $DATASET_ID -pl $nnUNetPlanner -overwrite_plans_name $nnUNetPlans -c $configuration -np $JOBSNN --verify_dataset_integrity

# Overwrite nnUNet splits_final.json
smaugbench_create_splits_nnunet -i "$data_json" -o "$nnUNet_preprocessed"/$SRC_DATASET/splits_final.json -r $JOBS

# Move back
cd "$CURR_DIR"