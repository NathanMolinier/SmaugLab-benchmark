#!/bin/bash

# This script runs the inference pipeline for a given dataset and model configurations. To output comparisons between different models.

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

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
SMAUGBENCH="$(realpath "$(dirname "$0")/..")"

# Parse arguments
IMAGES_ONLY=false
config_json=""
for arg in "$@"; do
    case "$arg" in
        --images-only)
            IMAGES_ONLY=true
            ;;
        *)
            if [ -z "$config_json" ]; then
                config_json="$arg"
            else
                echo "Unknown argument: $arg"
                echo "Usage: $0 <config.json> [--images-only]"
                exit 1
            fi
            ;;
    esac
done

if [ -z "$config_json" ] || [ ! -f "$config_json" ]; then
    echo "Usage: $0 <config.json> [--images-only]"
    exit 1
fi
config_json="$(realpath "$config_json")"

# Read a required string; error out if missing
jreq() {
    local val
    val=$(jq -r "$1 // empty" "$config_json")
    if [ -z "$val" ]; then
        echo "config: $1 is required"
        exit 1
    fi
    printf '%s' "$val"
}

# Read an optional value; return $2 if absent
# Note: uses `if == null` (not the `//` operator) so that `false` is preserved
# instead of being replaced by the default.
jopt() { jq -r "if $1 == null then $2 else $1 end" "$config_json"; }

# Shared parameters
DATA_JSON=$(jreq '.data_json')
NNUNET_PLANNER=$(jreq '.nnunet_planner' )
NNUNET_PLANS=$(jreq   '.nnunet_plans' )
CONFIGURATION=$(jreq  '.configuration' )

# Dataset parameters
DATASET_ID=$(jreq   '.dataset_id' )
DATASET_NAME=$(jreq '.dataset_name' )

# Training parameters
NNUNET_TRAINER=$(jreq        '.nnunet_trainer' )
NNUNET_TRAINER_CONFIG=$(jopt '.nnunet_trainer_config' 'empty')
NNUNET_FOLDER_NAME=$(jreq  '.nnunet_folder_name' )

# Inference parameters
ALL_WEIGHTS=$(jopt '.inference_all_weights' false)
DO_PREPROCESSING=$(jopt '.inference_do_preprocessing' true)

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

# Set the device to cpu if cuda is not available
DEVICE=${SMAUGBENCH_DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}

FOLD=0 # Set the fold to 0: TODO: Add support for training on multiple folds

echo ""
echo "Running inference with the following parameters:"
echo "Config: $config_json"
echo "Data: $DATA_JSON"
echo "nnUNetTrainer=${NNUNET_TRAINER}"
echo "nnUNetPlanner=${NNUNET_PLANNER}"
echo "nnUNetPlans=${NNUNET_PLANS}"
echo "configuration=${CONFIGURATION}"
echo "nnUNetFolderName=${NNUNET_FOLDER_NAME}"
echo "ALL_WEIGHTS=${ALL_WEIGHTS}"
echo "DO_PREPROCESSING=${DO_PREPROCESSING}"
echo "JOBSNN=${JOBSNN}"
echo "DEVICE=${DEVICE}"
echo "DATASET_ID=${DATASET_ID}"
echo "FOLD=${FOLD}"
echo ""

# Get data_json path
if [ ! -f "$DATA_JSON" ]; then
    DATA_JSON="$SMAUGBENCH/smaugbench/datasets/$(basename "$DATA_JSON").json"
    if [ ! -f "$DATA_JSON" ]; then
        echo "Error: Could not find data JSON file: $DATA_JSON"
        exit 1
    fi
fi

SRC_DATASET=Dataset${DATASET_ID}_${DATASET_NAME}
inference_dir="$SMAUGBENCH_DATA"/inference
IMAGES_DIR="$inference_dir"/"$SRC_DATASET"/images
LABELS_DIR="$inference_dir"/"$SRC_DATASET"/labels

# Auto-enable images-only when the first TESTING entry has no LABEL
if [ "$IMAGES_ONLY" != "true" ]; then
    first_label=$(jq -r '.TESTING[0].LABEL // empty' "$DATA_JSON")
    if [ -z "$first_label" ]; then
        echo "No LABEL found in first TESTING entry: enabling --images-only mode"
        IMAGES_ONLY=true
    fi
fi

if [ "$DO_PREPROCESSING" = "true" ] || [ ! -d "$inference_dir" ]; then
    echo "Preprocess inference data"
    # Preprocessing parameters
    ORIENTATION="RPI"
    RESOLUTION="1x1x1"

    # Set the paths to the inference data folder
    bids="$SMAUGBENCH_DATA"/bids
    CURR_DIR="$(realpath .)"
    cd "$bids"

    # Create inference dir
    mkdir -p "$inference_dir"

    # Prepare inference data
    echo "Creating inference folders"
    mkdir -p "$IMAGES_DIR"
    if [ "$IMAGES_ONLY" != "true" ]; then
        mkdir -p "$LABELS_DIR"

        # Copy data in inference folders
        jq -r '.TESTING[].LABEL' "$DATA_JSON" | xargs -I{} cp "{}" "$LABELS_DIR/"
    fi

    # Copy images and add nnUNet suffix _0000 for inference
    jq -r '.TESTING[].IMAGE' "$DATA_JSON" | while IFS= read -r img; do
        cp "$img" "$IMAGES_DIR/$(basename "${img/%.nii.gz/_0000.nii.gz}")"
    done

    if [ "$IMAGES_ONLY" != "true" ]; then
        # Remove label suffix from filename
        for img in "$IMAGES_DIR"/*_0000.nii.gz; do
            # Get the filename without the path
            filename=$(basename "$img")
            # Extract the prefix by removing the _0000.nii.gz extension
            prefix="${filename%_0000.nii.gz}"
            # Use nullglob so the array is empty if no files match (instead of containing the literal '*' string)
            shopt -s nullglob
            # Find all labels that start with the prefix and have an underscore afterward
            label_matches=("$LABELS_DIR"/"$prefix"_*.nii.gz)
            shopt -u nullglob
            match_count=${#label_matches[@]}
            if [ "$match_count" -ne 1 ]; then
                echo "Error: Expected exactly one label file for image $filename, but found $match_count."
                exit 1
            fi
            # Rename label file
            mv "${label_matches[0]}" "$LABELS_DIR"/"$prefix".nii.gz
        done
    fi

    # Reorient images to same orientation
    echo "Transform test images to $ORIENTATION"
    smaugbench_reorient_image -i "$IMAGES_DIR" -o "$IMAGES_DIR" -ori $ORIENTATION -r -w $JOBS
    if [ "$IMAGES_ONLY" != "true" ]; then
        smaugbench_reorient_image -i "$LABELS_DIR" -o "$LABELS_DIR" -ori $ORIENTATION -r -w $JOBS
    fi

    # Resample images to a same resolution
    echo "Resample test images to $RESOLUTION"
    smaugbench_resample_image -i "$IMAGES_DIR" -o "$IMAGES_DIR" -res $RESOLUTION -int linear -r -w $JOBS
    if [ "$IMAGES_ONLY" != "true" ]; then
        smaugbench_resample_image -i "$LABELS_DIR" -o "$LABELS_DIR" -res $RESOLUTION -int nn -r -w $JOBS
    fi

    # Move back
    cd "$CURR_DIR"
fi

# Run model inference
export nnUNet_results="$SMAUGBENCH_DATA"/nnUNet/results/"$NNUNET_FOLDER_NAME"
JOBSMEASURE=$(( JOBS < $((MEMGB / 32)) ? JOBS : $((MEMGB / 32)) ))

run_metrics() {
    local pred_dir="$1"
    local metrics_dir="$2"
    echo "Compute pairwise measurements -> $metrics_dir"
    mkdir -p "$metrics_dir"
    jq '.LABELS | to_entries | map({key: .value, value: (.key | tonumber)}) | from_entries' "$DATA_JSON" > "$metrics_dir"/mapping.json
    smaugbench_compute_pairwise_measurements -pred "$pred_dir" -ref "$LABELS_DIR" -pred-map "$metrics_dir"/mapping.json -ref-map "$metrics_dir"/mapping.json -o "$metrics_dir"/metrics.csv -metrics "dsc" "nsd" "hd" -w $JOBSMEASURE
}

if [ "$ALL_WEIGHTS" = "true" ]; then
    weight_folder="$nnUNet_results/$SRC_DATASET/${NNUNET_TRAINER}__${NNUNET_PLANS}__${CONFIGURATION}/fold_${FOLD}"
    echo "Running inference for all weights in $weight_folder"
    shopt -s nullglob
    weights=("$weight_folder"/*.pth)
    shopt -u nullglob
    if [ ${#weights[@]} -eq 0 ]; then
        echo "Error: No .pth weights found in $weight_folder"
        exit 1
    fi
    for weight in "${weights[@]}"; do
        weight_file=$(basename "$weight")
        weight_name="${weight_file%.pth}"
        echo "Running inference for weight: $weight_file"
        PRED_DIR="$inference_dir/$SRC_DATASET/prediction_${NNUNET_FOLDER_NAME}_${weight_name}"
        nnUNetv2_predict -i "$IMAGES_DIR" -o "$PRED_DIR" -d "$DATASET_ID" -tr "$NNUNET_TRAINER" -p "$NNUNET_PLANS" -c "$CONFIGURATION" -f "$FOLD" -device $DEVICE -chk "$weight_file"
        if [ "$IMAGES_ONLY" != "true" ]; then
            METRICS_DIR="$inference_dir/$SRC_DATASET/metrics_${NNUNET_FOLDER_NAME}_${weight_name}"
            run_metrics "$PRED_DIR" "$METRICS_DIR"
        fi
    done
else
    echo "Running inference for the best model in $nnUNet_results"
    weight_name="checkpoint_best.pth"
    PRED_DIR="$inference_dir/$SRC_DATASET/prediction_$NNUNET_FOLDER_NAME"
    nnUNetv2_predict -i "$IMAGES_DIR" -o "$PRED_DIR" -d "$DATASET_ID" -tr "$NNUNET_TRAINER" -p "$NNUNET_PLANS" -c "$CONFIGURATION" -f "$FOLD" -device $DEVICE -chk "$weight_name"
    if [ "$IMAGES_ONLY" != "true" ]; then
        METRICS_DIR="$inference_dir/$SRC_DATASET/metrics_$NNUNET_FOLDER_NAME"
        run_metrics "$PRED_DIR" "$METRICS_DIR"
    fi
fi