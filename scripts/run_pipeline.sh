#!/bin/bash

# Read a JSON config, run the enabled pipeline steps (download / prepare / train),
# and archive the config next to the training weights.

# BASH SETTINGS
# ======================================================================================================================

# Immediately exit if error
set -e

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# SCRIPT STARTS HERE
# ======================================================================================================================

if [ -z "$SMAUGBENCH_DATA" ]; then
    echo "Please set the SMAUGBENCH_DATA environment variable. This folder will store all preprocessed datasets and training."
    exit 1
fi

config_json="$1"
if [ -z "$config_json" ] || [ ! -f "$config_json" ]; then
    echo "Usage: $0 <config.json>"
    exit 1
fi
config_json="$(realpath "$config_json")"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
SMAUGBENCH="$(realpath "$(dirname "$0")/..")"

# Check if data_json field isRead a required string; error out if missing
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

# Steps default to true
DO_DOWNLOAD=$(jopt '.steps.download' 'true')
DO_PREPARE=$(jopt  '.steps.prepare'  'true')
DO_TRAIN=$(jopt    '.steps.train'    'true')

# Shared parameters
DATA_JSON=$(jreq '.data_json')
NNUNET_PLANNER=$(jopt '.nnunet_planner' '"nnUNetPlannerResEncL"')
NNUNET_PLANS=$(jopt   '.nnunet_plans'   '"nnUNetPlans"')
CONFIGURATION=$(jopt  '.configuration'  '"3d_fullres"')

# Prepare / train parameters (validated per step)
DATASET_ID=$(jreq   '.dataset_id' )
DATASET_NAME=$(jreq '.dataset_name' )

# Train-only parameters
NNUNET_TRAINER=$(jopt        '.nnunet_trainer'        'empty')
NNUNET_TRAINER_CONFIG=$(jopt '.nnunet_trainer_config' 'empty')
NNUNET_FOLDER_NAME=$(jreq  '.nnunet_folder_name' )

echo ""
echo "Pipeline steps: download=$DO_DOWNLOAD prepare=$DO_PREPARE train=$DO_TRAIN"
echo "Config: $config_json"
echo "Data: $DATA_JSON"
echo ""

# Get data_json path
if [ ! -f "$DATA_JSON" ]; then
    DATA_JSON="$SMAUGBENCH/smaugbench/datasets/$(basename "$DATA_JSON").json"
    if [ ! -f "$DATA_JSON" ]; then
        echo "Error: Could not find data JSON file: $DATA_JSON"
        exit 1
    fi
fi

# 1. Download
if [ "$DO_DOWNLOAD" = "true" ]; then
    echo "==> download_datasets.sh"
    "$SCRIPT_DIR/download_datasets.sh" "$DATA_JSON"
fi

# 2. Prepare
if [ "$DO_PREPARE" = "true" ]; then
    if [ -z "$DATASET_ID" ] || [ -z "$DATASET_NAME" ]; then
        echo "config: .dataset_id and .dataset_name are required when steps.prepare=true"
        exit 1
    fi
    echo "==> prepare_datasets.sh"
    "$SCRIPT_DIR/prepare_datasets.sh" "$DATA_JSON" "$DATASET_ID" "$DATASET_NAME" \
        "$NNUNET_PLANNER" "$NNUNET_PLANS" "$CONFIGURATION"
fi

# 3. Train
export nnUNet_results="$SMAUGBENCH_DATA"/nnUNet/results/"$NNUNET_FOLDER_NAME"
if [ "$DO_TRAIN" = "true" ]; then
    if [ -z "$DATASET_ID" ] || [ -z "$NNUNET_TRAINER" ]; then
        echo "config: .dataset_id and .nnunet_trainer are required when steps.train=true"
        exit 1
    fi
    echo "==> train.sh"
    if [[ "$NNUNET_TRAINER" == nnUNetTrainerDAExt* ]]; then
        if [ -z "$NNUNET_TRAINER_CONFIG" ]; then
            echo "config: .nnunet_trainer_config is required for DAExt trainers"
            exit 1
        fi
        "$SCRIPT_DIR/train.sh" "$DATASET_ID" "$NNUNET_TRAINER" "$NNUNET_TRAINER_CONFIG" \
            "$NNUNET_PLANNER" "$NNUNET_PLANS" "$CONFIGURATION"
    else
        "$SCRIPT_DIR/train.sh" "$DATASET_ID" "$NNUNET_TRAINER" \
            "$NNUNET_PLANNER" "$NNUNET_PLANS" "$CONFIGURATION"
    fi

    # Archive config next to the weights (fold_0 is hardcoded in train.sh)
    FOLD_DIR="$nnUNet_results/Dataset${DATASET_ID}_${DATASET_NAME}/${NNUNET_TRAINER}__${NNUNET_PLANS}__${CONFIGURATION}/fold_0"
    if [ -d "$FOLD_DIR" ]; then
        cp "$config_json" "$FOLD_DIR/config.json"
        echo "Config archived to $FOLD_DIR/config.json"
        cp "$DATA_JSON" "$FOLD_DIR/data.json"
        echo "Data archived to $FOLD_DIR/data.json"
        pip freeze > "$FOLD_DIR/requirements.txt"
        echo "Requirements archived to $FOLD_DIR/requirements.txt"
    else
        echo "Warning: expected fold folder $FOLD_DIR does not exist; config not archived."
    fi
fi
