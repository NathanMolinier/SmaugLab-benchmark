#!/bin/bash

# Script based on https://github.com/neuropoly/totalspineseg/blob/main/scripts/download_datasets.sh

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
    data_json="$SMBENCH/smbench/datasets/$(basename "$1" .sh).json"
    if [ ! -f "$data_json" ]; then
        echo "Error: Could not find data JSON file."
        exit 1
    fi
fi

# Set the paths to the BIDS data folders
bids="$SMBENCH_DATA"/bids

# Make sure $SMBENCH_DATA/bids exists and enter it
mkdir -p "$bids"
CURR_DIR="$(realpath .)"
cd "$bids"

# Fetch sources and commits from the data_json file
datasets=($(jq -r '.SOURCE[]' "$data_json"))
commits=($(jq -r '.COMMIT[]' "$data_json"))

# Clone datasets and checkout on the right branch
for i in "${!datasets[@]}"; do
    ds=${datasets[i]}
    commit=${commits[i]}
    dsn=$(basename $ds .git)

    # Clone the dataset from the specified repository
    git clone "$ds"

    # Enter the dataset directory
    cd "$dsn"

    # Checkout on the commit
    git checkout "$commit"

    # Move back to the parent directory to process the next dataset
    cd ..
done

keys=(
    IMAGE
    LABEL
)

# Download necessary data from git annex
for key in "${keys[@]}"; do
    for path in $(jq -r ".TRAINING | .[].$key" "$data_json"); do
        IFS='/' read -r rep_path rel_path <<< "$path"
        git -C "$rep_path" annex get "$rel_path"
    done
done

for key in "${keys[@]}"; do
    for path in $(jq -r ".VALIDATION | .[].$key" "$data_json"); do
        IFS='/' read -r rep_path rel_path <<< "$path"
        git -C "$rep_path" annex get "$rel_path"
    done
done

for key in "${keys[@]}"; do
    for path in $(jq -r ".TESTING | .[].$key" "$data_json"); do
        IFS='/' read -r rep_path rel_path <<< "$path"
        git -C "$rep_path" annex get "$rel_path"
    done
done

# Return to the original working directory
cd "$CURR_DIR"