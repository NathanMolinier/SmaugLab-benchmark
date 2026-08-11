#!/bin/bash

# This script train nnUNet models.
# Based on https://github.com/neuropoly/totalspineseg/blob/main/scripts/train.sh

# The script expects the following environment variables to be set:
#   SMBENCH_DATA: The path to the data folder where data and weights will be stored.
#   SMBENCH_JOBS: The number of CPU cores to use. Default is the number of CPU cores available.
#   SMBENCH_JOBSNN: The number of jobs to use for the nnUNet. Default is the number of CPU cores available or the available memory in GB divided by 8, whichever is smaller.
#   SMBENCH_DEVICE: The device to use. Default is "cuda" if available, otherwise "cpu".

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

# Get the number of CPUs
CORES=${SLURM_JOB_CPUS_PER_NODE:-$(lscpu -p | egrep -v '^#' | wc -l)}

# Get memory in GB
MEMGB=$(awk '/MemTotal/ {print int($2/1024/1024)}' /proc/meminfo)

# Set the number of jobs
JOBS=${SMBENCH_JOBS:-$CORES}

# Set the number of jobs for the nnUNet
JOBSNN=$(( JOBS < $((MEMGB / 8)) ? JOBS : $((MEMGB / 8)) ))
JOBSNN=$(( JOBSNN < 1 ? 1 : JOBSNN ))
JOBSNN=${SMBENCH_JOBSNN:-$JOBSNN}

# Set the device to cpu if cuda is not available
DEVICE=${SMBENCH_DEVICE:-$(python3 -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}

# Set variables
DATASET_ID="$1"
nnUNetTrainer="$2"
shift 2 # Consume the first two arguments. The old $3 is now $1.

if [[ "$nnUNetTrainer" == nnUNetTrainerDAExt* ]]; then
    nnUNetTrainer_config="$1"
    if [ -z "$nnUNetTrainer_config" ] || [ ! "${nnUNetTrainer_config}" == *.json ]; then # Check if the config exist and it ends with .json
        echo "Please provide a configuration for the nnUNetTrainerDAExt trainer."
        exit 1
    fi
    shift 1 # Consume the config argument.
fi
FOLD=0 # Set the fold to 0: TODO: Add support for training on multiple folds
nnUNetPlanner=${1:-nnUNetPlannerResEncL}
nnUNetPlans=${2:-nnUNetPlans}
configuration=${3:-3d_fullres}
data_identifier=${nnUNetPlans}_${configuration}

# Set nnunet params
export nnUNet_def_n_proc=$JOBSNN
export nnUNet_n_proc_DA=$JOBSNN
export nnUNet_raw="$SMBENCH_DATA"/nnUNet/raw
export nnUNet_preprocessed="$SMBENCH_DATA"/nnUNet/preprocessed
export nnUNet_results="$SMBENCH_DATA"/nnUNet/results

# Copy smauglab trainer to nnunet folder
if [[ "$nnUNetTrainer" == "nnUNetTrainerDAExt*" ]]; then
    smauglab_add_nnunettrainer -t nnUNetTrainerDAExt --overwrite
fi

echo ""
echo "Running with the following parameters:"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "nnUNetTrainer=${nnUNetTrainer}"
echo "nnUNetPlanner=${nnUNetPlanner}"
echo "nnUNetPlans=${nnUNetPlans}"
echo "configuration=${configuration}"
echo "data_identifier=${data_identifier}"
echo "JOBSNN=${JOBSNN}"
echo "DEVICE=${DEVICE}"
echo "DATASET_ID=${DATASET_ID}"
echo "FOLD=${FOLD}"
echo ""

# Start training
echo "Start training"
if [[ "$nnUNetTrainer" == "nnUNetTrainerDAExt*" ]]; then
    SMAUGLAB_PARAMS_GPU_JSON=$(realpath $nnUNetTrainer_config) nnUNetv2_train $DATASET_ID $configuration $FOLD -tr $nnUNetTrainer -p $nnUNetPlans --c -device $DEVICE
else
    nnUNetv2_train $DATASET_ID $configuration $FOLD -tr $nnUNetTrainer -p $nnUNetPlans --c -device $DEVICE
fi
