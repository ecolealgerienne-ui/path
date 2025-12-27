#!/bin/bash

################################################################################
# Wrapper Script - Activate Conda and Run V13 Pipeline
################################################################################

set -e

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate cellvit environment
conda activate cellvit

# Run the pipeline
bash scripts/run_v13_smart_crops_pipeline.sh "$@"
