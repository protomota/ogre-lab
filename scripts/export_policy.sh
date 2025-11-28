#!/bin/bash
# Export the trained RL policy from Isaac Lab
#
# This script runs play.py to export the trained model to ONNX and JIT formats.
# The exported models are saved to the training run's exported/ directory.
#
# Usage:
#   ./scripts/export_policy.sh              # Export latest model with visualization (16 robots)
#   ./scripts/export_policy.sh --headless   # Export latest model headless (faster)
#   ./scripts/export_policy.sh <run_name>   # Export specific training run
#   ./scripts/export_policy.sh <run_name> --headless
#
# Examples:
#   ./scripts/export_policy.sh 2025-11-28_10-14-31
#   ./scripts/export_policy.sh 2025-11-28_10-14-31 --headless
#
# Output:
#   Creates policy.onnx and policy.pt in logs/rsl_rl/ogre_navigation/<run>/exported/

set -e

# Paths
ISAACLAB_DIR=~/isaac-lab/IsaacLab
LOGS_DIR="$ISAACLAB_DIR/logs/rsl_rl/ogre_navigation"

# Parse arguments
HEADLESS=""
RUN_NAME=""

for arg in "$@"; do
    if [ "$arg" == "--headless" ]; then
        HEADLESS="--headless"
    else
        RUN_NAME="$arg"
    fi
done

# Find training run
if [ -z "$RUN_NAME" ]; then
    # Use most recent run
    RUN_NAME=$(ls -t "$LOGS_DIR" 2>/dev/null | head -1)
    if [ -z "$RUN_NAME" ]; then
        echo "Error: No training runs found in $LOGS_DIR"
        exit 1
    fi
    echo "Using latest training run: $RUN_NAME"
else
    if [ ! -d "$LOGS_DIR/$RUN_NAME" ]; then
        echo "Error: Training run not found: $LOGS_DIR/$RUN_NAME"
        echo ""
        echo "Available runs:"
        ls -lt "$LOGS_DIR" | head -10
        exit 1
    fi
fi

RUN_DIR="$LOGS_DIR/$RUN_NAME"

# Find the latest model checkpoint
CHECKPOINT=$(ls -t "$RUN_DIR"/model_*.pt 2>/dev/null | head -1)

if [ -z "$CHECKPOINT" ]; then
    echo "Error: No model checkpoints found in $RUN_DIR"
    exit 1
fi

echo "Exporting policy from: $RUN_NAME"
echo "  Checkpoint: $(basename $CHECKPOINT)"
echo "  Mode: ${HEADLESS:-visualization (16 robots)}"
echo ""

# Set number of environments based on mode
if [ -n "$HEADLESS" ]; then
    NUM_ENVS=1
else
    NUM_ENVS=16
fi

# Activate conda environment and run export
cd "$ISAACLAB_DIR"

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Run play.py to export
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs $NUM_ENVS \
    --checkpoint "$CHECKPOINT" \
    $HEADLESS

# Check if export succeeded
EXPORTED_DIR="$RUN_DIR/exported"
if [ -f "$EXPORTED_DIR/policy.onnx" ]; then
    echo ""
    echo "Export successful!"
    echo "  ONNX: $EXPORTED_DIR/policy.onnx"
    if [ -f "$EXPORTED_DIR/policy.pt" ]; then
        echo "  JIT:  $EXPORTED_DIR/policy.pt"
    fi
    echo ""
    echo "Next step: Deploy to ogre-lab"
    echo "  ./scripts/deploy_model.sh --rebuild"
else
    echo ""
    echo "Warning: Export may have failed - policy.onnx not found"
    echo "Check $EXPORTED_DIR for output files"
fi
