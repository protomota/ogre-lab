#!/bin/bash
# Train Ogre Navigation Policy
#
# This script syncs the training environment from ogre-lab to Isaac Lab,
# then runs training. This ensures any changes in version control are
# applied before training starts.
#
# Usage:
#   ./scripts/train_ogre_navigation.sh                    # Default: 4096 envs, 1000 iterations
#   ./scripts/train_ogre_navigation.sh 1024 500           # Custom envs and iterations
#   ./scripts/train_ogre_navigation.sh --no-sync 4096     # Skip sync (use existing Isaac Lab env)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate Isaac Lab conda environment (Isaac Lab 5.1)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Parse arguments
NO_SYNC=""
NUM_ENVS=4096
MAX_ITERATIONS=1000
NUM_ENVS_SET=""

for arg in "$@"; do
    if [ "$arg" == "--no-sync" ]; then
        NO_SYNC=1
    elif [[ "$arg" =~ ^[0-9]+$ ]]; then
        if [ -z "$NUM_ENVS_SET" ]; then
            NUM_ENVS=$arg
            NUM_ENVS_SET=1
        else
            MAX_ITERATIONS=$arg
        fi
    fi
done

echo "=========================================="
echo "  Ogre Navigation Policy Training"
echo "=========================================="
echo ""

# Sync training environment first
if [ -z "$NO_SYNC" ]; then
    echo "Step 1: Syncing training environment..."
    "$SCRIPT_DIR/sync_env.sh"
    echo ""
else
    echo "Step 1: Skipping sync (--no-sync specified)"
    echo ""
fi

echo "Step 2: Starting training..."
echo "  Environments: $NUM_ENVS"
echo "  Max iterations: $MAX_ITERATIONS"
echo ""

# Change to Isaac Lab directory
cd ~/isaac-lab/IsaacLab

# Run training with RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --headless

echo ""
echo "=========================================="
echo "  Training Complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to:"
echo "  ~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/"
echo ""
echo "Next steps:"
echo "  1. Export: ./scripts/export_policy.sh --headless"
echo "  2. Deploy: ./scripts/deploy_model.sh --rebuild"
