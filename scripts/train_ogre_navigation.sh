#!/bin/bash
# Train Ogre Navigation Policy
# Usage: ./scripts/train_ogre_navigation.sh [num_envs] [max_iterations]

NUM_ENVS=${1:-4096}
MAX_ITERATIONS=${2:-1000}

echo "Training Ogre Navigation Policy"
echo "================================"
echo "Number of environments: $NUM_ENVS"
echo "Max iterations: $MAX_ITERATIONS"
echo ""

# Run training with RSL-RL
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs $NUM_ENVS \
    --max_iterations $MAX_ITERATIONS \
    --headless

echo ""
echo "Training complete!"
echo "Checkpoints saved to: logs/rsl_rl/ogre_navigation/"
