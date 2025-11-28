#!/bin/bash
# Deploy the most recent trained model from Isaac Lab to ogre-lab
#
# Usage:
#   ./scripts/deploy_model.sh              # Copy latest model
#   ./scripts/deploy_model.sh --rebuild    # Copy and rebuild ROS2 package

set -e

# Paths
ISAACLAB_LOGS=~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation
OGRE_LAB_MODELS=~/ogre-lab/ros2_controller/models
ROS2_WS=~/ros2_ws

# Find the most recent training run
LATEST_RUN=$(ls -t "$ISAACLAB_LOGS" 2>/dev/null | head -1)

if [ -z "$LATEST_RUN" ]; then
    echo "Error: No training runs found in $ISAACLAB_LOGS"
    exit 1
fi

EXPORTED_DIR="$ISAACLAB_LOGS/$LATEST_RUN/exported"

if [ ! -d "$EXPORTED_DIR" ]; then
    echo "Error: No exported models found in $EXPORTED_DIR"
    echo "Run play.py first to export the model:"
    echo "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\"
    echo "      --task Isaac-Ogre-Navigation-Direct-v0 --num_envs 1 --headless \\"
    echo "      --checkpoint logs/rsl_rl/ogre_navigation/$LATEST_RUN/model_*.pt"
    exit 1
fi

# Check for ONNX model
if [ ! -f "$EXPORTED_DIR/policy.onnx" ]; then
    echo "Error: policy.onnx not found in $EXPORTED_DIR"
    exit 1
fi

# Show what we're copying
echo "Deploying model from: $LATEST_RUN"
echo "  Source: $EXPORTED_DIR"
echo "  Target: $OGRE_LAB_MODELS"

# Copy models
cp "$EXPORTED_DIR/policy.onnx" "$OGRE_LAB_MODELS/"
if [ -f "$EXPORTED_DIR/policy.pt" ]; then
    cp "$EXPORTED_DIR/policy.pt" "$OGRE_LAB_MODELS/"
fi

echo "Copied:"
ls -la "$OGRE_LAB_MODELS"/policy.*

# Rebuild ROS2 package if requested
if [ "$1" == "--rebuild" ]; then
    echo ""
    echo "Rebuilding ROS2 package..."
    cd "$ROS2_WS"
    colcon build --packages-select ogre_policy_controller --symlink-install
    echo "Done! Source the workspace before running:"
    echo "  source $ROS2_WS/install/setup.bash"
else
    echo ""
    echo "Model copied. To rebuild ROS2 package, run:"
    echo "  ./scripts/deploy_model.sh --rebuild"
    echo "Or manually:"
    echo "  cd $ROS2_WS && colcon build --packages-select ogre_policy_controller --symlink-install"
fi
