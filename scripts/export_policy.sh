#!/bin/bash
# Export trained policy to ONNX and JIT formats for deployment
# Usage: ./scripts/export_policy.sh <checkpoint_path>
#
# Example:
#   ./scripts/export_policy.sh ~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/2025-11-26_10-22-58/model_99.pt

CHECKPOINT=${1:-""}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: ./scripts/export_policy.sh <checkpoint_path>"
    echo ""
    echo "Available checkpoints:"
    ls -d ~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/*/ 2>/dev/null | while read dir; do
        echo "  $dir"
        ls "$dir"/*.pt 2>/dev/null | head -3 | sed 's/^/    /'
    done
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

echo "Exporting policy from: $CHECKPOINT"
echo "================================"
echo ""
echo "This will create exported/ folder with:"
echo "  - policy.pt (JIT compiled)"
echo "  - policy.onnx (ONNX format for deployment)"
echo ""

cd ~/isaac-lab/IsaacLab

# Run play.py which exports the model automatically
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 1 \
    --checkpoint "$CHECKPOINT" \
    --headless

EXPORT_DIR=$(dirname "$CHECKPOINT")/exported
if [ -d "$EXPORT_DIR" ]; then
    echo ""
    echo "Export complete!"
    echo "Exported files:"
    ls -la "$EXPORT_DIR"
else
    echo ""
    echo "Warning: Export directory not found. Check for errors above."
fi
