#!/bin/bash
# Sync training environment from ogre-lab (version controlled) to Isaac Lab (training)
#
# This ensures the Isaac Lab training uses the same environment code that's
# tracked in version control. Run this before training to apply any changes.
#
# Usage:
#   ./scripts/sync_env.sh          # Sync and show diff
#   ./scripts/sync_env.sh --quiet  # Sync without diff output

set -e

# Paths
OGRE_LAB_ENV=~/ogre-lab/isaaclab_env/ogre_navigation
ISAACLAB_ENV=~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation

QUIET=""
if [ "$1" == "--quiet" ]; then
    QUIET=1
fi

# Check source exists
if [ ! -d "$OGRE_LAB_ENV" ]; then
    echo "Error: Source directory not found: $OGRE_LAB_ENV"
    exit 1
fi

# Check target exists
if [ ! -d "$ISAACLAB_ENV" ]; then
    echo "Error: Target directory not found: $ISAACLAB_ENV"
    echo "Is Isaac Lab installed at ~/isaac-lab/IsaacLab?"
    exit 1
fi

# Show what's different before sync
if [ -z "$QUIET" ]; then
    echo "Syncing training environment:"
    echo "  From: $OGRE_LAB_ENV (version controlled)"
    echo "  To:   $ISAACLAB_ENV (Isaac Lab)"
    echo ""

    # Check for differences
    if diff -q "$OGRE_LAB_ENV/ogre_navigation_env.py" "$ISAACLAB_ENV/ogre_navigation_env.py" > /dev/null 2>&1; then
        echo "Files are already in sync."
    else
        echo "Changes to be applied:"
        echo "----------------------------------------"
        diff "$ISAACLAB_ENV/ogre_navigation_env.py" "$OGRE_LAB_ENV/ogre_navigation_env.py" || true
        echo "----------------------------------------"
        echo ""
    fi
fi

# Sync the environment file
cp "$OGRE_LAB_ENV/ogre_navigation_env.py" "$ISAACLAB_ENV/ogre_navigation_env.py"

# Sync __init__.py if it exists
if [ -f "$OGRE_LAB_ENV/__init__.py" ]; then
    cp "$OGRE_LAB_ENV/__init__.py" "$ISAACLAB_ENV/__init__.py"
fi

# Sync agents config if it exists
if [ -d "$OGRE_LAB_ENV/agents" ]; then
    cp -r "$OGRE_LAB_ENV/agents/"* "$ISAACLAB_ENV/agents/" 2>/dev/null || true
fi

# Clear Python cache to ensure fresh code is used
rm -rf "$ISAACLAB_ENV/__pycache__" 2>/dev/null || true

if [ -z "$QUIET" ]; then
    echo "Sync complete!"
    echo ""
    echo "Files synced:"
    ls -la "$ISAACLAB_ENV"/*.py
fi
