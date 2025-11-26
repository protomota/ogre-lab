# Ogre-Lab

Isaac Lab reinforcement learning environment for training navigation controllers for the Project Ogre mecanum drive robot.

## Overview

This project trains a neural network policy to efficiently execute velocity commands for a 4-wheel mecanum drive robot. The trained policy can be used as a local controller within Nav2, replacing DWB/MPPI for smoother, more efficient navigation.

**Architecture:**
- **Nav2 Global Planner**: Computes path on map (unchanged)
- **Trained RL Policy**: Replaces local controller - executes velocity commands efficiently
- **Benefits**: Learns robot dynamics, handles mecanum kinematics optimally, smoother motion

## Prerequisites

- NVIDIA Isaac Sim 4.2+
- Isaac Lab (installed at `~/isaac-lab/IsaacLab`)
- RSL-RL for training
- Robot USD model (`ogre.usd` from ogre-slam repo)

## Quick Start

```bash
# 1. Symlink the environment into Isaac Lab
ln -sf ~/ogre-lab/isaaclab_env/ogre_navigation \
  ~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation

# 2. Activate Isaac Lab conda environment
conda activate env_isaaclab

# 3. Run training
cd ~/isaac-lab/IsaacLab
./scripts/train_ogre_navigation.sh
```

## Installation

### Step 1: Symlink Environment into Isaac Lab

Isaac Lab discovers environments by looking in its `isaaclab_tasks/direct/` directory. Rather than copying files, we create a symbolic link so that changes in ogre-lab are automatically reflected in Isaac Lab.

```bash
ln -sf ~/ogre-lab/isaaclab_env/ogre_navigation \
  ~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation
```

**What this does:**
- Creates a symbolic link named `ogre_navigation` inside Isaac Lab's direct environments folder
- Points to our environment code in `~/ogre-lab/isaaclab_env/ogre_navigation`
- The `-s` flag creates a symbolic (soft) link
- The `-f` flag forces overwrite if a link already exists

**Why symlink instead of copy?**
- Edit code in one place (ogre-lab), changes apply everywhere
- Keep ogre-lab as the source of truth for version control
- Easy to update or remove without touching Isaac Lab installation

### Step 2: Register the Environment

Isaac Lab needs to know about our environment. Add this import to Isaac Lab's direct environments init file:

**File:** `~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py`

```python
from . import ogre_navigation
```

This tells Python to load our `ogre_navigation` module when Isaac Lab starts, which triggers the Gym environment registration in our `__init__.py`.

### Step 3: Activate Isaac Lab Environment

Isaac Lab requires its own conda environment with Isaac Sim Python bindings:

```bash
conda activate env_isaaclab
```

### Step 4: Run Training

```bash
cd ~/isaac-lab/IsaacLab
./scripts/train_ogre_navigation.sh
```

**What happens during training:**
1. Isaac Lab spawns 4096 parallel robot instances in GPU-accelerated simulation
2. Each robot receives random velocity commands (vx, vy, vtheta)
3. The policy network outputs wheel velocities to track those commands
4. Rewards are computed based on tracking accuracy, energy use, and smoothness
5. PPO algorithm updates the policy every 24 steps
6. After 1000 iterations, a trained policy checkpoint is saved

## Training

### Custom Training

```bash
cd ~/isaac-lab/IsaacLab

# Train with 4096 parallel environments for 1000 iterations (headless)
./isaaclab.sh -p scripts/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 4096 \
    --max_iterations 1000 \
    --headless

# Train with visualization (slower but useful for debugging)
./isaaclab.sh -p scripts/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 64 \
    --max_iterations 500
```

### Training Output

Checkpoints saved to: `~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/`

## Environment Details

### Observation Space (10 dimensions)
| Index | Description |
|-------|-------------|
| 0-2 | Target velocity command (vx, vy, vtheta) |
| 3-5 | Current base velocity (vx, vy, vtheta) |
| 6-9 | Current wheel velocities (fl, fr, rl, rr) |

### Action Space (4 dimensions)
| Index | Description |
|-------|-------------|
| 0 | Front-left wheel velocity target |
| 1 | Front-right wheel velocity target |
| 2 | Rear-left wheel velocity target |
| 3 | Rear-right wheel velocity target |

### Reward Function

The policy is trained to maximize:
- **Velocity tracking** (weight: 5.0): Minimize error between target and actual velocity
- **XY velocity accuracy** (weight: 2.0): Extra reward for linear velocity tracking
- **Angular velocity accuracy** (weight: 1.0): Reward for rotational velocity tracking
- **Energy efficiency** (weight: -0.001): Penalize excessive wheel velocities
- **Smoothness** (weight: -0.01): Penalize jerky motion

### Robot Parameters

| Parameter | Value |
|-----------|-------|
| Wheel radius | 40mm |
| Wheelbase | 95mm |
| Track width | 205mm |
| Max linear velocity | 8.0 m/s |
| Max angular velocity | 6.0 rad/s |

## Policy Deployment

After training, the policy can be deployed as a ROS2 node that:
1. Subscribes to Nav2's local planner velocity commands
2. Runs the trained neural network
3. Publishes wheel velocities to motor controller

See `ros2_controller/` for the deployment node (coming soon).

## File Structure

```
ogre-lab/
├── README.md                     # This file
├── isaaclab_env/                 # Isaac Lab environment
│   └── ogre_navigation/
│       ├── __init__.py           # Gym registration
│       ├── ogre_navigation_env.py # Environment definition
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py # PPO training config
├── scripts/
│   └── train_ogre_navigation.sh  # Training launcher
├── ros2_controller/              # ROS2 deployment node (TODO)
└── docs/                         # Additional documentation
```

## Related Projects

- **ogre-slam**: ROS2 SLAM and navigation package for the real robot
- **Isaac Lab**: GPU-accelerated robot learning framework

## License

MIT License
