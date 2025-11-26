# Ogre-Lab

Isaac Lab reinforcement learning environment for training navigation controllers for the Project Ogre mecanum drive robot.

## Overview

This project trains a neural network policy to efficiently execute velocity commands for a 4-wheel mecanum drive robot. The trained policy can be used as a local controller within Nav2, replacing DWB/MPPI for smoother, more efficient navigation.

**Architecture:**
- **Nav2 Global Planner**: Computes path on map (unchanged)
- **Trained RL Policy**: Replaces local controller - executes velocity commands efficiently
- **Benefits**: Learns robot dynamics, handles mecanum kinematics optimally, smoother motion

## Project Status / TODO

- [x] Install Isaac Lab prerequisites and framework
- [x] Create mecanum robot environment in Isaac Lab
- [x] Define observation and action spaces for navigation policy
- [x] Design reward function for efficient navigation
- [x] Set up ogre-lab repo structure
- [x] Train RL policy using RSL-RL
- [x] Export trained policy for deployment (ONNX + JIT)
- [ ] Create ROS2 node to run policy as Nav2 controller

## Prerequisites

- NVIDIA Isaac Sim 5.0+
- Isaac Lab (installed at `~/isaac-lab/IsaacLab`)
- RSL-RL for training
- Robot USD model (`ogre_robot.usd` from ogre-slam repo - robot-only, no scene)
- Conda environment `env_isaaclab`

## Quick Start

```bash
# 1. Activate Isaac Lab conda environment
conda activate env_isaaclab

# 2. Symlink the environment into Isaac Lab (first time only)
ln -sf ~/ogre-lab/isaaclab_env/ogre_navigation \
  ~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation

# 3. Run training (headless)
cd ~/isaac-lab/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 1024 \
    --max_iterations 1000 \
    --headless
```

## Installation

### Step 1: Activate Isaac Lab Conda Environment

Isaac Lab requires its own conda environment with Isaac Sim Python bindings:

```bash
conda activate env_isaaclab
```

### Step 2: Symlink Environment into Isaac Lab

Isaac Lab discovers environments by looking in its `isaaclab_tasks/direct/` directory. Create a symbolic link so changes in ogre-lab are automatically reflected:

```bash
ln -sf ~/ogre-lab/isaaclab_env/ogre_navigation \
  ~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation
```

### Step 3: Register the Environment

Add this import to Isaac Lab's direct environments init file (if not already present):

**File:** `~/isaac-lab/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/__init__.py`

```python
from . import ogre_navigation
```

## Training

### Headless Training (Fast, No GUI)

```bash
conda activate env_isaaclab
cd ~/isaac-lab/IsaacLab

# Train with 1024 parallel environments for 1000 iterations
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 1024 \
    --max_iterations 1000 \
    --headless
```

### Training with Visualization (Debug Mode)

```bash
conda activate env_isaaclab
cd ~/isaac-lab/IsaacLab

# Train with 64 environments and GUI to watch training
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 64 \
    --max_iterations 100
```

### Training Output

- Checkpoints saved to: `~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/<timestamp>/`
- Models saved every 100 iterations: `model_0.pt`, `model_100.pt`, etc.
- Final model: `model_<max_iterations-1>.pt`

### What to Look For During Training

```
Learning iteration 50/100
  Mean reward: 1.25        # Should be positive and increasing
  Mean episode length: 299  # Should be ~300 (10s episodes at 30Hz)
```

- **Positive increasing reward** = learning is working
- **Negative reward** = reward function needs tuning
- **Short episodes** = robots falling or crashing

## Testing a Trained Model

### View Policy in Isaac Sim

```bash
conda activate env_isaaclab
cd ~/isaac-lab/IsaacLab

# Test with 16 robots visualized
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/ogre_navigation/<run_folder>/model_99.pt
```

Replace `<run_folder>` with your training run timestamp (e.g., `2025-11-26_10-22-58`).

### List Available Checkpoints

```bash
ls ~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/
```

## Exporting for Deployment

Running `play.py` automatically exports the model to deployment formats:

```bash
conda activate env_isaaclab
cd ~/isaac-lab/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/ogre_navigation/<run_folder>/model_99.pt \
    --headless
```

This creates an `exported/` folder with:
- `policy.pt` - JIT compiled PyTorch model
- `policy.onnx` - ONNX format for cross-platform deployment

Pre-exported models are available in `models/` directory.

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
- **Velocity tracking** (weight: 1.0): Minimize error between target and actual velocity
- **XY velocity accuracy** (weight: 0.5): Extra reward for linear velocity tracking
- **Angular velocity accuracy** (weight: 0.25): Reward for rotational velocity tracking
- **Energy efficiency** (weight: -0.0001): Small penalty for excessive wheel velocities
- **Smoothness** (weight: -0.001): Small penalty for jerky motion

### Robot Parameters

| Parameter | Value |
|-----------|-------|
| Wheel radius | 40mm |
| Wheelbase | 95mm |
| Track width | 205mm |
| Max linear velocity | 0.5 m/s |
| Max angular velocity | 1.0 rad/s |
| Action scale | 10.0 rad/s |

## Policy Deployment (TODO)

After training, the policy can be deployed as a ROS2 node that:
1. Subscribes to `/cmd_vel` (geometry_msgs/Twist)
2. Runs the trained neural network (ONNX or JIT)
3. Publishes wheel velocities to motor controller

See `ros2_controller/` for the deployment node (coming soon).

## File Structure

```
ogre-lab/
├── README.md                     # This file
├── models/                       # Exported models for deployment
│   ├── policy.pt                 # JIT compiled model
│   └── policy.onnx               # ONNX model
├── isaaclab_env/                 # Isaac Lab environment
│   └── ogre_navigation/
│       ├── __init__.py           # Gym registration
│       ├── ogre_navigation_env.py # Environment definition
│       └── agents/
│           ├── __init__.py
│           └── rsl_rl_ppo_cfg.py # PPO training config
├── scripts/
│   ├── train_ogre_navigation.sh  # Training launcher
│   └── export_policy.sh          # Export helper script
├── ros2_controller/              # ROS2 deployment node (TODO)
└── docs/                         # Additional documentation
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'isaaclab'"
Activate the conda environment: `conda activate env_isaaclab`

### "Patch buffer overflow detected"
Reduce `num_envs` or increase `gpu_max_rigid_patch_count` in the environment config.

### Very negative rewards (-20000)
The velocity targets are too aggressive. Reduce `max_lin_vel` and `max_ang_vel` in the config.

### Robots flying or exploding
Check the USD file has proper physics setup. Use robot-only USD without scene elements.

## Related Projects

- **ogre-slam**: ROS2 SLAM and navigation package for the real robot
- **Isaac Lab**: GPU-accelerated robot learning framework

## License

MIT License
