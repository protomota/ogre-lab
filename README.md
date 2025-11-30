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
- [x] Create ROS2 node to run policy as Nav2 controller

## Prerequisites

- NVIDIA Isaac Sim 5.0+
- Isaac Lab (installed at `~/isaac-lab/IsaacLab`)
- RSL-RL for training
- Robot USD model (`ogre_robot.usd` from ogre-slam repo - robot-only, no scene)
- Conda environment `env_isaaclab`

### Required Directory Structure

For all scripts to work correctly, use these exact paths:

```
~/
├── isaac-lab/
│   └── IsaacLab/                  # Isaac Lab installation (REQUIRED PATH)
│       ├── isaaclab.sh
│       └── source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation/
│                                  # Training environment (symlinked from ogre-lab)
├── ogre-lab/                      # This repository
│   └── scripts/
│       ├── export_policy.sh       # Export trained model
│       └── deploy_model.sh        # Deploy to ROS2
├── ros2_ws/                       # ROS2 workspace
│   └── src/
│       ├── ogre-slam/             # SLAM and navigation
│       │   ├── usds/ogre_robot.usd  # Robot USD (used by training)
│       │   └── ogre_policy_controller/  # ROS2 policy controller
│       └── ogre_policy_controller/  # Symlink to above
└── miniconda3/
    └── envs/
        └── env_isaaclab/          # Isaac Lab conda environment
```

| Component | Required Path | Notes |
|-----------|---------------|-------|
| Isaac Lab | `~/isaac-lab/IsaacLab/` | Scripts expect this path |
| Isaac Sim | pip installed in `env_isaaclab` | `pip install isaacsim-rl` |
| ROS2 Workspace | `~/ros2_ws/` | Standard ROS2 layout |
| Robot USD | `~/ros2_ws/src/ogre-slam/usds/ogre_robot.usd` | Referenced by training |
| Policy Controller | `~/ros2_ws/src/ogre-slam/ogre_policy_controller/` | ROS2 package |

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

### Monitoring Training with TensorBoard

When training a policy, you can monitor progress in real-time using TensorBoard.

**Step 1:** Open a new terminal (keep your training running in the current one)

**Step 2:** Run these commands:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
tensorboard --logdir ~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/ \
    --reload_multifile=true --reload_interval=5
```

**Note:** The `--reload_multifile=true --reload_interval=5` flags ensure TensorBoard properly tracks updates from RSL-RL's multi-file logging structure.

**Step 3:** Open your web browser and go to: http://localhost:6006

**What you'll see:**

The TensorBoard dashboard shows several graphs. The most important ones for your training:

1. **Train/mean_reward** - This is your main progress indicator. It should go up over time. Higher is better.
2. **Train/mean_episode_length** - How long episodes last. For 10-second episodes, this should approach 300 steps (at 30 Hz) as the policy improves.
3. **Loss/value_function** - Should generally decrease as the critic learns.
4. **Loss/surrogate** - Policy loss, will fluctuate but should stabilize.

**Tips:**
- Click the refresh button (⟳) in TensorBoard to see latest data
- Use the smoothing slider on the left to make noisy graphs easier to read
- You can zoom in on graphs by clicking and dragging

Training logs are saved in timestamped directories under `logs/rsl_rl/ogre_navigation/`.

## Testing and Deploying a Trained Model

### Step 1: Export the Policy

Use the export script to visualize and export the trained model:

```bash
cd ~/ogre-lab

# Export latest model with visualization (16 robots)
./scripts/export_policy.sh

# OR export headless (faster)
./scripts/export_policy.sh --headless

# Export a specific training run
./scripts/export_policy.sh 2025-11-28_10-14-31 --headless
```

This runs `play.py` and creates `policy.onnx` and `policy.pt` in the training run's `exported/` directory.

See [IMPLEMENTATION.md](docs/IMPLEMENTATION.md#export-and-deploy-scripts) for manual export commands.

### Step 2: Deploy to ROS2

Use the deployment script to copy the latest exported model:

```bash
cd ~/ogre-lab

# Copy latest model
./scripts/deploy_model.sh

# Copy and rebuild ROS2 package
./scripts/deploy_model.sh --rebuild
```

The script automatically finds the most recent training run and copies the exported models. See [IMPLEMENTATION.md](docs/IMPLEMENTATION.md#manual-model-export-reference) for manual export steps.

### Step 3: Test with Isaac Sim or Real Robot

See the **[ogre-slam README](https://github.com/protomota/ogre-slam#using-the-trained-rl-policy)** for instructions on testing the deployed policy with:
- Isaac Sim simulation
- Real robot hardware
- Nav2 integration

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

| Parameter | Value | Notes |
|-----------|-------|-------|
| Wheel radius | 40mm (0.04m) | |
| Wheelbase | 95mm | Front-rear axle distance |
| Track width | 205mm | Left-right wheel distance |
| **action_scale** | **20.0 rad/s** | Max wheel velocity - tested stable value |
| Max linear velocity | 1.0 m/s | Target velocity range for training |
| Max angular velocity | 2.0 rad/s | Target angular velocity range |

### Velocity Configuration (CRITICAL)

**NOTE:** Testing showed 6.0 rad/s is too slow, 200 rad/s causes flipping. 20.0 rad/s is stable.

These parameters must match between training and deployment:

```
action_scale = 20.0       # Max wheel velocity in rad/s (policy outputs ×20)
max_lin_vel = 1.0         # Target velocity range for training
max_ang_vel = 2.0         # Target angular velocity range
```

**Math:**
- Policy outputs values in range [-1, 1]
- Wheel velocity = policy_output × action_scale = ±20 rad/s max
- Body velocity = wheel_velocity × wheel_radius = 20 × 0.04 = 0.8 m/s max (theoretical)

**Files that must have matching values:**
1. `ogre_navigation_env.py` - training config
2. `policy_controller_params.yaml` - deployment config

### Training Notes

**Rewards:**

The training environment uses these reward signals:
- **Velocity tracking** (`rew_scale_vel_tracking = 2.0`): Main reward for matching target velocity
- **Uprightness** (`rew_scale_uprightness = 1.0`): Penalty for tilting (+1 upright, -1 flipped)

Episodes terminate early if the robot flips (base height < 0.02m).

**Wheel Sign Corrections (Important for Deployment):**

The USD robot model (`ogre_robot.usd`) has the **right wheels (FR, RR) with opposite joint axis orientation** from the left wheels (FL, RL). This was empirically verified: sending `[+10, +10, +10, +10]` to all wheels causes the robot to spin CCW instead of moving forward.

To create a normalized action space where `[+,+,+,+]` means "all wheels forward", sign corrections are applied to the **right wheels only**.

**Why BOTH training AND deployment need the same correction:**

Training and deployment are **separate execution environments**. The policy learns in a normalized action space where `[+,+,+,+] = forward`. Both environments must apply the same transformation.

**Training Flow:**
```
Policy outputs: [+, +, +, +] (normalized: all positive = forward intent)
       ↓
_apply_action() negates FR and RR: [+, -, +, -]
       ↓
Sent to USD physics: [+, -, +, -]
       ↓
Robot moves FORWARD in training ✓
```

**Deployment Flow (must match training):**
```
Policy outputs: [+, +, +, +] (same normalized output)
       ↓
Isaac Sim action graph negates FR and RR: [+, -, +, -]  ← SAME correction
       ↓
Sent to USD physics: [+, -, +, -]
       ↓
Robot moves FORWARD in deployment ✓
```

**If you DON'T correct the right wheels in Isaac Sim:**
```
Policy outputs: [+, +, +, +]
       ↓
NO correction
       ↓
Sent to USD physics: [+, +, +, +]
       ↓
Robot SPINS instead of going forward ✗
```

**Summary - Where sign corrections are applied:**

| Component | Sign Correction | Notes |
|-----------|-----------------|-------|
| Training `_apply_action()` | Negate FR and RR | Right wheel joint axes are opposite |
| Training `_get_observations()` | Negate FR and RR | For observation consistency |
| Isaac Sim Action Graph | Negate FR and RR | Multiply FR, RR velocities × -1 |
| ROS2 Policy Controller | None | Uses standard mecanum forward kinematics |

**Flip Termination:**
- Episodes terminate early if robot base height drops below 0.02m (half wheel radius)
- This prevents learning from unstable states and implicitly penalizes flipping

## Policy Deployment (ROS2)

The trained policy is deployed using the `ogre_policy_controller` ROS2 package in **ogre-slam**.

See the [ogre-slam README](https://github.com/protomota/ogre-slam#using-the-trained-rl-policy) for deployment instructions.

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
│   ├── export_policy.sh          # Export trained model
│   └── deploy_model.sh           # Deploy to ogre-slam
└── docs/                         # Additional documentation

ogre-slam/ogre_policy_controller/ # ROS2 policy controller (in ogre-slam repo)
├── package.xml
├── setup.py
├── config/
│   └── policy_controller_params.yaml
├── launch/
│   └── policy_controller.launch.py
├── models/                       # Deployed models
│   ├── policy.onnx
│   └── policy.pt
└── ogre_policy_controller/
    └── policy_controller_node.py
```

## Nav2 Costmap Reference

When integrating with Nav2 for autonomous navigation, understanding costmaps is essential for debugging planning failures.

### What Are Costmaps?

Nav2 uses **costmaps** to represent where the robot can and cannot go. Costmaps are 2D grids where each cell has a cost value (0-254) indicating how "expensive" it is to traverse that location.

**Cost Values:**

| Cost | Name | Meaning |
|------|------|---------|
| 0 | FREE_SPACE | Safe to traverse |
| 1-252 | INFLATED | Increasingly expensive (near obstacles) |
| 253 | INSCRIBED | Inside robot's radius from obstacle (collision likely) |
| 254 | LETHAL | Obstacle cell (certain collision) |
| 255 | NO_INFORMATION | Unknown space |

### Two Costmaps

| Costmap | Topic | Purpose | Size |
|---------|-------|---------|------|
| **Global** | `/global_costmap/costmap` | Path planning on full map | Entire map |
| **Local** | `/local_costmap/costmap` | Reactive obstacle avoidance | 3m × 3m rolling window |

The **global costmap** is used by the planner (NavFn/Smac) to find a path from start to goal. The **local costmap** is used by the controller (DWB) to follow that path while avoiding obstacles detected in real-time by sensors.

### How Inflation Works

```
                    inflation_radius (0.35m)
                    ◄────────────────────────►

Wall ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
     ▲              ▲                          ▲
     │              │                          │
  Lethal         Inscribed                   Free
  (cost=254)     (cost=253)                 (cost=0)
                    │
                    └─ robot_radius (0.20m)

Cost drops from 253 to 0 based on cost_scaling_factor.
Higher cost_scaling_factor = faster dropoff = paths closer to walls.
```

**Key Parameters:**
- `robot_radius` (0.20m): The inscribed radius - if robot center enters this zone, collision is certain
- `inflation_radius` (0.35m): How far from obstacles to spread costs - paths prefer to stay outside this zone
- `cost_scaling_factor` (3.0): How quickly costs drop off - higher means costs drop faster

### Costmap Layers

Both costmaps are built from multiple layers:

| Layer | Purpose | Data Source |
|-------|---------|-------------|
| `static_layer` | Walls from saved map | `/map` topic |
| `obstacle_layer` | Real-time obstacles | LIDAR (`/scan`), Camera (`/camera_points`) |
| `voxel_layer` | 3D obstacle detection | PointCloud2 (local costmap only) |
| `inflation_layer` | Expand obstacles by robot radius | Applied last to all obstacles |

### Visualizing Costmaps in RViz

The `navigation.rviz` config includes both costmaps. When viewing:

**Color Coding (costmap color scheme):**
- **Red/Magenta**: Lethal obstacles (cost=254) - walls, detected obstacles
- **Purple/Blue**: High cost inflated zone - avoid if possible
- **Cyan/Light Blue**: Lower cost zone - prefer to stay here
- **Gray**: Free space (cost=0) - optimal for planning

**Topics to display:**
```
/global_costmap/costmap       # Full map with inflation
/local_costmap/costmap        # Rolling window around robot
/global_costmap/costmap_raw   # Without inflation (debug)
/local_costmap/costmap_raw    # Without inflation (debug)
```

### Common Costmap Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Goal appears in red zone | Goal too close to wall | Click further from walls or increase planner tolerance |
| Planner fails "legal potential but no path" | Goal in inscribed zone | Increase `tolerance`, switch to Smac planner |
| Robot gets stuck in corridors | Inflation too wide | Reduce `inflation_radius` (try 0.30-0.35m) |
| Robot clips walls | Inflation too narrow | Increase `inflation_radius` (try 0.40-0.50m) |
| Obstacles not appearing | Sensor not publishing | Check `/scan` and `/camera_points` topics |
| Stale obstacles | Clearing not working | Check `clearing: True` in observation sources |

### Planner Selection

Nav2 supports multiple planners. Choose based on your needs:

| Planner | Plugin | Best For | Known Issues |
|---------|--------|----------|--------------|
| **NavFn** | `nav2_navfn_planner/NavfnPlanner` | Simple environments | "Legal potential but no path" bug in tight spaces |
| **Smac 2D** | `nav2_smac_planner/SmacPlanner2D` | Tight spaces, reliable | Slower than NavFn |
| **Smac Hybrid** | `nav2_smac_planner/SmacPlannerHybrid` | Non-holonomic robots | Overkill for mecanum |
| **Theta\*** | `nav2_theta_star_planner/ThetaStarPlanner` | Any-angle paths | Can cut corners |

**Current Configuration:** Smac 2D (recommended for maze navigation)

```yaml
planner_server:
  GridBased:
    plugin: "nav2_smac_planner/SmacPlanner2D"
    tolerance: 0.50  # Search 50cm around goal for valid endpoint
    max_planning_time: 2.0  # seconds
```

### Debugging Commands

```bash
# Check costmap is publishing
ros2 topic hz /global_costmap/costmap
ros2 topic hz /local_costmap/costmap

# View costmap info
ros2 topic echo /global_costmap/costmap --once | head -20

# Check costmap parameters
ros2 param get /global_costmap/global_costmap inflation_layer.inflation_radius
ros2 param get /local_costmap/local_costmap inflation_layer.inflation_radius
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

### Project Ogre Ecosystem

This repository is part of the **Project Ogre** ecosystem for mecanum drive robot navigation:

| Repository | Purpose | Key Contents |
|------------|---------|--------------|
| **ogre-lab** (this repo) | RL policy training | Isaac Lab environment, trained models, ROS2 policy controller |
| **[ogre-slam](https://github.com/protomota/ogre-slam)** | SLAM & Navigation | ROS2 package, Nav2 config, robot USD models, sensor drivers |

### How They Work Together

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING (ogre-lab)                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ Isaac Lab    │───▶│ Train Policy │───▶│ Export ONNX  │                  │
│  │ Environment  │    │ (RSL-RL/PPO) │    │ & JIT Models │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                                        │                          │
│         │ uses                                   │ produces                 │
│         ▼                                        ▼                          │
│  ogre_robot.usd                          models/policy.onnx                 │
│  (from ogre-slam)                        models/policy.pt                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ deploy
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DEPLOYMENT (ogre-slam)                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    Nav2      │───▶│   Policy     │───▶│ Robot/Isaac  │                  │
│  │ (planning)   │    │  Controller  │    │     Sim      │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                             │                                               │
│                             │ runs                                          │
│                             ▼                                               │
│                      policy.onnx (from ogre-lab)                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dependencies Between Repos

- **ogre-lab → ogre-slam**: Uses `ogre_robot.usd` for training physics simulation
- **ogre-slam → ogre-lab**: Uses trained `policy.onnx` for velocity command optimization

### External Dependencies

- **[Isaac Lab](https://github.com/isaac-sim/IsaacLab)**: GPU-accelerated robot learning framework (NVIDIA)
- **[RSL-RL](https://github.com/leggedrobotics/rsl_rl)**: Reinforcement learning library for robotics

## License

MIT License
