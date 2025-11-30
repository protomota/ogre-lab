# Implementation Notes

Technical details on how the ogre-lab RL training system works.

## Tools and Technologies Overview

This project uses several interconnected tools for robot simulation, training, and autonomous navigation.

### NVIDIA Isaac Sim

**What it is:** A high-fidelity robotics simulator built on NVIDIA Omniverse, using PhysX for physics simulation.

**What it does:**
- Renders realistic 3D environments with accurate physics
- Simulates robot dynamics (wheels, joints, sensors)
- Publishes ROS2 topics (`/odom`, `/scan`, `/clock`) for integration with Nav2
- Uses USD (Universal Scene Description) files to define robot models and environments

**When we use it:** Testing the trained policy in simulation before deploying to the real robot. Isaac Sim runs the robot scene and responds to `/joint_command` messages from the policy controller.

### NVIDIA Isaac Lab

**What it is:** A reinforcement learning framework built on top of Isaac Sim, designed for training robot policies.

**What it does:**
- Runs thousands of robot instances in parallel for fast RL training
- Provides gym-like environments for policy training
- Integrates with RL libraries (RSL-RL, Stable Baselines3)
- Exports trained policies to ONNX/PyTorch formats

**When we use it:** Training the velocity tracking policy. Isaac Lab runs 1024+ robots simultaneously, each learning to convert velocity commands to wheel velocities.

### ROS2 (Robot Operating System 2)

**What it is:** A middleware framework for robot software development, providing publish/subscribe messaging, services, and actions.

**What it does:**
- Manages communication between nodes via topics (e.g., `/cmd_vel`, `/odom`, `/scan`)
- Provides standard message types (`geometry_msgs/Twist`, `sensor_msgs/LaserScan`)
- Handles coordinate frame transforms (TF2)
- Launches and manages multiple nodes

**Key ROS2 concepts:**
- **Node:** A single process that performs a specific function
- **Topic:** A named channel for publishing/subscribing messages
- **Message:** A data structure (e.g., `Twist` for velocity commands)
- **Launch file:** Python script that starts multiple nodes with configuration

### Nav2 (Navigation 2)

**What it is:** The ROS2 navigation stack - a collection of packages for autonomous robot navigation.

**What it does:**
- **Path Planning:** Computes optimal paths from current position to goal (NavFn planner)
- **Local Control:** Generates velocity commands to follow paths while avoiding obstacles (DWB controller)
- **Localization:** Estimates robot position on a map using LIDAR (AMCL)
- **Costmaps:** Maintains 2D obstacle maps for planning and avoidance
- **Behavior Trees:** Coordinates navigation behaviors (replanning, recovery)

**Key Nav2 components in this project:**
| Component | Purpose |
|-----------|---------|
| `map_server` | Loads and serves the saved map |
| `amcl` | Particle filter localization (map→odom transform) |
| `planner_server` | Global path planning (NavFn) |
| `controller_server` | Local trajectory control (DWB) |
| `behavior_server` | Recovery behaviors (spin, backup, wait) |
| `bt_navigator` | Behavior tree coordinator |
| `velocity_smoother` | Smooths velocity commands to reduce jerk |

### RViz2

**What it is:** The ROS2 3D visualization tool.

**What it does:**
- Visualizes robot state, sensor data, and navigation in real-time
- Displays the map, LIDAR scans, costmaps, and planned paths
- Allows setting initial pose (2D Pose Estimate) and navigation goals (Nav2 Goal)
- Shows TF coordinate frames for debugging

**Key RViz displays for navigation:**
- **Map:** The static occupancy grid from `map_server`
- **LaserScan:** Real-time LIDAR data from `/scan`
- **RobotModel:** Robot visualization from URDF/USD
- **Path:** Global and local planned paths
- **Costmap:** Obstacle inflation and planning costs

### slam_toolbox

**What it is:** A ROS2 SLAM (Simultaneous Localization and Mapping) package.

**What it does:**
- Builds maps from LIDAR scans while the robot moves
- Performs loop closure to correct accumulated drift
- Can operate in mapping mode (build new map) or localization mode (use existing map)

**When we use it:** Creating maps of the environment before autonomous navigation. The saved map is then loaded by `map_server` for Nav2.

### ONNX Runtime

**What it is:** A cross-platform inference engine for running trained neural network models.

**What it does:**
- Executes the trained policy model efficiently
- Provides consistent inference across different platforms (GPU, CPU)
- Supports models exported from PyTorch, TensorFlow, etc.

**When we use it:** The policy controller loads the ONNX model to convert velocity commands to wheel velocities in real-time.

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING (Isaac Lab)                           │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Isaac Lab      │    │   RSL-RL PPO    │    │  Policy Export  │         │
│  │  Environment    │───►│   Training      │───►│  (ONNX/JIT)     │         │
│  │  (1024 robots)  │    │                 │    │                 │         │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘         │
└────────────────────────────────────────────────────────┼────────────────────┘
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DEPLOYMENT (ROS2)                               │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   RViz2     │    │    Nav2     │    │   Policy    │    │  Isaac Sim  │  │
│  │ Visualization│◄──│  Navigation │───►│  Controller │───►│  or Real    │  │
│  │             │    │   Stack     │    │  (ONNX)     │    │   Robot     │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        ▲                  │                  │                   │          │
│        │                  ▼                  ▼                   ▼          │
│        │            /cmd_vel_smoothed  /joint_command       /odom, /scan   │
│        └───────────────────────────────────────────────────────────────────│
│                              ROS2 Topics                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## How Isaac Lab Controls the Robot

Isaac Lab uses **joint names** from your USD file to control the robot wheels.

### Joint Name Matching

**In the training environment (`ogre_navigation_env.py`):**

```python
# Wheel joint names (must match USD exactly)
wheel_joint_names = ["fl_joint", "fr_joint", "rl_joint", "rr_joint"]

# Isaac Lab finds those joints in the USD:
self._wheel_joint_ids, _ = self.robot.find_joints(self.cfg.wheel_joint_names)

# Applies velocity targets to them:
self.robot.set_joint_velocity_target(self.actions, joint_ids=self._wheel_joint_ids)
```

**In your USD file (`ogre_robot.usd`):**

The robot must have an `ArticulationRoot` with `RevoluteJoint` prims named:
- `fl_joint` (front-left)
- `fr_joint` (front-right)
- `rl_joint` (rear-left)
- `rr_joint` (rear-right)

Isaac Lab's `Articulation` class finds joints by name, then uses PhysX to apply velocity commands. The physics simulation handles wheel rotation and resulting robot motion.

**If your USD has different joint names** (e.g., `wheel_front_left`), update `wheel_joint_names` in the training config to match.

## Joint Order and Wheel Sign Corrections

### VERIFIED Joint Order

When calling `robot.find_joints(["fl_joint", "fr_joint", "rl_joint", "rr_joint"])`, Isaac Lab returns joints in this order:

| Index | Physical Wheel |
|-------|---------------|
| 0     | FR (front-right) |
| 1     | RR (rear-right) |
| 2     | RL (rear-left) |
| 3     | FL (front-left) |

**This is NOT the order specified in the query!** Isaac Lab returns joints based on their order in the USD file, not the query order.

### Wheel Axis Orientation

Empirically verified by sending `[+6, +6, +6, +6]` to all wheels:
- **Right wheels (FR, RR)** spin backward with positive velocity
- **Left wheels (RL, FL)** spin forward with positive velocity
- Robot spins **clockwise (CW)**

### Sign Corrections

The training environment applies sign corrections so the policy learns in a normalized space where `[+,+,+,+] = forward`:

```python
def _apply_action(self) -> None:
    """Apply wheel velocity targets to the robot.

    VERIFIED JOINT ORDER: [FR, RR, RL, FL] (indices 0, 1, 2, 3)
    Negate RIGHT wheels (FR=0, RR=1) so [+,+,+,+] = forward motion.
    """
    corrected_actions = self.actions.clone()
    corrected_actions[:, 0] *= -1  # FR (index 0)
    corrected_actions[:, 1] *= -1  # RR (index 1)
    self.robot.set_joint_velocity_target(corrected_actions, joint_ids=self._wheel_joint_ids)
```

The same correction is applied to observations:

```python
def _get_observations(self) -> dict:
    joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids].clone()
    joint_vel[:, 0] *= -1  # FR
    joint_vel[:, 1] *= -1  # RR
```

This ensures the policy learns: **positive wheel velocity = forward motion** for all wheels.

### Isaac Sim Deployment

The Isaac Sim action graph must also negate FR and RR wheel velocities:
```
wheel_fr = -(vx + vy + vtheta * L)       # NEGATED (index 0)
wheel_rr = -(vx - vy + vtheta * L)       # NEGATED (index 1)
wheel_rl =  (vx + vy - vtheta * L)       # No negation (index 2)
wheel_fl =  (vx - vy - vtheta * L)       # No negation (index 3)
```

## How Velocity Tracking Training Works

**This is NOT goal-navigation training.** The robots are NOT all driving toward a target position.

Instead, each robot receives a **random target velocity** (vx, vy, vtheta) at the start of each episode:
- vx: forward/backward speed (-1.0 to +1.0 m/s)
- vy: left/right strafe speed (-1.0 to +1.0 m/s)
- vtheta: rotation speed (-2.0 to +2.0 rad/s)

**Expected training behavior:**
- Each robot moves in a DIFFERENT direction based on its random target
- Some go forward, some backward, some strafe, some rotate
- The policy learns: "given target velocity X, output wheel velocities Y"
- Reward = how well current velocity matches target velocity

**Why this is useful:**
When deployed, Nav2 sends velocity commands (Twist messages). The policy converts these to optimal wheel velocities, compensating for the robot's dynamics better than simple inverse kinematics.

## Training Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `action_scale` | 8.0 | Multiplier for policy output → wheel velocity (rad/s) |
| `max_lin_vel` | 0.5 | Maximum linear velocity for training (m/s) |
| `max_ang_vel` | 2.0 | Maximum angular velocity for training (rad/s) |
| `max_wheel_vel` | 8.0 | Safety limit - robot flips above this (rad/s) |
| `rew_scale_vel_tracking` | 2.0 | Main velocity tracking reward |
| `rew_scale_exceed_limit` | -1.0 | Penalty for wheel velocities > max_wheel_vel |
| `rew_scale_uprightness` | 1.0 | Reward for staying upright (+1 upright, -1 flipped) |
| `decimation` | 4 | Physics steps per control step |
| `episode_length_s` | 10.0 | Episode duration (seconds) |

### Understanding Velocity Limits

There are two different velocity concepts that often get confused:

**1. Target velocity limits (`max_lin_vel`, `max_ang_vel`):**
- Range of random velocities sampled during training
- Policy learns to track velocities in this range
- Can be set higher than physically achievable - policy learns to get as close as possible

**2. Wheel velocity limit (`max_wheel_vel`):**
- Hardware safety limit (robot flips above 8 rad/s)
- Enforced via penalty in reward function
- Should NOT be changed unless hardware changes

**Physical constraints:**
- Wheel radius: 0.04m
- Max wheel velocity: 8 rad/s (before robot flips)
- Theoretical max linear velocity: 8 × 0.04 = **0.32 m/s**

With `max_lin_vel=0.5` (higher than physically achievable), the policy learns to:
1. Output maximum wheel velocities for high-speed targets
2. Achieve ~0.32 m/s actual velocity (the physical limit)
3. Still get partial tracking reward for getting as close as possible

### Velocity Configuration Chain (Training → Nav2 → Isaac Sim)

**CRITICAL:** All velocity parameters must match across training and deployment. Mismatches cause navigation failures.

```
Training (Isaac Lab)           Nav2 (ROS2)                  Isaac Sim
─────────────────────         ─────────────────────        ────────────────
max_lin_vel = 0.5 m/s    ═══► DWB max_vel_x = 0.5     ═══► Policy Controller
max_ang_vel = 2.0 rad/s  ═══► DWB max_vel_theta = 2.0 ═══►   action_scale = 8.0
action_scale = 8.0       ═══► velocity_smoother:     ═══►   max_lin_vel = 0.5
                               max_velocity: [0.5, 0.5, 2.0]
```

**Common Mistake:** Confusing `action_scale` (8.0 rad/s wheel velocity) with `max_lin_vel` (0.5 m/s body velocity).
- `action_scale = 8.0` → Wheel velocity in **rad/s**
- `max_lin_vel = 0.5` → Body velocity in **m/s**

These are different units! Nav2 uses body velocity (m/s), not wheel velocity (rad/s).

**Configuration Files That Must Match:**

| Parameter | Training Config | Nav2 nav2_params.yaml | Policy Controller |
|-----------|-----------------|----------------------|-------------------|
| Max linear velocity | `max_lin_vel: 0.5` | `max_vel_x: 0.5`, `max_speed_xy: 0.5` | `max_lin_vel: 0.5` |
| Max angular velocity | `max_ang_vel: 2.0` | `max_vel_theta: 2.0` | `max_ang_vel: 2.0` |
| Action scale | `action_scale: 8.0` | N/A | `action_scale: 8.0` |
| Velocity smoother | N/A | `max_velocity: [0.5, 0.5, 2.0]` | N/A |

**What Happens With Mismatched Velocities:**

If Nav2 is configured for higher velocities than training (e.g., 8.0 m/s instead of 0.5 m/s):
1. DWB plans trajectories assuming 16x faster robot
2. Trajectory predictions extend far beyond actual capability
3. Robot constantly fails "make progress" checks
4. Strange red trajectory predictions in RViz
5. Navigation fails with "FollowPath failed" errors

**Verifying Configuration:**
```bash
# Check Nav2 velocity limits
grep -E "max_vel|max_speed" ~/ros2_ws/src/ogre-slam/config/nav2_params.yaml

# Check training limits
grep -E "max_lin_vel|max_ang_vel|action_scale" ~/ogre-lab/isaaclab_env/ogre_navigation/ogre_navigation_env.py

# Check policy controller
grep -E "max_lin_vel|max_ang_vel|action_scale" ~/ros2_ws/src/ogre-slam/ogre_policy_controller/config/policy_controller_params.yaml
```

### Observation Space (10 dimensions)

| Index | Description | Source |
|-------|-------------|--------|
| 0-2 | Target velocity (vx, vy, vtheta) | Randomly sampled each episode |
| 3-5 | Current velocity (vx, vy, vtheta) | `robot.data.root_lin_vel_b`, `root_ang_vel_b` |
| 6-9 | Wheel velocities (fl, fr, rl, rr) | `robot.data.joint_vel` (sign-corrected) |

### Action Space (4 dimensions)

| Index | Description |
|-------|-------------|
| 0 | Front-left wheel velocity target |
| 1 | Front-right wheel velocity target |
| 2 | Rear-left wheel velocity target |
| 3 | Rear-right wheel velocity target |

Policy outputs are in range [-1, 1], then multiplied by `action_scale` to get rad/s.

### Reward Function

```python
# Velocity tracking (exponential reward, peaks at zero error)
vel_error_norm = torch.sum(vel_error ** 2, dim=-1)
rew_vel_tracking = 2.0 * torch.exp(-vel_error_norm / 0.25)

# Exceed-limit penalty: ONLY penalize wheel velocities ABOVE max_wheel_vel
# This is CRITICAL - see "Critical Bugs Fixed" section below
excess = torch.clamp(torch.abs(actions) - max_wheel_vel, min=0.0)
rew_exceed_limit = -1.0 * torch.sum(excess ** 2, dim=-1)

# Uprightness reward (+1 upright, -1 flipped)
rew_uprightness = 1.0 * up_z

total_reward = rew_vel_tracking + rew_exceed_limit + rew_uprightness
```

## Training Health Indicators

When training is going well, you should see:

| Metric | Good Values | Bad Values |
|--------|-------------|------------|
| Mean reward | 700-850 (positive) | Negative or < 100 |
| Mean episode length | ~299 (full episode) | < 100 (robots crashing) |
| Action noise std | ~0.10 at end (deterministic) | Stuck > 0.5 |
| Surrogate loss | Small (< 0.01) | Large or oscillating |

Training typically takes ~1 hour on RTX 4090 for 1000 iterations.

## Deployment Flow

```
Training (Isaac Lab)          Deployment (ROS2)
─────────────────────         ─────────────────────
policy output [-1, 1]    →    policy output [-1, 1]
      × action_scale           × action_scale
      = wheel vel (rad/s)      = wheel vel (rad/s)
      ↓                              ↓
 PhysX simulation            forward kinematics
      ↓                              ↓
 robot moves                  Twist message
                                    ↓
                              /cmd_vel topic
                                    ↓
                              Isaac Sim / Real Robot
```

## File Locations

| File | Purpose |
|------|---------|
| `~/ogre-lab/isaaclab_env/ogre_navigation/ogre_navigation_env.py` | Training environment |
| `~/ogre-lab/isaaclab_env/ogre_navigation/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters |
| `~/ogre-lab/models/policy.onnx` | Exported ONNX model |
| `~/ogre-lab/models/policy.pt` | Exported JIT model |
| `~/ros2_ws/src/ogre-slam/usds/ogre_robot.usd` | Robot USD for training |
| `~/ros2_ws/src/ogre-slam/usds/ogre.usd` | Robot + scene for Isaac Sim testing |

## Troubleshooting

### "Failed to find articulation"
- Check USD has `ArticulationRoot` applied to robot root prim
- Verify prim path in training config matches USD structure

### Joint names not found
- Run `robot.find_joints(["*"])` to list all available joints
- Update `wheel_joint_names` to match your USD

### Very negative rewards
- Velocity limits too high for robot capabilities
- Check wheel sign corrections match USD joint orientations
- Verify robot isn't flying/exploding (reduce forces, increase damping)

### Policy outputs near zero
- Action scale too low
- Policy not trained long enough
- Observation/action space mismatch between training and deployment

## Export and Deploy Scripts

### Export Script (`scripts/export_policy.sh`)

Exports the trained policy to ONNX and JIT formats by running `play.py`:

```bash
cd ~/ogre-lab

# Export latest model with visualization (16 robots)
./scripts/export_policy.sh

# Export latest model headless (faster)
./scripts/export_policy.sh --headless

# Export specific training run
./scripts/export_policy.sh 2025-11-28_10-14-31

# Export specific run headless
./scripts/export_policy.sh 2025-11-28_10-14-31 --headless
```

**What it does:**
1. Finds the latest training run in `~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/`
2. Locates the most recent model checkpoint (`model_*.pt`)
3. Runs `play.py` to visualize and export the policy
4. Creates `policy.onnx` and `policy.pt` in the run's `exported/` directory

### Deploy Script (`scripts/deploy_model.sh`)

Copies exported models to the ROS2 package:

```bash
cd ~/ogre-lab

# Copy latest model
./scripts/deploy_model.sh

# Copy and rebuild ROS2 package
./scripts/deploy_model.sh --rebuild
```

**What it does:**
1. Finds the most recent training run with exported models
2. Copies `policy.onnx` and `policy.pt` to `~/ros2_ws/src/ogre-slam/ogre_policy_controller/models/`
3. Optionally rebuilds the ROS2 package

### Training Run Structure

```
~/isaac-lab/IsaacLab/logs/rsl_rl/ogre_navigation/
└── 2025-11-28_10-14-31/           # Training run (timestamp)
    ├── model_999.pt               # Checkpoint at iteration 999
    ├── model_1999.pt              # Checkpoint at iteration 1999
    ├── params/
    │   └── env.yaml               # Training parameters
    └── exported/                  # Created by export_policy.sh
        ├── policy.pt              # JIT compiled PyTorch model
        └── policy.onnx            # ONNX format for ROS2 deployment
```

## Required Directory Structure

For all scripts to work correctly, follow this directory layout:

```
~/
├── isaac-lab/
│   └── IsaacLab/                  # Isaac Lab installation
│       ├── isaaclab.sh
│       ├── logs/rsl_rl/ogre_navigation/  # Training outputs
│       └── source/isaaclab_tasks/isaaclab_tasks/direct/ogre_navigation/
│                                  # Training environment (symlinked)
├── ogre-lab/                      # This repository (training only)
│   └── scripts/
│       ├── export_policy.sh       # Export trained model
│       └── deploy_model.sh        # Deploy to ogre-slam
├── ros2_ws/                       # ROS2 workspace
│   └── src/
│       ├── ogre-slam/             # SLAM, navigation, and policy controller
│       │   ├── usds/ogre_robot.usd  # Robot USD for training
│       │   └── ogre_policy_controller/  # ROS2 policy controller
│       └── ogre_policy_controller/  # Symlink to above
└── miniconda3/
    └── envs/
        └── env_isaaclab/          # Isaac Lab conda environment
```

### Installation Paths

| Component | Expected Path | Notes |
|-----------|---------------|-------|
| Isaac Lab | `~/isaac-lab/IsaacLab/` | Must use this path for scripts |
| Isaac Sim | Installed via pip in `env_isaaclab` | `pip install isaacsim-rl` |
| ROS2 Workspace | `~/ros2_ws/` | Standard ROS2 workspace |
| Conda Env | `env_isaaclab` | Created during Isaac Lab setup |
| Robot USD | `~/ros2_ws/src/ogre-slam/usds/ogre_robot.usd` | Referenced by training env |
| Policy Controller | `~/ros2_ws/src/ogre-slam/ogre_policy_controller/` | ROS2 package in ogre-slam |

## Manual Export (Reference)

If you need to manually export without using the scripts:

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
cd ~/isaac-lab/IsaacLab

# List training runs
ls -lt logs/rsl_rl/ogre_navigation/

# Export with visualization
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 16 \
    --checkpoint logs/rsl_rl/ogre_navigation/<RUN>/model_1999.pt

# Export headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/ogre_navigation/<RUN>/model_1999.pt \
    --headless
```

## Critical Bugs Fixed (2025-11-30)

These bugs caused policy training to fail or produce unusable policies. **DO NOT reintroduce these issues.**

### Bug 1: Action Clipping After Scaling

**Symptom:** Policy always outputs saturated values (all wheels at max velocity), no proportional control.

**Root Cause:** Clipping was applied AFTER scaling, destroying the full output range.

```python
# WRONG - clips scaled values back to [-1, 1], losing all range
def _apply_action(self) -> None:
    # self.actions is already scaled to [-8, 8] in _pre_physics_step
    clipped_actions = torch.clamp(self.actions, -1.0, 1.0)  # BUG: clips to [-1, 1]!
```

With `action_scale=8.0`, any policy output > 0.125 becomes 1.0 after clipping. The policy has no incentive to output values between 0.125 and 1.0 - they all produce the same wheel velocity!

**Fix:** Remove all clipping in `_apply_action()`. Apply scaled actions directly:

```python
# CORRECT - no clipping, policy learns appropriate outputs naturally
def _apply_action(self) -> None:
    # Apply sign corrections for right wheels
    corrected_actions = self.actions.clone()
    corrected_actions[:, 0] *= -1  # FR
    corrected_actions[:, 1] *= -1  # RR
    self.robot.set_joint_velocity_target(corrected_actions, joint_ids=self._wheel_joint_ids)
```

### Bug 2: Energy Penalty Kills Performance

**Symptom:** Policy outputs very low values (~0.1-0.2), robot barely moves, wheel velocities ~1.6 rad/s when 7.5 rad/s needed.

**Root Cause:** Penalizing ALL wheel velocities causes the policy to minimize output to reduce penalty, sacrificing velocity tracking.

```python
# WRONG - penalizes all velocities quadratically
rew_energy = -0.01 * torch.sum(actions ** 2, dim=-1)
# At 8 rad/s per wheel: penalty = -0.01 × 4 × 64 = -2.56 per step
# This OVERWHELMS the tracking reward of ~2.0!
```

**Fix:** Only penalize velocities ABOVE the safe threshold:

```python
# CORRECT - only penalize excessive velocities
excess = torch.clamp(torch.abs(actions) - max_wheel_vel, min=0.0)
rew_exceed_limit = -1.0 * torch.sum(excess ** 2, dim=-1)
# 0-8 rad/s: NO penalty
# 9 rad/s: penalty = -1.0 × 4 = -4.0
# 10 rad/s: penalty = -1.0 × 16 = -16.0
```

This allows the policy to use the full velocity range (0-8 rad/s) for accurate velocity tracking.

### Bug 3: Wheel Sign Corrections

**Symptom:** Robot spins in circles when commanded to go forward.

**Root Cause:** Right wheels (FR, RR) have opposite joint axis orientation in the USD file.

**Fix:** Negate right wheel velocities in both `_apply_action()` AND `_get_observations()`:

```python
# In _apply_action():
corrected_actions = self.actions.clone()
corrected_actions[:, 0] *= -1  # FR (index 0 in Isaac Lab order)
corrected_actions[:, 1] *= -1  # RR (index 1 in Isaac Lab order)

# In _get_observations():
joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids].clone()
joint_vel[:, 0] *= -1  # FR
joint_vel[:, 1] *= -1  # RR
```

**IMPORTANT:** Isaac Lab joint order is `[FR, RR, RL, FL]` (indices 0,1,2,3) - NOT the query order!

### Summary of Correct Configuration

```python
# ogre_navigation_env.py - OgreNavigationEnvCfg

# Action scaling
action_scale = 8.0  # Scales policy output to wheel velocity (rad/s)

# Velocity ranges (target velocities for training)
max_lin_vel = 0.5   # Target velocity range (m/s) - higher than physical limit
max_ang_vel = 2.0   # Target angular velocity range (rad/s)
max_wheel_vel = 8.0 # Safety limit - robot flips above this (rad/s)

# Reward configuration
rew_scale_vel_tracking = 2.0    # Main tracking reward
rew_scale_exceed_limit = -1.0   # Penalty ONLY for velocities > max_wheel_vel
rew_scale_uprightness = 1.0     # Stay upright bonus
```

### ROS2 Controller Must Match

The ROS2 policy controller (`ogre_policy_controller`) must use the same velocity limits:

```yaml
# policy_controller_params.yaml
action_scale: 8.0       # MUST match training
max_lin_vel: 0.5        # MUST match training
max_ang_vel: 2.0        # MUST match training
output_mode: "joint_state"
output_topic: "/joint_command"
```

## Nav2 Integration with Isaac Sim

The trained policy runs in a ROS2 node that converts Nav2 velocity commands into wheel velocities for Isaac Sim.

### Topic Routing Architecture

```
Nav2 Stack                    Policy Controller              Isaac Sim
───────────────────          ──────────────────────         ─────────────
controller_server            ogre_policy_controller         Articulation
     │                              │                        Controller
     │ /cmd_vel                     │                            │
     ▼                              │                            │
velocity_smoother                   │                            │
     │                              │                            │
     │ /cmd_vel_smoothed            │                            │
     └─────────────────────────────►│                            │
                                    │ /joint_command             │
                                    └───────────────────────────►│
                                                                 │
                                                                 ▼
                                                            Robot moves
```

**Topic Details:**

| Topic | Type | Publisher | Subscriber | Description |
|-------|------|-----------|------------|-------------|
| `/cmd_vel` | Twist | controller_server | velocity_smoother | Raw velocity commands from DWB planner |
| `/cmd_vel_smoothed` | Twist | velocity_smoother | policy_controller | Smoothed velocities (reduces jerk) |
| `/joint_command` | JointState | policy_controller | Isaac Sim | Wheel velocity targets (rad/s) |

### Launching Nav2 with the Policy Controller

**Terminal 1 - Isaac Sim:**
```bash
# Start Isaac Sim with the Ogre robot scene
# Press Play to begin simulation
```

**Terminal 2 - Navigation Stack:**
```bash
export ROS_DOMAIN_ID=42
source ~/ros2_ws/install/setup.bash

# Launch Nav2 with a saved map
ros2 launch ogre_slam navigation.launch.py \
    map:=~/ros2_ws/src/ogre-slam/maps/isaac_sim_map.yaml \
    use_sim_time:=true \
    use_rviz:=true
```

**Terminal 3 - Policy Controller:**
```bash
export ROS_DOMAIN_ID=42
source ~/ros2_ws/install/setup.bash

# Launch the policy controller
ros2 launch ogre_policy_controller policy_controller.launch.py
```

### Policy Controller Configuration

The policy controller subscribes to `/cmd_vel_smoothed` (NOT `/cmd_vel` directly) because:
1. Nav2's `velocity_smoother` applies acceleration limits and jerk reduction
2. This prevents sudden velocity spikes that could destabilize the robot
3. The smoothed output is better suited for the learned policy

**Configuration file (`policy_controller_params.yaml`):**

```yaml
ogre_policy_controller:
  ros__parameters:
    # Model configuration
    model_path: ""  # Auto-finds in models/ directory
    model_type: "onnx"

    # Topic routing - MUST match Nav2 velocity_smoother output
    input_topic: "/cmd_vel_smoothed"   # From Nav2 velocity_smoother
    output_topic: "/joint_command"      # To Isaac Sim
    output_mode: "joint_state"

    # MUST match training config
    action_scale: 8.0
    max_lin_vel: 0.3
    max_ang_vel: 1.0

    # Robot geometry (for inverse kinematics fallback)
    wheel_radius: 0.040
    wheelbase: 0.095
    track_width: 0.205

    # Wheel joint order (matches Isaac Lab: FR, RR, RL, FL)
    wheel_joint_names:
      - "fr_joint"
      - "rr_joint"
      - "rl_joint"
      - "fl_joint"
```

### Common Nav2 Integration Issues

**Issue: No velocity commands reaching policy controller**

Check topic connectivity:
```bash
# List all cmd_vel topics
ros2 topic list | grep cmd_vel

# Check publishers/subscribers
ros2 topic info /cmd_vel_smoothed
# Should show: Publisher count: 1 (velocity_smoother)
#              Subscription count: 1 (policy_controller)

# Echo to verify data flow
ros2 topic echo /cmd_vel_smoothed --once
```

**Issue: TF_OLD_DATA errors**

Isaac Sim uses simulation time (`/clock` topic). All Nav2 nodes must use `use_sim_time:=true`:
```bash
ros2 launch ogre_slam navigation.launch.py use_sim_time:=true ...
```

**Issue: Policy controller not outputting**

Check if the policy loaded:
```bash
ros2 topic echo /joint_command --once
# Should show JointState with 4 wheel velocities

# If no output, check node logs
ros2 node info /ogre_policy_controller
```

### Isaac Sim Action Graph for /joint_command

Isaac Sim must subscribe to `/joint_command` and apply wheel velocities:

1. **ROS2 Subscribe JointState** node:
   - Topic: `/joint_command`
   - Queue Size: 1

2. **Script Node** (extract velocities):
   ```python
   # Extract wheel velocities from JointState.velocity array
   # Order: [fr, rr, rl, fl] matching wheel_joint_names
   ```

3. **Articulation Controller**:
   - Robot Path: `/World/Ogre/base_link`
   - Joint Names: `["fr_joint", "rr_joint", "rl_joint", "fl_joint"]`
   - Apply velocities from JointState

**Alternative: Direct /cmd_vel Action Graph**

If not using the policy controller, Isaac Sim can directly subscribe to `/cmd_vel` and compute wheel velocities using mecanum kinematics (see CLAUDE.md for equations).

### Nav2 Costmap Configuration (Obstacle Avoidance)

The costmap configuration controls how close the robot gets to obstacles. If the robot clips corners or collides with walls, adjust these parameters in `nav2_params.yaml`.

**Key Parameters:**

| Parameter | Location | Purpose |
|-----------|----------|---------|
| `robot_radius` | local/global_costmap | Robot's circumscribed radius (m) |
| `inflation_radius` | inflation_layer | How far to inflate obstacles (m) |
| `cost_scaling_factor` | inflation_layer | How quickly costs drop off from obstacles |
| `BaseObstacle.scale` | FollowPath (DWB) | Weight for obstacle avoidance in trajectory scoring |

**Current Configuration (safe defaults):**

```yaml
# Costmap inflation (both local and global)
inflation_layer:
  # inflation_radius = robot_radius + desired_clearance
  # 0.20m robot + 0.20m clearance = 0.40m
  inflation_radius: 0.40

  # Lower = costs stay high further from obstacles = more cautious
  # Range: 1.0 (very cautious) to 10.0 (aggressive)
  cost_scaling_factor: 2.5

# DWB controller obstacle avoidance
FollowPath:
  # Trajectory lookahead (seconds) - balance between seeing curves and getting stuck
  sim_time: 1.2

  # Higher = stronger obstacle avoidance
  BaseObstacle.scale: 0.02

  # Higher = follow planned path more strictly (less corner cutting)
  PathAlign.scale: 48.0
  PathDist.scale: 48.0
  GoalDist.scale: 24.0
```

**Tuning Guide:**

| Problem | Solution |
|---------|----------|
| Robot clips corners | Increase `inflation_radius` (try 0.50-0.60m) |
| Robot too far from walls | Decrease `inflation_radius` (try 0.35-0.40m) |
| Robot cuts across obstacles | Increase `BaseObstacle.scale` (try 0.03-0.05) |
| Robot won't go through narrow gaps | Decrease `inflation_radius` and `cost_scaling_factor` |
| Path planning fails in tight spaces | Decrease `inflation_radius` below robot_radius + gap_width/2 |
| Robot doesn't follow green path | Increase `sim_time` (1.5-2.0s) and `PathDist.scale` |
| Robot too aggressive near obstacles | Decrease `sim_time` (0.8-1.2s) |

### Wall Thickness Requirements

**CRITICAL:** Maze walls must be thick enough for reliable costmap detection.

| Component | Value |
|-----------|-------|
| Costmap resolution | 5cm (0.05m) |
| **Minimum wall thickness** | **10cm (0.10m)** - 2× resolution |
| **Recommended wall thickness** | **20cm (0.20m)** - 4× resolution |

**Why this matters:**
- Walls thinner than 2× costmap resolution may be unreliably detected
- A 2cm wall might fall between costmap cells or only occupy a partial cell
- At corners, thin walls may not inflate properly, causing the robot to clip them

**Isaac Sim Maze Generator (generate_maze.py):**
```python
wall_thickness=0.20,  # 20cm thick (4x costmap resolution for reliable detection)
```

If you're still having corner collision issues after ensuring proper wall thickness, check:
1. LIDAR `obstacle_min_range` isn't filtering out nearby walls
2. The map was saved with proper resolution (5cm)
3. Costmap update frequency is high enough (5 Hz for local)

**How Inflation Works:**

```
                    inflation_radius (0.40m)
                    ◄────────────────────────►

Wall ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░
     ▲              ▲                          ▲
     │              │                          │
  Lethal         Inscribed                   Free
  (cost=254)     (cost=253)                 (cost=0)
                    │
                    └─ robot_radius (0.20m)

Cost drops from 253 to 0 based on cost_scaling_factor.
Lower cost_scaling_factor = slower dropoff = more cautious paths.
```

**Verifying Costmap in RViz:**

1. Add a "Map" display for `/local_costmap/costmap`
2. You should see:
   - **Purple/Red** around obstacles (high cost, avoid)
   - **Blue/Cyan** gradient showing inflation zone
   - **Gray** for free space
3. If inflation looks too small, increase `inflation_radius`

## Version History

| Date | Changes |
|------|---------|
| 2025-11-30 | Added velocity configuration chain documentation explaining relationship between training values and Nav2 config |
| 2025-11-30 | Added wall thickness requirements (20cm min for reliable costmap detection), updated costmap values |
| 2025-11-30 | Increased velocity limits (max_lin_vel: 0.3→0.5, max_ang_vel: 1.0→2.0) for faster navigation |
| 2025-11-30 | Fixed action clipping bug, replaced energy penalty with exceed-limit penalty, added Nav2 integration docs, added costmap tuning guide |
| 2025-11-29 | Added wheel sign corrections for FR/RR joints |
| 2025-11-28 | Added export and deploy scripts |
| 2025-11-27 | Initial implementation |
