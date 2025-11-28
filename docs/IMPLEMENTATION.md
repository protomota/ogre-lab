# Implementation Notes

Technical details on how the ogre-lab RL training system works.

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

## Wheel Sign Corrections

The USD robot has ALL wheel joint axes inverted (negative velocity = forward motion). The training environment applies sign corrections so the policy learns in a normalized space where `[+,+,+,+] = forward`:

```python
def _apply_action(self) -> None:
    """Apply wheel velocity targets to the robot.

    The USD has ALL wheel joint axes inverted - negative velocity = forward.
    We negate ALL wheel velocities so the policy can output positive values
    for forward motion (normalized action space: [+,+,+,+] = forward).
    """
    corrected_actions = self.actions.clone()
    corrected_actions *= -1  # Negate ALL wheels
    self.robot.set_joint_velocity_target(corrected_actions, joint_ids=self._wheel_joint_ids)
```

The same correction is applied to observations so the policy sees consistent data:

```python
def _get_observations(self) -> dict:
    joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids].clone()
    joint_vel *= -1  # Negate ALL wheel velocities to match action space
```

This ensures the policy learns: **positive wheel velocity = forward motion** for all wheels.

**Important:** The Isaac Sim action graph must also negate ALL wheel velocities during deployment:
```
wheel_fl = -(vx - vy - vtheta * L)
wheel_fr = -(vx + vy + vtheta * L)
wheel_rl = -(vx + vy - vtheta * L)
wheel_rr = -(vx - vy + vtheta * L)
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
| `action_scale` | 20.0 | Multiplier for policy output → wheel velocity (rad/s) |
| `max_lin_vel` | 1.0 | Maximum linear velocity for training (m/s) |
| `max_ang_vel` | 2.0 | Maximum angular velocity for training (rad/s) |
| `rew_scale_uprightness` | 1.0 | Reward for staying upright (+1 upright, -1 flipped) |
| `decimation` | 4 | Physics steps per control step |
| `episode_length_s` | 10.0 | Episode duration (seconds) |

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
rew_vel_tracking = 1.0 * torch.exp(-vel_error_norm)

# Extra reward for XY velocity accuracy
xy_error = torch.sum(vel_error[:, :2] ** 2, dim=-1)
rew_vel_xy = 0.5 * torch.exp(-xy_error)

# Angular velocity tracking
ang_error = vel_error[:, 2] ** 2
rew_ang_vel = 0.25 * torch.exp(-ang_error)

# Energy penalty (small)
rew_energy = -0.0001 * torch.sum(actions ** 2, dim=-1)

# Smoothness penalty (small)
action_diff = actions - prev_actions
rew_smoothness = -0.001 * torch.sum(action_diff ** 2, dim=-1)
```

## Training Health Indicators

When training is going well, you should see:

| Metric | Good Values | Bad Values |
|--------|-------------|------------|
| Mean reward | 30-60+ (positive) | Negative (especially < -1000) |
| Mean episode length | ~299 (full episode) | < 100 (robots crashing) |
| Action noise std | Decreasing over time | Stuck high |
| Surrogate loss | Small (< 0.01) | Large or oscillating |

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
2. Copies `policy.onnx` and `policy.pt` to `~/ogre-lab/ros2_controller/models/`
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
├── ogre-lab/                      # This repository
│   ├── scripts/
│   │   ├── export_policy.sh       # Export trained model
│   │   └── deploy_model.sh        # Deploy to ROS2
│   └── ros2_controller/
│       └── models/                # Deployed models
│           ├── policy.onnx
│           └── policy.pt
├── ros2_ws/                       # ROS2 workspace
│   └── src/
│       ├── ogre-slam/             # SLAM and navigation
│       │   └── usds/ogre_robot.usd  # Robot USD for training
│       └── ogre_policy_controller/  # Symlink to ros2_controller
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
