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
| `max_lin_vel` | 0.3 | Maximum linear velocity for training (m/s) |
| `max_ang_vel` | 1.0 | Maximum angular velocity for training (rad/s) |
| `max_wheel_vel` | 8.0 | Safety limit - robot flips above this (rad/s) |
| `rew_scale_vel_tracking` | 2.0 | Main velocity tracking reward |
| `rew_scale_exceed_limit` | -1.0 | Penalty for wheel velocities > max_wheel_vel |
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

# Velocity ranges
max_lin_vel = 0.3   # Target velocity range (m/s)
max_ang_vel = 1.0   # Target angular velocity range (rad/s)
max_wheel_vel = 8.0 # Safety limit - robot flips above this (rad/s)

# Reward configuration
rew_scale_vel_tracking = 2.0    # Main tracking reward
rew_scale_exceed_limit = -1.0   # Penalty ONLY for velocities > max_wheel_vel
rew_scale_uprightness = 1.0     # Stay upright bonus
```

### ROS2 Controller Must Match

The ROS2 policy controller (`ogre_policy_controller`) must use the same `action_scale`:

```yaml
# policy_controller_params.yaml
action_scale: 8.0       # MUST match training
max_lin_vel: 0.3        # MUST match training
max_ang_vel: 1.0        # MUST match training
output_mode: "joint_state"
output_topic: "/joint_command"
```

## Version History

| Date | Changes |
|------|---------|
| 2025-11-30 | Fixed action clipping bug, replaced energy penalty with exceed-limit penalty |
| 2025-11-29 | Added wheel sign corrections for FR/RR joints |
| 2025-11-28 | Added export and deploy scripts |
| 2025-11-27 | Initial implementation |
