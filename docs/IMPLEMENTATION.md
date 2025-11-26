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

The training environment applies wheel sign corrections to match joint axis orientations:

```python
def _apply_action(self) -> None:
    """Apply wheel velocity targets to the robot."""
    corrected_actions = self.actions.clone()
    # Negate front wheels to match joint axis orientation
    corrected_actions[:, 0] = -corrected_actions[:, 0]  # FL
    corrected_actions[:, 1] = -corrected_actions[:, 1]  # FR

    self.robot.set_joint_velocity_target(corrected_actions, joint_ids=self._wheel_joint_ids)
```

The same correction is applied to observations so the policy sees consistent data:

```python
def _get_observations(self) -> dict:
    joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]

    # Correct wheel velocity signs to match action convention
    corrected_joint_vel = joint_vel.clone()
    corrected_joint_vel[:, 0] = -corrected_joint_vel[:, 0]  # FL
    corrected_joint_vel[:, 1] = -corrected_joint_vel[:, 1]  # FR
```

This ensures the policy learns: **positive wheel velocity = forward motion** for all wheels.

## Training Configuration

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `action_scale` | 50.0 | Multiplier for policy output → wheel velocity (rad/s) |
| `max_lin_vel` | 8.0 | Maximum linear velocity for training (m/s) |
| `max_ang_vel` | 6.0 | Maximum angular velocity for training (rad/s) |
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
