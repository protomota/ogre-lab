# Wheel Direction Testing

This document explains how to empirically test USD wheel joint axis orientations to determine the correct sign corrections for RL training.

## Why This Matters

The RL policy outputs wheel velocities in a normalized space where `[+,+,+,+]` should mean "all wheels forward". However, USD wheel joints may have different axis orientations that require sign corrections.

**Without correct sign corrections:** The trained policy will output patterns like `[+,+,+,-]` for forward motion, causing the robot to spin instead of going straight.

## Testing Procedure

### Step 1: Launch the Wheel Direction Test

```bash
# Activate Isaac Lab environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Run the test script (launches Isaac Sim with GUI)
cd ~/isaac-lab/IsaacLab
./isaaclab.sh -p ~/ogre-lab/scripts/test_wheel_direction.py
```

### Step 2: Observe Robot Behavior

The test script runs two tests:

**Test 1: `[+6, +6, +6, +6]` rad/s**
- Sends the same positive velocity to all 4 wheels
- Watch which way the robot spins:
  - **CW (clockwise)** → LEFT wheels (FL, RL) have opposite axis
  - **CCW (counter-clockwise)** → RIGHT wheels (FR, RR) have opposite axis
  - **Straight forward/backward** → All wheels have same axis orientation

**Test 2: Corrected velocities**
- Based on Test 1 results, sends corrected velocities
- Robot should move FORWARD in a straight line

### Step 3: Interpret Results

The terminal will print X displacement after each test:
```
After [+6,+6,+6,+6]: Moved X=-0.05m, Y=0.32m  (spin - Y movement)
After [-6,+6,-6,+6]: Moved X=0.45m, Y=0.02m   (forward - X movement)
```

Positive X = forward, significant Y = rotation/spin.

## Ogre Robot Results

**Empirically tested result for `ogre_robot.usd`:**

| Command | Robot Behavior | Interpretation |
|---------|---------------|----------------|
| `[+6, +6, +6, +6]` | CW spin | RIGHT wheels have opposite axis |
| `[+6, -6, +6, -6]` | Forward motion | Negate FR and RR for forward |

**Observation with `[+6, +6, +6, +6]`:**
- Left wheels spin **forward** (positive = forward)
- Right wheels spin **backward** (positive = backward)
- Robot spins **clockwise**

**Correction needed:**
- Keep **LEFT wheels** (FL=index 0, RL=index 2) as-is
- Negate **RIGHT wheels** (FR=index 1, RR=index 3)

## Applying Corrections

### In Training (`ogre_navigation_env.py`)

```python
def _apply_action(self) -> None:
    corrected_actions = self.actions.clone()
    # Negate RIGHT wheels (FR=1, RR=3)
    corrected_actions[:, 1] *= -1  # FR
    corrected_actions[:, 3] *= -1  # RR
    self.robot.set_joint_velocity_target(corrected_actions, ...)

def _get_observations(self) -> dict:
    joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids].clone()
    # Same correction for consistency
    joint_vel[:, 1] *= -1  # FR
    joint_vel[:, 3] *= -1  # RR
```

### In Isaac Sim Action Graph (Deployment)

```
wheel_fl =  (vx - vy - vtheta * L)   # not negated
wheel_fr = -(vx + vy + vtheta * L)   # NEGATED
wheel_rl =  (vx + vy - vtheta * L)   # not negated
wheel_rr = -(vx - vy + vtheta * L)   # NEGATED
```

## Troubleshooting

### Robot flips over during test
- Reduce velocity from 10 rad/s to 6 rad/s
- The test script uses 6 rad/s by default

### Robot doesn't move
- Check that Isaac Sim physics is playing (press Play button)
- Verify robot has articulation root applied
- Check joint names match: `fl_joint`, `fr_joint`, `rl_joint`, `rr_joint`

### Unexpected rotation direction
- Double-check which way is "forward" (positive X axis)
- Camera view may be rotated - use the grid as reference

## Test Script Location

```
~/ogre-lab/scripts/test_wheel_direction.py
```

Run with:
```bash
cd ~/isaac-lab/IsaacLab
./isaaclab.sh -p ~/ogre-lab/scripts/test_wheel_direction.py
```
