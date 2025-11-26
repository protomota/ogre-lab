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

Running `play.py` automatically exports the model to deployment formats.

### Step 1: Find Your Training Run

Training runs are saved with timestamps like `2025-11-26_13-34-55`:

```bash
conda activate env_isaaclab
cd ~/isaac-lab/IsaacLab

# List training runs (newest first)
ls -lt logs/rsl_rl/ogre_navigation/

# Example output:
# drwxrwxr-x 4 brad brad 4096 Nov 26 15:09 2025-11-26_13-34-55  <-- latest
# drwxrwxr-x 5 brad brad 4096 Nov 26 12:16 2025-11-26_10-38-02
# drwxrwxr-x 5 brad brad 4096 Nov 26 10:32 2025-11-26_10-22-58

# Check what models are in a run
ls logs/rsl_rl/ogre_navigation/2025-11-26_13-34-55/
# Shows: model_0.pt, model_100.pt, ..., model_999.pt
```

### Step 2: Export the Model

Use the **full path** with your run folder name:

```bash
# Example with run folder "2025-11-26_13-34-55"
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Isaac-Ogre-Navigation-Direct-v0 \
    --num_envs 1 \
    --checkpoint logs/rsl_rl/ogre_navigation/2025-11-26_13-34-55/model_999.pt \
    --headless
```

This creates an `exported/` folder inside the run directory:
```
logs/rsl_rl/ogre_navigation/2025-11-26_13-34-55/
├── model_999.pt          # Training checkpoint
└── exported/
    ├── policy.pt         # JIT compiled PyTorch model
    └── policy.onnx       # ONNX format for deployment
```

### Step 3: Copy Models to Deployment Location

```bash
# Still in ~/isaac-lab/IsaacLab with your run folder name
cp logs/rsl_rl/ogre_navigation/2025-11-26_13-34-55/exported/policy.onnx ~/ogre-lab/models/
cp logs/rsl_rl/ogre_navigation/2025-11-26_13-34-55/exported/policy.pt ~/ogre-lab/models/
```

### Step 4: Rebuild ROS2 Package

```bash
# Exit Isaac Lab conda env (uses different Python)
conda deactivate

cd ~/ros2_ws
colcon build --packages-select ogre_policy_controller --symlink-install
source install/setup.bash
```

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
| Action scale | 50.0 rad/s |

### Training Notes

The training environment applies wheel sign corrections to match the wheel joint axis orientations in the USD model. The front wheels (FL, FR) are negated in both actions and observations to ensure consistent behavior between training and deployment. This allows the policy to learn that positive wheel velocities = forward motion for all wheels.

## Policy Deployment (ROS2)

The trained policy can be deployed as a ROS2 node that runs the neural network in real-time.

### Install the ROS2 Package

```bash
# Symlink or copy to your ROS2 workspace
ln -sf ~/ogre-lab/ros2_controller ~/ros2_ws/src/ogre_policy_controller

# IMPORTANT: Deactivate Isaac Lab conda env if active (uses different Python)
conda deactivate

# Install Python dependencies (in your ROS2 Python environment)
pip install onnxruntime numpy

# Build the package
cd ~/ros2_ws
colcon build --packages-select ogre_policy_controller
source install/setup.bash
```

**Note:** The ROS2 package uses system Python, not the Isaac Lab conda environment. Make sure `onnxruntime` is installed in your ROS2 Python environment.

### Run the Policy Controller

```bash
# Terminal 1: Launch the policy controller
ros2 launch ogre_policy_controller policy_controller.launch.py

# With custom model path
ros2 launch ogre_policy_controller policy_controller.launch.py \
    model_path:=/path/to/your/policy.onnx

# Use JIT model instead of ONNX
ros2 launch ogre_policy_controller policy_controller.launch.py \
    model_type:=jit \
    model_path:=/path/to/your/policy.pt
```

### Test the Policy Controller

```bash
# Terminal 2: Send test velocity commands
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
    "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10

# Terminal 3: Watch wheel velocity output
ros2 topic echo /wheel_velocities
```

You should see the policy outputting wheel velocities in response to the velocity command.

### Output Modes

The policy controller supports two output modes:

**Mode 1: Twist-to-Twist (default)** - For Nav2 integration
- Subscribes: `/policy_cmd_vel_in` (Twist from Nav2)
- Publishes: `/cmd_vel` (Twist to robot/simulator)
- Use case: Policy sits between Nav2 and robot, improving velocity tracking

**Mode 2: Twist-to-WheelVelocities** - For direct motor control
- Subscribes: `/cmd_vel` (Twist)
- Publishes: `/wheel_velocities` (Float32MultiArray)
- Use case: Robot with direct wheel velocity control interface

### Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/policy_cmd_vel_in` (sub) | geometry_msgs/Twist | Velocity commands (Mode 1 default) |
| `/cmd_vel` (pub/sub) | geometry_msgs/Twist | Output (Mode 1) or Input (Mode 2) |
| `/odom` (sub) | nav_msgs/Odometry | Current robot velocity feedback |
| `/joint_states` (sub) | sensor_msgs/JointState | Current wheel velocities |
| `/wheel_velocities` (pub) | std_msgs/Float32MultiArray | Wheel velocity targets (Mode 2 only) |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | auto-detect | Path to ONNX or JIT model |
| `model_type` | "onnx" | Model format: "onnx" or "jit" |
| `output_mode` | "twist" | Output mode: "twist" or "wheel_velocities" |
| `input_topic` | "/policy_cmd_vel_in" | Input velocity command topic |
| `output_topic` | "/cmd_vel" | Output topic (Twist or Float32MultiArray) |
| `action_scale` | 10.0 | Wheel velocity scaling factor |
| `max_lin_vel` | 0.5 | Max linear velocity (m/s) |
| `max_ang_vel` | 1.0 | Max angular velocity (rad/s) |
| `control_frequency` | 30.0 | Control loop rate (Hz) |
| `wheel_radius` | 0.040 | Wheel radius in meters |
| `wheelbase` | 0.095 | Front-rear axle distance in meters |
| `track_width` | 0.205 | Left-right wheel distance in meters |

### Integration with Nav2

The trained policy improves velocity tracking by learning the robot's dynamics. It sits between Nav2 and the robot:

```
Nav2 Controller Server → /policy_cmd_vel_in → Policy Node → /cmd_vel → Robot
```

**Setup for Nav2 Integration:**

1. **Remap Nav2's output topic** in your Nav2 params:
   ```yaml
   controller_server:
     ros__parameters:
       # Remap cmd_vel output to policy input
       cmd_vel_topic: "/policy_cmd_vel_in"
   ```

2. **Launch the policy controller:**
   ```bash
   ros2 launch ogre_policy_controller policy_controller.launch.py
   ```

3. **Verify the data flow:**
   ```bash
   # Check Nav2 publishes to policy input
   ros2 topic echo /policy_cmd_vel_in

   # Check policy outputs to cmd_vel
   ros2 topic echo /cmd_vel
   ```

**Alternative: Use topic remapping** instead of modifying Nav2 params:
```bash
ros2 launch ogre_policy_controller policy_controller.launch.py \
    input_topic:=/cmd_vel \
    output_topic:=/robot_cmd_vel
```

Then remap your robot's velocity input to `/robot_cmd_vel`.

### Testing with Isaac Sim

To test the policy controller with Isaac Sim before deploying to hardware:

```bash
# Terminal 1: Start Isaac Sim with ogre.usd and press Play

# Terminal 2: Launch the policy controller (uses system Python, not conda)
conda deactivate  # Important: exit Isaac Lab conda env
cd ~/ros2_ws
source install/setup.bash
ros2 launch ogre_policy_controller policy_controller.launch.py

# Terminal 3: Send test commands
ros2 topic pub /policy_cmd_vel_in geometry_msgs/msg/Twist \
    "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}" -r 10

# Terminal 4: Monitor policy output
ros2 topic echo /cmd_vel
```

The robot in Isaac Sim should move forward. The policy converts the velocity command through its neural network, optimizing for the learned robot dynamics.

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
├── ros2_controller/              # ROS2 deployment package
│   ├── package.xml
│   ├── setup.py
│   ├── config/
│   │   └── policy_controller_params.yaml
│   ├── launch/
│   │   └── policy_controller.launch.py
│   ├── models/                   # Copy of exported models
│   └── ogre_policy_controller/
│       └── policy_controller_node.py
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
