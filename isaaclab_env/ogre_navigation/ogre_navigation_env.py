# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Ogre Mecanum Robot Navigation Environment.

This environment trains a neural network policy to efficiently track velocity commands
for a mecanum drive robot. The policy learns the robot's dynamics and outputs optimal
wheel velocities to achieve commanded body velocities (vx, vy, vtheta).

Observation Space (10 dimensions):
    - Target velocity command: vx, vy, vtheta (3)
    - Current base velocity: vx, vy, vtheta (3)
    - Current wheel velocities: fl, fr, rl, rr (4)

Action Space (4 dimensions):
    - Wheel velocity targets: fl, fr, rl, rr

Rewards:
    - Velocity tracking: minimize error between target and actual velocity
    - Energy efficiency: penalize excessive wheel velocities
    - Smoothness: penalize jerky motion
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


# Robot configuration - matches your ogre.usd robot
OGRE_MECANUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/brad/ros2_ws/src/ogre-slam/usds/ogre_robot.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.04),  # Wheel axle height = wheel radius
        joint_pos={
            "fl_joint": 0.0,
            "fr_joint": 0.0,
            "rl_joint": 0.0,
            "rr_joint": 0.0,
        },
        joint_vel={
            "fl_joint": 0.0,
            "fr_joint": 0.0,
            "rl_joint": 0.0,
            "rr_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["fl_joint", "fr_joint", "rl_joint", "rr_joint"],
            effort_limit=10.0,
            stiffness=0.0,
            damping=10.0,
        ),
    },
)


@configclass
class OgreNavigationEnvCfg(DirectRLEnvCfg):
    """Configuration for the Ogre mecanum navigation environment."""

    # Environment settings
    decimation = 4  # Physics steps per control step
    episode_length_s = 10.0  # Episode duration

    # CRITICAL: action_scale multiplies raw policy outputs to get wheel velocity in rad/s
    # Robot flips at velocities > 8 rad/s, so training targets stay within safe range
    # Max achievable linear vel = 8 * 0.04 = 0.32 m/s
    action_scale = 8.0  # Scales policy output to wheel velocity (rad/s)

    # Observation and action dimensions
    observation_space = 10  # target_vel(3) + current_vel(3) + wheel_vel(4)
    action_space = 4  # 4 wheel velocities
    state_space = 0

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,  # 120 Hz physics
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**21,  # Increased for many collisions
        ),
    )

    # Robot configuration
    robot_cfg: ArticulationCfg = OGRE_MECANUM_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Wheel joint names (must match USD)
    wheel_joint_names = ["fl_joint", "fr_joint", "rl_joint", "rr_joint"]

    # Robot physical parameters (from CLAUDE.md)
    wheel_radius = 0.040  # 40mm
    wheelbase = 0.095  # 95mm front-to-rear
    track_width = 0.205  # 205mm left-to-right

    # Scene configuration
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,  # Reduced for stability - can increase once working
        env_spacing=3.0,  # Increased spacing to reduce collisions
        replicate_physics=True,
    )

    # Target velocity ranges (for random sampling during training)
    # With action_scale=8, max wheel vel = 8 rad/s
    # Max achievable body vel = 8 × 0.04 = 0.32 m/s for pure translation
    # NOTE: These are the target velocity ranges for training - NOT limits!
    # The policy will learn to track velocities within this range.
    max_lin_vel = 8.0   # Target velocity range for training (m/s)
    max_ang_vel = 6.0   # Target angular velocity range (rad/s)

    # Reward scales - from working training run 2025-11-30_11-39-07
    rew_scale_vel_tracking = 2.0  # Main reward for velocity tracking
    rew_scale_vel_xy = 0.0  # Disabled - let main tracking handle it
    rew_scale_ang_vel = 0.0  # Disabled - let main tracking handle it
    rew_scale_exceed_limit = -1.0  # Penalty for wheel velocities exceeding max_wheel_vel
    rew_scale_smoothness = 0.0  # Disabled - focus on tracking first
    rew_scale_uprightness = 1.0  # Reward for staying upright (+1 upright, -1 flipped)
    rew_scale_symmetry = -1.0  # Penalty for asymmetric wheel velocities (prevents veering)

    # Safety limit for wheel velocities (robot flips above this)
    max_wheel_vel = 8.0  # rad/s - same as action_scale


class OgreNavigationEnv(DirectRLEnv):
    """Ogre mecanum robot velocity tracking environment."""

    cfg: OgreNavigationEnvCfg

    def __init__(self, cfg: OgreNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Find wheel joint indices
        self._wheel_joint_ids, _ = self.robot.find_joints(self.cfg.wheel_joint_names)

        # Store action scale
        self.action_scale = self.cfg.action_scale

        # Initialize target velocities (will be randomized each episode)
        self.target_vel = torch.zeros((self.num_envs, 3), device=self.device)  # vx, vy, vtheta

        # Previous actions for smoothness reward
        self.prev_actions = torch.zeros((self.num_envs, 4), device=self.device)

        # Compute L parameter for mecanum kinematics
        self.L = (self.cfg.wheelbase + self.cfg.track_width) / 2.0

    def _setup_scene(self):
        """Set up the simulation scene."""
        self.robot = Articulation(self.cfg.robot_cfg)

        # Add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # Clone environments
        self.scene.clone_environments(copy_from_source=False)

        # Filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Add robot to scene
        self.scene.articulations["robot"] = self.robot

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Process actions before physics step."""
        self.actions = self.action_scale * actions.clone()

    def _apply_action(self) -> None:
        """Apply wheel velocity targets to the robot.

        VERIFIED JOINT ORDER from find_joints(["fl_joint", "fr_joint", "rl_joint", "rr_joint"]):
            Index 0 = FR (front-right)
            Index 1 = RR (rear-right)
            Index 2 = RL (rear-left)
            Index 3 = FL (front-left)

        EMPIRICALLY TESTED: With [+6, +6, +6, +6] the robot spins CW (clockwise).
        - Right wheels (FR, RR) spin backward with positive velocity
        - Left wheels (RL, FL) spin forward with positive velocity

        To make [+,+,+,+] = forward motion, negate RIGHT wheels (indices 0, 1).
        """
        # Apply sign corrections for right wheels to scaled actions
        corrected_actions = self.actions.clone()
        corrected_actions[:, 0] *= -1  # FR (right wheel)
        corrected_actions[:, 1] *= -1  # RR (right wheel)
        self.robot.set_joint_velocity_target(corrected_actions, joint_ids=self._wheel_joint_ids)

    def _get_observations(self) -> dict:
        """Compute observations for the policy.

        VERIFIED JOINT ORDER: [FR, RR, RL, FL] (indices 0, 1, 2, 3)

        Wheel velocities are corrected to match the normalized action space
        where positive values = forward motion for all wheels. This matches
        the sign corrections applied in _apply_action().
        """
        # Get current robot state
        root_vel = self.robot.data.root_lin_vel_b  # Linear velocity in body frame
        root_ang_vel = self.robot.data.root_ang_vel_b  # Angular velocity in body frame
        joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids].clone()

        # Negate RIGHT wheel velocities to match normalized action space
        # FR=index 0, RR=index 1
        joint_vel[:, 0] *= -1  # FR
        joint_vel[:, 1] *= -1  # RR

        # Current body velocity (vx, vy, vtheta)
        current_vel = torch.cat([
            root_vel[:, 0:1],  # vx (forward)
            root_vel[:, 1:2],  # vy (left)
            root_ang_vel[:, 2:3],  # vtheta (yaw rate)
        ], dim=-1)

        # Concatenate observations
        obs = torch.cat([
            self.target_vel,  # Target velocity command (3)
            current_vel,  # Current velocity (3)
            joint_vel,  # Wheel velocities (4) - corrected for normalized space
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get current velocities
        root_vel = self.robot.data.root_lin_vel_b
        root_ang_vel = self.robot.data.root_ang_vel_b

        current_vel = torch.cat([
            root_vel[:, 0:1],
            root_vel[:, 1:2],
            root_ang_vel[:, 2:3],
        ], dim=-1)

        # Velocity tracking error
        vel_error = self.target_vel - current_vel

        # Compute uprightness from quaternion (how much the robot is tilted)
        # Get the z-component of the "up" vector in world frame
        root_quat = self.robot.data.root_quat_w  # (num_envs, 4) wxyz format
        qw, qx, qy, qz = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
        up_z = 1.0 - 2.0 * (qx * qx + qy * qy)  # 1.0 = upright, -1.0 = upside down

        # Rewards
        rewards = compute_rewards(
            vel_error=vel_error,
            target_vel=self.target_vel,
            actions=self.actions,
            prev_actions=self.prev_actions,
            up_z=up_z,
            rew_scale_vel_tracking=self.cfg.rew_scale_vel_tracking,
            rew_scale_vel_xy=self.cfg.rew_scale_vel_xy,
            rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
            rew_scale_exceed_limit=self.cfg.rew_scale_exceed_limit,
            max_wheel_vel=self.cfg.max_wheel_vel,
            rew_scale_smoothness=self.cfg.rew_scale_smoothness,
            rew_scale_uprightness=self.cfg.rew_scale_uprightness,
            rew_scale_symmetry=self.cfg.rew_scale_symmetry,
        )

        # Store current actions for next step
        self.prev_actions = self.actions.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # Terminate if robot has flipped (base height drops below threshold)
        # Robot base_link is at wheel axle height (0.04m) when upright
        # If base drops below 0.02m (half wheel radius), robot has flipped
        root_pos_z = self.robot.data.root_pos_w[:, 2]  # Z position of base
        env_origins_z = self.scene.env_origins[:, 2]
        height_above_ground = root_pos_z - env_origins_z

        # Flipped if base is below half wheel radius (0.02m)
        flipped = height_above_ground < 0.02

        terminated = flipped

        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset environments."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Sample new random target velocities
        self.target_vel[env_ids, 0] = sample_uniform(
            -self.cfg.max_lin_vel, self.cfg.max_lin_vel,
            (num_reset,), self.device
        )
        self.target_vel[env_ids, 1] = sample_uniform(
            -self.cfg.max_lin_vel, self.cfg.max_lin_vel,
            (num_reset,), self.device
        )
        self.target_vel[env_ids, 2] = sample_uniform(
            -self.cfg.max_ang_vel, self.cfg.max_ang_vel,
            (num_reset,), self.device
        )

        # Reset previous actions
        self.prev_actions[env_ids] = 0.0


@torch.jit.script
def compute_rewards(
    vel_error: torch.Tensor,
    target_vel: torch.Tensor,
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    up_z: torch.Tensor,
    rew_scale_vel_tracking: float,
    rew_scale_vel_xy: float,
    rew_scale_ang_vel: float,
    rew_scale_exceed_limit: float,
    max_wheel_vel: float,
    rew_scale_smoothness: float,
    rew_scale_uprightness: float,
    rew_scale_symmetry: float,
) -> torch.Tensor:
    """Compute rewards for velocity tracking.

    Args:
        vel_error: Velocity tracking error [vx_err, vy_err, vtheta_err]
        target_vel: Target velocity command [vx, vy, vtheta]
        actions: Current wheel velocity commands (in rad/s, already scaled)
        prev_actions: Previous wheel velocity commands
        up_z: Z-component of robot's up vector (1.0=upright, -1.0=flipped)
        rew_scale_exceed_limit: Penalty scale for exceeding velocity limit
        max_wheel_vel: Maximum safe wheel velocity (rad/s)
        rew_scale_*: Other reward scaling factors

    Returns:
        Total reward tensor
    """
    # Velocity tracking reward using softer exponential decay
    vel_error_norm = torch.sum(vel_error ** 2, dim=-1)
    rew_vel_tracking = rew_scale_vel_tracking * torch.exp(-vel_error_norm / 0.25)

    # Extra reward for xy velocity tracking (if enabled)
    xy_error = torch.sum(vel_error[:, :2] ** 2, dim=-1)
    rew_vel_xy = rew_scale_vel_xy * torch.exp(-xy_error / 0.25)

    # Angular velocity tracking (if enabled)
    ang_error = vel_error[:, 2] ** 2
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-ang_error / 0.25)

    # Exceed limit penalty: Only penalize wheel velocities ABOVE the safe threshold
    # This allows any velocity 0-8 rad/s without penalty, but strongly penalizes > 8
    abs_actions = torch.abs(actions)
    excess = torch.clamp(abs_actions - max_wheel_vel, min=0.0)
    rew_exceed_limit = rew_scale_exceed_limit * torch.sum(excess ** 2, dim=-1)

    # Smoothness penalty (penalize jerky motion)
    action_diff = actions - prev_actions
    rew_smoothness = rew_scale_smoothness * torch.sum(action_diff ** 2, dim=-1)

    # Uprightness reward: +1 when upright (up_z=1), -1 when flipped (up_z=-1)
    rew_uprightness = rew_scale_uprightness * up_z

    # Wheel symmetry penalty - penalize left-right imbalance during straight motion
    # Policy outputs actions as [FL, FR, RL, RR] = indices [0, 1, 2, 3]
    # (Note: _apply_action() reorders these to match joint_ids order)
    # Left wheels: FL (0), RL (2)
    # Right wheels: FR (1), RR (3)
    left_avg = (actions[:, 0] + actions[:, 2]) / 2.0   # (FL + RL) / 2
    right_avg = (actions[:, 1] + actions[:, 3]) / 2.0  # (FR + RR) / 2

    # Only apply symmetry penalty when not rotating (vtheta near zero)
    rotation_mask = torch.abs(target_vel[:, 2]) < 0.5  # rad/s threshold

    # Left-right symmetry error (squared difference)
    lr_symmetry_error = (left_avg - right_avg) ** 2

    # Apply penalty only when not rotating
    rew_symmetry = rew_scale_symmetry * lr_symmetry_error * rotation_mask.float()

    total_reward = rew_vel_tracking + rew_vel_xy + rew_ang_vel + rew_exceed_limit + rew_smoothness + rew_uprightness + rew_symmetry

    return total_reward
