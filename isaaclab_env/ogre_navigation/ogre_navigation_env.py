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
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform


# Robot configuration - matches your ogre.usd robot
OGRE_MECANUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/brad/ros2_ws/src/ogre-slam/usds/ogre.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),  # Slightly above ground
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
        "wheels": sim_utils.ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            velocity_limit=200.0,  # rad/s - high for simulation
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
    action_scale = 50.0  # Scale for wheel velocity actions (rad/s)

    # Observation and action dimensions
    observation_space = 10  # target_vel(3) + current_vel(3) + wheel_vel(4)
    action_space = 4  # 4 wheel velocities
    state_space = 0

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,  # 120 Hz physics
        render_interval=decimation,
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
        num_envs=4096,  # Parallel environments for fast training
        env_spacing=2.0,
        replicate_physics=True,
    )

    # Target velocity ranges (for random sampling during training)
    max_lin_vel = 8.0  # m/s - max linear velocity
    max_ang_vel = 6.0  # rad/s - max angular velocity

    # Reward scales
    rew_scale_vel_tracking = 5.0  # Reward for accurate velocity tracking
    rew_scale_vel_xy = 2.0  # Extra reward for xy velocity accuracy
    rew_scale_ang_vel = 1.0  # Reward for angular velocity accuracy
    rew_scale_energy = -0.001  # Penalty for energy use
    rew_scale_smoothness = -0.01  # Penalty for jerky actions


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
        """Apply wheel velocity targets to the robot."""
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self._wheel_joint_ids)

    def _get_observations(self) -> dict:
        """Compute observations for the policy."""
        # Get current robot state
        root_vel = self.robot.data.root_lin_vel_b  # Linear velocity in body frame
        root_ang_vel = self.robot.data.root_ang_vel_b  # Angular velocity in body frame
        joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]

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
            joint_vel,  # Wheel velocities (4)
        ], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards."""
        # Get current velocities
        root_vel = self.robot.data.root_lin_vel_b
        root_ang_vel = self.robot.data.root_ang_vel_b
        joint_vel = self.robot.data.joint_vel[:, self._wheel_joint_ids]

        current_vel = torch.cat([
            root_vel[:, 0:1],
            root_vel[:, 1:2],
            root_ang_vel[:, 2:3],
        ], dim=-1)

        # Velocity tracking error
        vel_error = self.target_vel - current_vel

        # Rewards
        rewards = compute_rewards(
            vel_error=vel_error,
            actions=self.actions,
            prev_actions=self.prev_actions,
            rew_scale_vel_tracking=self.cfg.rew_scale_vel_tracking,
            rew_scale_vel_xy=self.cfg.rew_scale_vel_xy,
            rew_scale_ang_vel=self.cfg.rew_scale_ang_vel,
            rew_scale_energy=self.cfg.rew_scale_energy,
            rew_scale_smoothness=self.cfg.rew_scale_smoothness,
        )

        # Store current actions for next step
        self.prev_actions = self.actions.clone()

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions."""
        # Timeout
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # No early termination for velocity tracking - let it learn from mistakes
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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
    actions: torch.Tensor,
    prev_actions: torch.Tensor,
    rew_scale_vel_tracking: float,
    rew_scale_vel_xy: float,
    rew_scale_ang_vel: float,
    rew_scale_energy: float,
    rew_scale_smoothness: float,
) -> torch.Tensor:
    """Compute rewards for velocity tracking.

    Args:
        vel_error: Velocity tracking error [vx_err, vy_err, vtheta_err]
        actions: Current wheel velocity commands
        prev_actions: Previous wheel velocity commands
        rew_scale_*: Reward scaling factors

    Returns:
        Total reward tensor
    """
    # Velocity tracking reward (exponential - peaks at zero error)
    vel_error_norm = torch.sum(vel_error ** 2, dim=-1)
    rew_vel_tracking = rew_scale_vel_tracking * torch.exp(-vel_error_norm)

    # Extra reward for xy velocity tracking
    xy_error = torch.sum(vel_error[:, :2] ** 2, dim=-1)
    rew_vel_xy = rew_scale_vel_xy * torch.exp(-xy_error)

    # Angular velocity tracking
    ang_error = vel_error[:, 2] ** 2
    rew_ang_vel = rew_scale_ang_vel * torch.exp(-ang_error)

    # Energy penalty (penalize high wheel velocities)
    rew_energy = rew_scale_energy * torch.sum(actions ** 2, dim=-1)

    # Smoothness penalty (penalize jerky motion)
    action_diff = actions - prev_actions
    rew_smoothness = rew_scale_smoothness * torch.sum(action_diff ** 2, dim=-1)

    total_reward = rew_vel_tracking + rew_vel_xy + rew_ang_vel + rew_energy + rew_smoothness

    return total_reward
