# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL PPO configuration for Ogre Navigation environment."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class OgreNavigationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for Ogre mecanum navigation."""

    num_steps_per_env = 24
    max_iterations = 1000  # Train for 1000 iterations
    save_interval = 100  # Save checkpoint every 100 iterations
    experiment_name = "ogre_navigation"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[128, 128, 64],  # Larger network for complex dynamics
        critic_hidden_dims=[128, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Encourage exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
