# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Ogre Mecanum Robot Navigation Environment.

This environment trains a velocity tracking controller for the Ogre mecanum drive robot.
The policy learns to efficiently execute velocity commands (vx, vy, vtheta) by controlling
individual wheel velocities, optimized for use with Nav2 navigation stack.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Ogre-Navigation-Direct-v0",
    entry_point=f"{__name__}.ogre_navigation_env:OgreNavigationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ogre_navigation_env:OgreNavigationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OgreNavigationPPORunnerCfg",
    },
)
