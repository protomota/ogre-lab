#!/usr/bin/env python3
"""
Print the actual joint order as returned by Isaac Lab.

Run with:
    cd ~/isaac-lab/IsaacLab
    ./isaaclab.sh -p ~/ogre-lab/scripts/print_joint_order.py
"""

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/brad/ros2_ws/src/ogre-slam/usds/ogre_robot.usd",
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.04),
        joint_pos={"fl_joint": 0.0, "fr_joint": 0.0, "rl_joint": 0.0, "rr_joint": 0.0},
        joint_vel={"fl_joint": 0.0, "fr_joint": 0.0, "rl_joint": 0.0, "rr_joint": 0.0},
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

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1/120)
    sim = sim_utils.SimulationContext(sim_cfg)

    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    robot_cfg = ROBOT_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()
    robot.reset()

    # Get ALL joints
    all_joint_ids, all_joint_names = robot.find_joints([".*"])
    print(f"\n{'='*60}")
    print("ALL JOINTS IN USD:")
    for i, name in enumerate(all_joint_names):
        print(f"  Index {i}: {name}")

    # Get wheel joints with our specific query
    wheel_joint_ids, wheel_joint_names = robot.find_joints(["fl_joint", "fr_joint", "rl_joint", "rr_joint"])
    print(f"\n{'='*60}")
    print("WHEEL JOINTS (queried in order [fl, fr, rl, rr]):")
    print(f"  Joint IDs returned: {wheel_joint_ids}")
    print(f"  Joint names returned: {wheel_joint_names}")
    print()
    print("This means when we send velocity tensor [v0, v1, v2, v3]:")
    for i, (jid, name) in enumerate(zip(wheel_joint_ids, wheel_joint_names)):
        print(f"  Index {i} (v{i}) -> Joint ID {jid} -> {name}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
    simulation_app.close()
