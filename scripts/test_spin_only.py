#!/usr/bin/env python3
"""
Simple test: Send [+6, +6, +6, +6] to all wheels and observe spin direction.

Run with:
    cd ~/isaac-lab/IsaacLab
    ./isaaclab.sh -p ~/ogre-lab/scripts/test_spin_only.py
"""

import torch
import time
from isaaclab.app import AppLauncher

# Launch Isaac Sim with GUI
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

# Now import sim-dependent modules
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# Robot config
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
    """Test spin direction with all positive velocities."""
    # Create sim context
    sim_cfg = sim_utils.SimulationCfg(dt=1/120)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.0])

    # Setup ground
    spawn_ground_plane("/World/ground", GroundPlaneCfg())

    # Spawn robot
    robot_cfg = ROBOT_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    # Play simulation
    sim.reset()
    robot.reset()

    # Get wheel joint IDs
    wheel_joint_ids, _ = robot.find_joints(["fl_joint", "fr_joint", "rl_joint", "rr_joint"])

    print(f"\n{'='*60}")
    print("SPIN TEST: Sending [+6, +6, +6, +6] rad/s to all wheels")
    print("Watch the robot - which way does it spin?")
    print("  - CW (clockwise) = LEFT wheels have opposite axis")
    print("  - CCW (counter-clockwise) = RIGHT wheels have opposite axis")
    print(f"{'='*60}\n")

    # Send all positive velocities
    test_vel = torch.tensor([[6.0, 6.0, 6.0, 6.0]], device=sim.device)

    print("Starting spin test... (runs for 5 seconds)")
    start_time = time.time()
    while time.time() - start_time < 5.0:
        robot.set_joint_velocity_target(test_vel, joint_ids=wheel_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    print("\nTest complete. Keeping simulation open.")
    print("Press Ctrl+C to exit.\n")

    # Keep sim running for observation
    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
