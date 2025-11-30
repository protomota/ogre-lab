#!/usr/bin/env python3
"""
Test each wheel individually to identify which joint index corresponds to which wheel.

Run with:
    cd ~/isaac-lab/IsaacLab
    ./isaaclab.sh -p ~/ogre-lab/scripts/test_each_wheel.py
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
    """Test each wheel individually."""
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

    # Get wheel joint IDs - the order matters!
    wheel_joint_ids, wheel_names = robot.find_joints(["fl_joint", "fr_joint", "rl_joint", "rr_joint"])

    print(f"\n{'='*60}")
    print("WHEEL IDENTIFICATION TEST")
    print(f"Joint IDs: {wheel_joint_ids}")
    print(f"Joint names (in order): {wheel_names}")
    print(f"{'='*60}\n")

    # Test each wheel one at a time - LOOP CONTINUOUSLY
    # VERIFIED ORDER: [FR, RR, RL, FL] - NOT the query order!
    wheel_labels = ["FR (index 0)", "RR (index 1)", "RL (index 2)", "FL (index 3)"]

    print("\n" + "="*60)
    print("LOOPING TEST - Press Ctrl+C to exit")
    print("Each wheel will spin for 4 seconds, then pause 2 seconds")
    print("="*60)

    loop_count = 0
    while simulation_app.is_running():
        loop_count += 1
        print(f"\n{'='*60}")
        print(f"LOOP {loop_count}")
        print(f"{'='*60}")

        for i in range(4):
            print(f"\n>>> INDEX {i}: Testing with +6 rad/s")
            print(f"    (Expected: {wheel_labels[i]})")
            print("    WATCH: Which physical wheel spins?")

            # Create velocity tensor with only one wheel active
            test_vel = torch.zeros((1, 4), device=sim.device)
            test_vel[0, i] = 6.0

            start_time = time.time()
            while time.time() - start_time < 4.0:  # 4 seconds per wheel
                robot.set_joint_velocity_target(test_vel, joint_ids=wheel_joint_ids)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.cfg.dt)

            # Stop
            print("    Stopping...")
            zero_vel = torch.zeros((1, 4), device=sim.device)
            for _ in range(240):  # 2 seconds stop
                robot.set_joint_velocity_target(zero_vel, joint_ids=wheel_joint_ids)
                robot.write_data_to_sim()
                sim.step()
                robot.update(sim.cfg.dt)

        print(f"\n--- End of loop {loop_count}, starting over ---")


if __name__ == "__main__":
    main()
    simulation_app.close()
