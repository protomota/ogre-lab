#!/usr/bin/env python3
"""
Test script to empirically determine USD wheel joint axis orientation.

This script sends velocity commands directly to the wheel joints without
any sign corrections, and observes which way the robot moves.

Run with:
    cd ~/isaac-lab/IsaacLab
    ./isaaclab.sh -p ~/ogre-lab/scripts/test_wheel_direction.py

Watch the robot in the GUI. The script will:
1. Send [+6, +6, +6, +6] rad/s for 3 seconds - expect CW spin
2. Stop for 2 seconds
3. Send [+6, -6, +6, -6] rad/s for 3 seconds - expect FORWARD motion
4. Report results

FINDING: The right wheels (FR, RR) have opposite axis orientation.
- [+6,+6,+6,+6]: Left wheels forward, right wheels backward → CW spin
- [+6,-6,+6,-6]: All wheels forward → forward motion
For forward motion: [+, -, +, -] or negate FR and RR.
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
    """Test wheel direction."""
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

    # Record initial position
    initial_pos = robot.data.root_pos_w[0].clone()
    print(f"\n{'='*60}")
    print(f"Initial position: X={initial_pos[0]:.4f}, Y={initial_pos[1]:.4f}")
    print(f"{'='*60}\n")

    # Test 1: All positive [+6, +6, +6, +6] - expect CW spin
    print("TEST 1: Sending [+6, +6, +6, +6] rad/s to all wheels...")
    print("Expected: CW spin (proving RIGHT wheels have opposite axis)")

    test_vel = torch.tensor([[6.0, 6.0, 6.0, 6.0]], device=sim.device)

    start_time = time.time()
    while time.time() - start_time < 3.0:
        robot.set_joint_velocity_target(test_vel, joint_ids=wheel_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    pos_after = robot.data.root_pos_w[0].clone()
    delta = pos_after - initial_pos
    print(f"\nAfter [+,+,+,+]: Moved X={delta[0]:.4f}m, Y={delta[1]:.4f}m")

    # Stop and reset
    print("\nStopping and resetting...")
    zero_vel = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=sim.device)
    for _ in range(60):
        robot.set_joint_velocity_target(zero_vel, joint_ids=wheel_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    # Reset position
    root_state = robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    for _ in range(60):
        sim.step()
        robot.update(sim.cfg.dt)

    initial_pos = robot.data.root_pos_w[0].clone()

    # Test 2: Corrected [+6, -6, +6, -6] - expect FORWARD motion
    print("\nTEST 2: Sending [+6, -6, +6, -6] rad/s (RIGHT wheels negated)...")
    print("Expected: FORWARD motion (straight ahead)")

    # FL=+6, FR=-6, RL=+6, RR=-6
    test_vel = torch.tensor([[6.0, -6.0, 6.0, -6.0]], device=sim.device)

    start_time = time.time()
    while time.time() - start_time < 3.0:
        robot.set_joint_velocity_target(test_vel, joint_ids=wheel_joint_ids)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.cfg.dt)

    pos_after = robot.data.root_pos_w[0].clone()
    delta = pos_after - initial_pos
    print(f"\nAfter [+,-,+,-]: Moved X={delta[0]:.4f}m, Y={delta[1]:.4f}m")

    if delta[0] > 0.1:
        print("\n✓ SUCCESS: Robot moved FORWARD with [+,-,+,-]!")
        print("  This confirms: negate FR and RR for forward motion")
    elif delta[0] < -0.1:
        print("\n✗ Robot moved BACKWARD - may need opposite correction")
    else:
        print("\n? Robot didn't move much in X - check simulation")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print("  [+6,+6,+6,+6] → CW spin (RIGHT wheels have opposite axis)")
    print("  [+6,-6,+6,-6] → Should be forward motion")
    print("")
    print("TRAINING CORRECTION NEEDED:")
    print("  In _apply_action(): negate FR (index 1) and RR (index 3)")
    print("  In _get_observations(): same correction for consistency")
    print(f"{'='*60}")

    print("\nKeeping simulation open. Press Ctrl+C to exit.")

    # Keep sim running for observation
    while simulation_app.is_running():
        sim.step()
        robot.update(sim.cfg.dt)


if __name__ == "__main__":
    main()
    simulation_app.close()
