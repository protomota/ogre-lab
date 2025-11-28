#!/usr/bin/env python3
"""
Ogre Policy Controller Node

This ROS2 node runs a trained RL policy to control the Ogre mecanum robot.
It acts as an intermediary between Nav2 and the robot, using a learned policy
to convert velocity commands to optimal wheel velocities, then back to Twist.

The policy was trained in Isaac Lab to efficiently track velocity commands
for a mecanum drive robot.

Mode 1 (default): Twist-to-Twist
    Subscribes: /policy_cmd_vel_in (Twist from Nav2)
    Publishes:  /cmd_vel (Twist to robot/simulator)

Mode 2: Twist-to-WheelVelocities
    Subscribes: /cmd_vel (Twist)
    Publishes:  /wheel_velocities (Float32MultiArray)

The policy improves velocity tracking by learning the robot's dynamics.
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

# Try to import ONNX runtime, fall back to PyTorch JIT
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARN] onnxruntime not available, will try PyTorch JIT")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available")


class PolicyControllerNode(Node):
    """ROS2 node that runs trained RL policy for mecanum robot control."""

    def __init__(self):
        super().__init__('ogre_policy_controller')

        # Declare parameters
        self.declare_parameter('use_policy', False)  # If False, pass through Twist directly
        self.declare_parameter('model_path', '')
        self.declare_parameter('model_type', 'onnx')  # 'onnx' or 'jit'
        self.declare_parameter('action_scale', 10.0)
        self.declare_parameter('max_lin_vel', 0.5)
        self.declare_parameter('max_ang_vel', 1.0)
        self.declare_parameter('control_frequency', 30.0)
        self.declare_parameter('wheel_joint_names', ['fl_joint', 'fr_joint', 'rl_joint', 'rr_joint'])
        self.declare_parameter('output_mode', 'twist')  # 'twist' or 'wheel_velocities'
        self.declare_parameter('input_topic', '/policy_cmd_vel_in')
        self.declare_parameter('output_topic', '/cmd_vel')

        # Robot parameters for inverse kinematics (wheel vel -> twist)
        self.declare_parameter('wheel_radius', 0.040)  # 40mm
        self.declare_parameter('wheelbase', 0.095)     # 95mm
        self.declare_parameter('track_width', 0.205)   # 205mm

        # Get parameters
        self.use_policy = self.get_parameter('use_policy').get_parameter_value().bool_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        model_type = self.get_parameter('model_type').get_parameter_value().string_value
        self.action_scale = self.get_parameter('action_scale').get_parameter_value().double_value
        self.max_lin_vel = self.get_parameter('max_lin_vel').get_parameter_value().double_value
        self.max_ang_vel = self.get_parameter('max_ang_vel').get_parameter_value().double_value
        control_freq = self.get_parameter('control_frequency').get_parameter_value().double_value
        self.wheel_joint_names = self.get_parameter('wheel_joint_names').get_parameter_value().string_array_value
        self.output_mode = self.get_parameter('output_mode').get_parameter_value().string_value
        input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        output_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        # Robot parameters
        self.wheel_radius = self.get_parameter('wheel_radius').get_parameter_value().double_value
        self.wheelbase = self.get_parameter('wheelbase').get_parameter_value().double_value
        self.track_width = self.get_parameter('track_width').get_parameter_value().double_value
        self.L = (self.wheelbase + self.track_width) / 2.0

        # Load the policy model only if use_policy is True
        self.model = None
        self.model_type = model_type
        if self.use_policy:
            # Find model if not specified
            if not model_path:
                model_path = self._find_default_model(model_type)
            self._load_model(model_path, model_type)

        # State variables
        self.target_vel = np.zeros(3, dtype=np.float32)  # vx, vy, vtheta
        self.current_vel = np.zeros(3, dtype=np.float32)  # vx, vy, vtheta
        self.wheel_vel = np.zeros(4, dtype=np.float32)  # fl, fr, rl, rr
        self.last_cmd_time = self.get_clock().now()

        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriber for velocity commands
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            input_topic,
            self._cmd_vel_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self._odom_callback,
            sensor_qos
        )

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            sensor_qos
        )

        # Publisher based on output mode
        if self.output_mode == 'twist':
            self.output_pub = self.create_publisher(Twist, output_topic, 10)
        else:
            self.output_pub = self.create_publisher(Float32MultiArray, output_topic, 10)

        # Control loop timer
        self.control_timer = self.create_timer(
            1.0 / control_freq,
            self._control_loop
        )

        self.get_logger().info(f'Ogre Policy Controller started')
        self.get_logger().info(f'  Use policy: {self.use_policy}')
        if self.use_policy:
            self.get_logger().info(f'  Model: {model_path}')
            self.get_logger().info(f'  Type: {model_type}')
            self.get_logger().info(f'  Action scale: {self.action_scale}')
        else:
            self.get_logger().info(f'  Mode: Pass-through (Twist in -> Twist out)')
        self.get_logger().info(f'  Input topic: {input_topic}')
        self.get_logger().info(f'  Output topic: {output_topic}')
        self.get_logger().info(f'  Control frequency: {control_freq} Hz')

    def _find_default_model(self, model_type: str) -> str:
        """Find default model in standard locations."""
        # Check relative to this package
        pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        search_paths = [
            os.path.join(pkg_dir, '..', 'models'),
            os.path.join(pkg_dir, 'models'),
            os.path.expanduser('~/ogre-lab/models'),
        ]

        ext = '.onnx' if model_type == 'onnx' else '.pt'
        filename = f'policy{ext}'

        for path in search_paths:
            full_path = os.path.join(path, filename)
            if os.path.exists(full_path):
                return full_path

        raise FileNotFoundError(
            f"Could not find {filename} in: {search_paths}. "
            f"Please specify model_path parameter."
        )

    def _load_model(self, model_path: str, model_type: str):
        """Load the policy model."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        if model_type == 'onnx':
            if not ONNX_AVAILABLE:
                raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")
            self.model = ort.InferenceSession(model_path)
            self.input_name = self.model.get_inputs()[0].name
            self.get_logger().info(f'Loaded ONNX model: {model_path}')

        elif model_type == 'jit':
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not installed. Install with: pip install torch")
            self.model = torch.jit.load(model_path)
            self.model.eval()
            self.get_logger().info(f'Loaded JIT model: {model_path}')

        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'onnx' or 'jit'")

    def _cmd_vel_callback(self, msg: Twist):
        """Handle incoming velocity commands from Nav2."""
        # Clip to max velocities
        self.target_vel[0] = np.clip(msg.linear.x, -self.max_lin_vel, self.max_lin_vel)
        self.target_vel[1] = np.clip(msg.linear.y, -self.max_lin_vel, self.max_lin_vel)
        self.target_vel[2] = np.clip(msg.angular.z, -self.max_ang_vel, self.max_ang_vel)
        self.last_cmd_time = self.get_clock().now()

    def _odom_callback(self, msg: Odometry):
        """Handle odometry for current velocity feedback."""
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.linear.y
        self.current_vel[2] = msg.twist.twist.angular.z

    def _joint_state_callback(self, msg: JointState):
        """Handle joint states for wheel velocity feedback."""
        for i, name in enumerate(self.wheel_joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.velocity):
                    self.wheel_vel[i] = msg.velocity[idx]

    def _build_observation(self) -> np.ndarray:
        """Build observation vector for the policy.

        Observation space (10 dimensions):
            [0-2]: Target velocity (vx, vy, vtheta)
            [3-5]: Current velocity (vx, vy, vtheta)
            [6-9]: Wheel velocities (fl, fr, rl, rr)
        """
        obs = np.concatenate([
            self.target_vel,
            self.current_vel,
            self.wheel_vel
        ]).astype(np.float32)

        return obs.reshape(1, -1)  # Batch dimension

    def _run_policy(self, obs: np.ndarray) -> np.ndarray:
        """Run the policy network to get wheel velocity actions."""
        if self.model_type == 'onnx':
            outputs = self.model.run(None, {self.input_name: obs})
            actions = outputs[0][0]  # Remove batch dimension
        else:  # jit
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs)
                actions = self.model(obs_tensor).numpy()[0]

        return actions

    def _wheel_vel_to_twist(self, wheel_vel: np.ndarray) -> Twist:
        """Convert wheel velocities to Twist using forward kinematics.

        Mecanum forward kinematics:
            vx = (w_fl + w_fr + w_rl + w_rr) * R / 4
            vy = (-w_fl + w_fr + w_rl - w_rr) * R / 4
            vtheta = (-w_fl + w_fr - w_rl + w_rr) * R / (4 * L)

        Where:
            w_fl, w_fr, w_rl, w_rr = wheel angular velocities (rad/s)
            R = wheel radius
            L = (wheelbase + track_width) / 2

        The policy outputs wheel velocities directly without sign corrections.
        Isaac Sim action graph handles any joint axis corrections for deployment.
        """
        w_fl = wheel_vel[0]
        w_fr = wheel_vel[1]
        w_rl = wheel_vel[2]
        w_rr = wheel_vel[3]

        R = self.wheel_radius
        L = self.L

        vx = (w_fl + w_fr + w_rl + w_rr) * R / 4.0
        vy = (-w_fl + w_fr + w_rl - w_rr) * R / 4.0
        vtheta = (-w_fl + w_fr - w_rl + w_rr) * R / (4.0 * L)

        twist = Twist()
        twist.linear.x = float(vx)
        twist.linear.y = float(vy)
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = float(vtheta)

        return twist

    def _control_loop(self):
        """Main control loop - runs at control_frequency."""
        # Check for stale commands (stop if no cmd_vel for 0.5s)
        time_since_cmd = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if time_since_cmd > 0.5:
            self.target_vel[:] = 0.0

        # Pass-through mode: just forward the Twist directly
        if not self.use_policy:
            twist_msg = Twist()
            twist_msg.linear.x = float(self.target_vel[0])
            twist_msg.linear.y = float(self.target_vel[1])
            twist_msg.linear.z = 0.0
            twist_msg.angular.x = 0.0
            twist_msg.angular.y = 0.0
            twist_msg.angular.z = float(self.target_vel[2])
            self.output_pub.publish(twist_msg)
            return

        # Policy mode: run the learned policy
        # Short-circuit: if target velocity is near zero, output zero (don't run policy)
        target_magnitude = np.sqrt(np.sum(self.target_vel ** 2))
        if target_magnitude < 0.01:  # Threshold for "zero" command
            if self.output_mode == 'twist':
                twist_msg = Twist()  # All zeros by default
                self.output_pub.publish(twist_msg)
            else:
                msg = Float32MultiArray()
                msg.data = [0.0, 0.0, 0.0, 0.0]
                self.output_pub.publish(msg)
            return

        # Build observation and run policy
        obs = self._build_observation()
        actions = self._run_policy(obs)

        # Clip actions to [-1, 1] range (RSL-RL actor outputs unbounded values)
        actions = np.clip(actions, -1.0, 1.0)

        # Scale actions to wheel velocities (rad/s)
        wheel_velocities = actions * self.action_scale

        # Debug logging (every 30 iterations = 1 second)
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0
        self._debug_counter += 1
        if self._debug_counter % 30 == 0:
            self.get_logger().info(
                f'Target: [{self.target_vel[0]:.3f}, {self.target_vel[1]:.3f}, {self.target_vel[2]:.3f}] | '
                f'Raw actions: [{actions[0]:.4f}, {actions[1]:.4f}, {actions[2]:.4f}, {actions[3]:.4f}] | '
                f'Wheel vel: [{wheel_velocities[0]:.2f}, {wheel_velocities[1]:.2f}, {wheel_velocities[2]:.2f}, {wheel_velocities[3]:.2f}]'
            )

        # Publish based on output mode
        if self.output_mode == 'twist':
            # Convert wheel velocities back to Twist
            twist_msg = self._wheel_vel_to_twist(wheel_velocities)
            self.output_pub.publish(twist_msg)
        else:
            # Publish raw wheel velocities
            msg = Float32MultiArray()
            msg.data = wheel_velocities.tolist()
            self.output_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = PolicyControllerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
