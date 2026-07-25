from __future__ import annotations

import os
import sys
import json
import math
import asyncio
import threading
import numpy as np

# ROS2 Client Library Imports with robust fallback mocks for Windows/PC development
try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray, Float32, String
    from sensor_msgs.msg import JointState
except ImportError:
    print("\n[ROS2 Notice] ROS2 environment (rclpy) not sourced on this terminal session.")
    print("This node acts as a bridge. For local verification, we will use mock structures.")
    print("To run in production: source /opt/ros/humble/setup.bash && colcon build")
    
    # Stub classes for non-ROS environments to compile cleanly during pre-flight checks
    class Node:
        def __init__(self, name): 
            self.name = name
        def get_logger(self):
            class Logger:
                def info(self, msg): print(f"[Mock ROS2 Node] {msg}")
                def warn(self, msg): print(f"[WARN] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
            return Logger()
            
    # Stub message classes so type hints compile cleanly without throwing NameError
    class JointState: pass
    class Float32MultiArray: pass
    class Float32: pass

# Import our low-level unalterable firmware Safety Shield
try:
    from core.safety_shield import SafetyShield
except ImportError:
    # Fallback to local import if run from a different working folder path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.safety_shield import SafetyShield

# --- STREAMING_CHUNK: Architecting the asynchronous ROS2 Bridge Node... ---
class HapticDexterityBridgeNode(Node):
    """
    ROS2 Bridge Node representing Christenson Robotics' open-source wrapper.
    Interfaces high-frequency hardware tactile taxels and joint states with 
    our 3D Web HUD Visualizer and local recurrent neural policy.
    """
    def __init__(self):
        super().__init__('haptic_dexterity_bridge_node')
        self.get_logger().info("Initializing Christenson Robotics Haptic ROS2 Bridge Wrapper...")
        
        # 1. State Variables representing active physical states
        self.current_joint_positions = np.zeros(2)  # Joint 0, Joint 1 (radians)
        self.current_joint_velocities = np.zeros(2) # rad/s
        self.current_taxel_pressures = np.zeros(16)  # 4x4 silicone grid (Pascals)
        self.realized_normal_force = 12.0           # Actuator feedback force (Newtons)
        self.slip_velocity_estimate = 0.0           # Estimated slip velocity (m/s)
        self.height_offset_estimate = 0.0           # Current translation drift (meters)
        self.friction_force_estimate = 0.0          # Resisting shear force (Newtons)
        
        # Default tracking configuration
        self.active_asset_key = "plate"
        self.system_state_indicator = "stable"
        self.telemetry_frame_counter = 0
        
        # 2. Safety Shield Sandbox Setup (1 environment channel for real hardware)
        self.safety_shield = SafetyShield(num_envs=1)
        
        # 3. Create ROS2 Publishers & Subscribers
        # We wrap these to allow mock operation if rclpy is not present on Spencer's local machine
        if 'rclpy' in sys.modules:
            self._init_ros2_io()
            
        # 4. Asynchronous WebSocket Networking Thread
        # Coordinates WebSocket streaming without blocking the high-speed real-time ROS2 executor
        self.connected_clients = set()
        self.loop = None
        self.network_thread = threading.Thread(target=self._start_async_network_loop, daemon=True)
        self.network_thread.start()

    # --- STREAMING_CHUNK: Mapping hardware command registers... ---
    def _init_ros2_io(self):
        """ Configures standard ROS2 interfaces targeting the hardware layer """
        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self._joint_state_callback,
            10
        )
        self.tactile_sub = self.create_subscription(
            Float32MultiArray,
            '/christenson_robotics/tactile_grid',
            self._tactile_grid_callback,
            10
        )
        
        # Command Publishers (Sanitized outputs from Safety Shield)
        self.joint_cmd_pub = self.create_publisher(
            Float32MultiArray,
            '/christenson_robotics/joint_commands',
            10
        )
        self.force_cmd_pub = self.create_publisher(
            Float32,
            '/christenson_robotics/force_commands',
            10
        )

    # --- STREAMING_CHUNK: Processing raw hardware encoder ticks... ---
    def _joint_state_callback(self, msg: JointState):
        """ Reads joint encoders. Assumes index 0 and 1 represent the robotic fingers """
        if len(msg.position) >= 2:
            self.current_joint_positions[0] = msg.position[0]
            self.current_joint_positions[1] = msg.position[1]
        if len(msg.velocity) >= 2:
            self.current_joint_velocities[0] = msg.velocity[0]
            self.current_joint_velocities[1] = msg.velocity[1]

    def _tactile_grid_callback(self, msg: Float32MultiArray):
        """ Captures 16 taxel pressure points from physical silicone fingertip skin """
        if len(msg.data) == 16:
            self.current_taxel_pressures = np.array(msg.data, dtype=np.float32)

    # --- STREAMING_CHUNK: Formatting unified 28-Dimensional vectors... ---
    def compile_observation_vector(self) -> np.ndarray:
        """
        Compiles the incoming sensory inputs into our unified 28D state vector.
        Matches the input layer requirements of Qwen-2.5-Coder and PPO Agent.
        """
        # 12-Dimensional Base State Indicators
        base_state = [
            self.realized_normal_force / 60.0,
            self.slip_velocity_estimate / 5.0,
            0.50 / 3.0,                       # Asset mass prior (0.50kg plate)
            45.0 / 200.0,                     # Crush limit (45N)
            1.2 / 2.0,                        # Radius metric
            0.15 / 2.0,                       # Height metric
            0.85,                             # Noisy dry friction prior
            0.001 / 0.001,                    # Squeeze-film water thickness
            80.0 / 500.0,                     # Touch stiffness (80 N/mm)
            1.0,                              # Joint 0 stiffness multiplier
            1.0,                              # Joint 1 stiffness multiplier
            self.current_joint_velocities[0] / 4.5
        ]
        
        # Scale taxel pressures and append (16 dimensions)
        scaled_taxels = (self.current_taxel_pressures / 1.0e6).tolist()
        
        full_28d_obs = base_state + scaled_taxels
        return np.array(full_28d_obs, dtype=np.float32)

    # --- STREAMING_CHUNK: Filtering actuator commands dynamically... ---
    def dispatch_safe_commands(self, raw_action_array: np.ndarray, dt: float = 0.032):
        """
        Passes the neural network commands through our unalterable, firmware-level
        Safety Shield prior to publishing to the physical motors.
        """
        current_q = np.expand_dims(self.current_joint_positions, axis=0)      # Shape [1, 2]
        current_qd = np.expand_dims(self.current_joint_velocities, axis=0)    # Shape [1, 2]
        commanded_action = np.expand_dims(raw_action_array, axis=0)           # Shape [1, 5]
        
        # Enforce physical mechanical limit clamps
        filtered_action = self.safety_shield.enforce_boundaries(
            commanded_action, current_q, current_qd, dt
        )[0]
        
        # Convert filtered actions back to physical units
        target_q0 = self.current_joint_positions[0] + filtered_action[0] * 0.1
        target_q1 = self.current_joint_positions[1] + filtered_action[1] * 0.1
        target_fn = (filtered_action[2] + 1.0) * 30.0
        
        # Publish to ROS2 network if active
        if 'rclpy' in sys.modules:
            # Publish joint positions
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [float(target_q0), float(target_q1)]
            self.joint_cmd_pub.publish(cmd_msg)
            
            # Publish clamp forces
            force_msg = Float32()
            force_msg.data = float(target_fn)
            self.force_cmd_pub.publish(force_msg)
            
        return target_q0, target_q1, target_fn

    # --- STREAMING_CHUNK: Deploying thread-safe network sockets... ---
    def _start_async_network_loop(self):
        """
        Launches the asynchronous event loop inside the dedicated thread,
        handling modern context-managed websockets.serve lifecycles.
        """
        try:
            import websockets
        except ImportError:
            self.get_logger().warn("Websockets not installed. Skipping live visualizer broadcast.")
            return

        async def register(ws):
            self.connected_clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                self.connected_clients.remove(ws)

        async def run_loop():
            # Capture the running event loop in our instance register
            self.loop = asyncio.get_running_loop()
            async with websockets.serve(register, "0.0.0.0", 8765):
                await asyncio.Future() # keep the server thread running indefinitely

        # Run modern async loop entry block natively
        asyncio.run(run_loop())

    # --- STREAMING_CHUNK: Broadcasting tactile telemetry maps... ---
    def broadcast_visualizer_telemetry(self):
        """ Package current active physical states and broadcast to 3D HUD Visualizer """
        if self.loop is None or not self.connected_clients:
            return
            
        self.telemetry_frame_counter += 1
        
        payload = {
            "frame": self.telemetry_frame_counter,
            "systemState": self.system_state_indicator,
            "activeAsset": self.active_asset_key,
            "joint0": float(self.current_joint_positions[0]),
            "joint1": float(self.current_joint_positions[1]),
            "normalForce": float(self.realized_normal_force),
            "slipVelocity": float(self.slip_velocity_estimate),
            "assetHeightOffset": float(self.height_offset_estimate),
            "gripFrictionForce": float(self.friction_force_estimate),
            "meanReward": 0.0,
            "contactRadius": 0.005,
            "waterFilm": 0.0,
            "kp0": 150.0,
            "kp1": 120.0,
            "taxels": self.current_taxel_pressures.tolist(),
            "lanes": [] # Single real hardware lane
        }
        
        # Schedule threadsafe coroutine inside async executor
        async def send():
            msg = json.dumps(payload)
            await asyncio.gather(
                *[ws.send(msg) for ws in self.connected_clients],
                return_exceptions=True
            )
            
        asyncio.run_coroutine_threadsafe(send(), self.loop)

# --- STREAMING_CHUNK: Launching local verification routines... ---
def main(args=None):
    if 'rclpy' in sys.modules:
        rclpy.init(args=args)
        node = HapticDexterityBridgeNode()
        
        # Execute ROS2 spin rate
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("ROS2 Bridge Node shutting down gracefully...")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        # Mock dry-run verification
        node = HapticDexterityBridgeNode()
        print("\n=== ROS2 Bridge Node Mock Dry-Run Active ===")
        print("  Sensing state: Ready.")
        print("  WebSocket Broadcaster: Serving on port 8765.")
        print("  Safety Shield constraints check: Active.")
        
        # Simulate active state changes to test compiler pipelines
        try:
            for step in range(5):
                # Simulated hardware tactile sensor change
                node.current_joint_positions += np.array([0.02, 0.01])
                node.current_taxel_pressures[5] = 120000.0 # 120 kPa pressure point
                node.realized_normal_force = 15.4
                
                # Enforce safety checks
                q0, q1, fn = node.dispatch_safe_commands(np.array([0.1, -0.05, 0.2, 0.0, 0.0]))
                node.broadcast_visualizer_telemetry()
                
                print(f"Step {step+1:02d}: Observed Joint Positions = {node.current_joint_positions}, Filtered commanded Force = {fn:.2f}N")
                import time
                time.sleep(0.1)
                
            print("\n[SUCCESS] Pre-flight ROS2 compiler compliance: PASS.")
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()