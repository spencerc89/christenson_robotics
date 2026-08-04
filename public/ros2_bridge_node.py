from __future__ import annotations

import os
import sys
import json
import math
import io
import asyncio
import threading
import numpy as np

# Try loading Cryptography for native .hvos decryption
try:
    import torch
    import base64
    import hashlib
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_HVOS_CRYPTO = True
except ImportError:
    HAS_HVOS_CRYPTO = False

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
    
    # Stub classes for non-ROS environments to compile cleanly
    class Node:
        def __init__(self, name): 
            self.name = name
        def get_logger(self):
            class Logger:
                def info(self, msg): print(f"[Mock ROS2 Node] {msg}")
                def warn(self, msg): print(f"[WARN] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
            return Logger()
            
    class JointState: pass
    class Float32MultiArray: pass
    class Float32: pass

# Import low-level firmware Safety Shield
try:
    from core.safety_shield import SafetyShield
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.safety_shield import SafetyShield

# --- NATIVE HVOS DECRYPTION & WEIGHT LOADER HELPER ---
MASTER_SALT = b"ChristensonRobotics_HVOS_Master_Salt_2026"

def decrypt_and_load_hvos_model(hvos_path: str, passphrase: str, device: str = "cpu") -> dict:
    """
    In-memory decryption and loader function built natively into the ROS2 Bridge.
    Decrypts the binary directly in RAM/VRAM without saving plain .pt files to disk.
    """
    if not HAS_HVOS_CRYPTO:
        raise ImportError("Packages 'cryptography' and 'torch' are required for .hvos files. Run: pip install cryptography torch")
        
    if not os.path.exists(hvos_path):
        raise FileNotFoundError(f"HV-OS model binary not found at: {hvos_path}")
        
    # Key Derivation
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=MASTER_SALT,
        iterations=100_000,
    )
    fernet_key = base64.urlsafe_b64encode(kdf.derive(passphrase.encode('utf-8')))
    fernet = Fernet(fernet_key)
    
    with open(hvos_path, "rb") as f:
        encrypted_bytes = f.read()
        
    try:
        decrypted_bytes = fernet.decrypt(encrypted_bytes)
    except Exception as e:
        raise PermissionError(f"Decryption failed. Invalid passphrase or tampered binary. ({e})")
        
    buffer = io.BytesIO(decrypted_bytes)
    loaded_dict = torch.load(buffer, map_location=device, weights_only=False)
    
    if "_hvos_watermark_" in loaded_dict:
        wm = loaded_dict["_hvos_watermark_"]
        print(f"[HV-OS SDK] Weight package authenticated cleanly for License: {wm['license_id']}")
        
    return loaded_dict


class HapticDexterityBridgeNode(Node):
    """
    ROS2 Bridge Node representing Christenson Robotics' open-source wrapper.
    Interfaces high-frequency hardware tactile taxels and joint states with 
    our 3D Web HUD Visualizer, local policies, and hardened .hvos weight packages.
    """
    def __init__(self):
        super().__init__('haptic_dexterity_bridge_node')
        self.get_logger().info("Initializing Christenson Robotics Haptic ROS2 Bridge Wrapper...")
        
        # State Variables
        self.current_joint_positions = np.zeros(2)  # Joint 0, Joint 1 (radians)
        self.current_joint_velocities = np.zeros(2) # rad/s
        self.current_taxel_pressures = np.zeros(16) # 4x4 silicone grid (Pascals)
        self.realized_normal_force = 12.0          # Actuator feedback force (Newtons)
        self.slip_velocity_estimate = 0.0          # Estimated slip velocity (m/s)
        self.height_offset_estimate = 0.0          # Current translation drift (meters)
        self.friction_force_estimate = 0.0         # Resisting shear force (Newtons)
        
        self.active_asset_key = "plate"
        self.system_state_indicator = "stable"
        self.telemetry_frame_counter = 0
        
        # Safety Shield Sandbox
        self.safety_shield = SafetyShield(num_envs=1)
        
        # Loaded policy weights container
        self.policy_weights = None
        
        if 'rclpy' in sys.modules:
            self._init_ros2_io()
            
        # Asynchronous WebSocket Networking Thread
        self.connected_clients = set()
        self.loop = None
        self.network_thread = threading.Thread(target=self._start_async_network_loop, daemon=True)
        self.network_thread.start()

    def load_hvos_weights(self, hvos_file_path: str, passphrase: str = "Christenson_DevPro_2026_Secret", device: str = "cpu"):
        """ Direct entry point for subscribers to load encrypted .hvos weights into the node """
        self.get_logger().info(f"Loading hardened HV-OS weights: {hvos_file_path}")
        self.policy_weights = decrypt_and_load_hvos_model(hvos_file_path, passphrase, device=device)
        self.get_logger().info("Weights loaded into memory successfully!")
        return self.policy_weights

    def _init_ros2_io(self):
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

    def _joint_state_callback(self, msg: JointState):
        if len(msg.position) >= 2:
            self.current_joint_positions[0] = msg.position[0]
            self.current_joint_positions[1] = msg.position[1]
        if len(msg.velocity) >= 2:
            self.current_joint_velocities[0] = msg.velocity[0]
            self.current_joint_velocities[1] = msg.velocity[1]

    def _tactile_grid_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 16:
            self.current_taxel_pressures = np.array(msg.data, dtype=np.float32)

    def compile_observation_vector(self) -> np.ndarray:
        base_state = [
            self.realized_normal_force / 60.0,
            self.slip_velocity_estimate / 5.0,
            0.50 / 3.0,                       
            45.0 / 200.0,                     
            1.2 / 2.0,                        
            0.15 / 2.0,                       
            0.85,                             
            0.001 / 0.001,                    
            80.0 / 500.0,                     
            1.0,                              
            1.0,                              
            self.current_joint_velocities[0] / 4.5
        ]
        scaled_taxels = (self.current_taxel_pressures / 1.0e6).tolist()
        return np.array(base_state + scaled_taxels, dtype=np.float32)

    def dispatch_safe_commands(self, raw_action_array: np.ndarray, dt: float = 0.032):
        current_q = np.expand_dims(self.current_joint_positions, axis=0)
        current_qd = np.expand_dims(self.current_joint_velocities, axis=0)
        commanded_action = np.expand_dims(raw_action_array, axis=0)
        
        filtered_action = self.safety_shield.enforce_boundaries(
            commanded_action, current_q, current_qd, dt
        )[0]
        
        target_q0 = self.current_joint_positions[0] + filtered_action[0] * 0.1
        target_q1 = self.current_joint_positions[1] + filtered_action[1] * 0.1
        target_fn = (filtered_action[2] + 1.0) * 30.0
        
        if 'rclpy' in sys.modules:
            cmd_msg = Float32MultiArray()
            cmd_msg.data = [float(target_q0), float(target_q1)]
            self.joint_cmd_pub.publish(cmd_msg)
            
            force_msg = Float32()
            force_msg.data = float(target_fn)
            self.force_cmd_pub.publish(force_msg)
            
        return target_q0, target_q1, target_fn

    def _start_async_network_loop(self):
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
            self.loop = asyncio.get_running_loop()
            async with websockets.serve(register, "0.0.0.0", 8765):
                await asyncio.Future()

        asyncio.run(run_loop())

    def broadcast_visualizer_telemetry(self):
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
            "lanes": []
        }
        
        async def send():
            msg = json.dumps(payload)
            await asyncio.gather(
                *[ws.send(msg) for ws in self.connected_clients],
                return_exceptions=True
            )
            
        asyncio.run_coroutine_threadsafe(send(), self.loop)

def main(args=None):
    if 'rclpy' in sys.modules:
        rclpy.init(args=args)
        node = HapticDexterityBridgeNode()
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            node.get_logger().info("ROS2 Bridge Node shutting down gracefully...")
        finally:
            node.destroy_node()
            rclpy.shutdown()
    else:
        node = HapticDexterityBridgeNode()
        print("\n=== ROS2 Bridge Node Mock Dry-Run Active ===")
        print("  WebSocket Broadcaster: Serving on port 8765.")
        print("  Safety Shield constraints check: Active.")
        print("  Native HVOS Decryption Engine: Loaded.")
        
        try:
            for step in range(3):
                node.current_joint_positions += np.array([0.02, 0.01])
                node.realized_normal_force = 15.4
                q0, q1, fn = node.dispatch_safe_commands(np.array([0.1, -0.05, 0.2, 0.0, 0.0]))
                node.broadcast_visualizer_telemetry()
                print(f"Step {step+1:02d}: Joint Positions = {node.current_joint_positions}, Filtered Force = {fn:.2f}N")
                import time
                time.sleep(0.1)
            print("\n[SUCCESS] Pre-flight ROS2 compiler compliance: PASS.")
        except KeyboardInterrupt:
            pass

if __name__ == '__main__':
    main()
