# we need 3 terminals
# deactivate conda for all of them
# one for isaacsim (see anleitung there)
# one for ros node , open a terminal and paste: ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e
# open a third terminal and run: python3 /home/AP_PathMatters/path_matters/Isaacsim/scripts/17_12_ur_move.py

# STEPS:

# 1st terminal:
# conda deactivate
# cd isaacsim/_build/linux-x86_64/release
# ./isaac-sim.sh 
# open : 17_12_robot_plane_graph.usd


# 2nd terminal:
# conda deactivate
# ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e

# 3rd terminal:
# conda deactivate
# python3 /home/AP_PathMatters/path_matters/Isaacsim/scripts/trajectory/26_01_ur_move.py
#!/usr/bin/env python3
"""
UR5e Photo Capture - CORRECTED Version
Uses "world" frame (which the diagnostic confirmed works!)
"""

import rclpy
from rclpy.node import Node
import time, math
import numpy as np
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Image
import os
from datetime import datetime


class URPhotoCapture(Node):
    def __init__(self, save_dir="captured_images"):
        super().__init__('ur_photo_capture')
        
        # Image capture setup
        self.latest_image = None
        self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"✓ Images will be saved to: {os.path.abspath(self.save_dir)}")
        
        # Subscribe to camera
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )
        
        # IK service
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        
        # Joint command publisher
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        
        # Current joint state
        self.current_joints = None
        
        # UR5e joint names
        self.joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # Wait for IK service
        self.get_logger().info("Waiting for MoveIt IK service...")
        if not self.ik.wait_for_service(timeout_sec=10):
            self.get_logger().error("IK service not available!")
            raise RuntimeError("MoveIt not started")
        
        self.get_logger().info("✓ MoveIt IK service ready!")
    
    def image_callback(self, msg):
        self.latest_image = msg
    
    def capture_photo(self, position_name):
        """Save current camera image"""
        if self.latest_image is None:
            self.get_logger().warn("⚠ No camera image")
            return False
        
        msg = self.latest_image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = position_name.replace(' ', '_').replace('/', '-')
        filename = f"{timestamp}_{safe_name}.ppm"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            if msg.encoding == 'rgb8':
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
                img_array = img_array[:, :, ::-1]
            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
                return False
            
            with open(filepath, 'wb') as f:
                f.write(f"P6\n{msg.width} {msg.height}\n255\n".encode())
                f.write(img_array.tobytes())
            
            self.get_logger().info(f"  📷 SAVED: {filename}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"  ✗ Save failed: {e}")
            return False
    
    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])
    
    @staticmethod
    def euler_to_quat(rx, ry, rz):
        """Convert Euler angles to quaternion"""
        cy, sy = math.cos(rz*0.5), math.sin(rz*0.5)
        cp, sp = math.cos(ry*0.5), math.sin(ry*0.5)
        cr, sr = math.cos(rx*0.5), math.sin(rx*0.5)
        
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        
        return x, y, z, w
    
    def compute_ik(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0):
        """Compute IK - USING "world" FRAME (diagnostic confirmed this works!)"""
        pose = PoseStamped()
        pose.header.frame_id = "world"  # CORRECT - diagnostic confirmed!
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        qx, qy, qz, qw = self.euler_to_quat(rx, ry, rz)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 5
        req.ik_request.avoid_collisions = False  # Disable for testing
        
        # Add current joint state as seed
        if self.current_joints:
            req.ik_request.robot_state.joint_state.name = self.joints
            req.ik_request.robot_state.joint_state.position = self.current_joints

        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=6)
        
        try:
            response = future.result()
            if response.error_code.val == 1:  # SUCCESS
                return list(response.solution.joint_state.position[:6])
            else:
                self.get_logger().error(f"IK failed with code: {response.error_code.val}")
                return None
        except Exception as e:
            self.get_logger().error(f"IK error: {e}")
            return None
    
    def interpolate_joints(self, start_joints, end_joints, steps=20):
        """Smooth interpolation"""
        trajectory = []
        for i in range(steps + 1):
            t = i / steps
            smooth_t = 3*t**2 - 2*t**3
            
            interpolated = [
                start + smooth_t * (end - start)
                for start, end in zip(start_joints, end_joints)
            ]
            trajectory.append(interpolated)
        
        return trajectory
    
    def execute_trajectory(self, trajectory, duration_per_step=0.05):
        """Execute trajectory"""
        for joints in trajectory:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joints
            msg.position = joints
            
            self.pub.publish(msg)
            time.sleep(duration_per_step)
    
    def move_smooth(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0, steps=30):
        """Move with smooth trajectory"""
        self.get_logger().info(f"→ Moving to ({x:.3f}, {y:.3f}, {z:.3f})")
        
        target_joints = self.compute_ik(x, y, z, rx, ry, rz)
        
        if target_joints is None:
            self.get_logger().error("✗ IK failed - position unreachable")
            return False
        
        if self.current_joints is None:
            # First move
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joints
            msg.position = target_joints
            self.pub.publish(msg)
            time.sleep(2)
            self.current_joints = target_joints
        else:
            trajectory = self.interpolate_joints(self.current_joints, target_joints, steps)
            self.execute_trajectory(trajectory, duration_per_step=0.05)
            self.current_joints = target_joints
        
        self.get_logger().info("✓ Movement complete")
        return True


def main():
    rclpy.init()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"/home/AP_PathMatters/path_matters/trajectory/captures/captured_photos_{timestamp}"
    
    ur = URPhotoCapture(save_dir=save_dir)
    
    print("\n" + "="*70)
    print("UR5e Photo Capture - CORRECTED (using 'world' frame)")
    print("="*70)
    print(f"\n📁 Save location: {os.path.abspath(save_dir)}")
    
    # Wait for camera
    print("\n⏳ Waiting for camera...")
    wait_time = 0
    while ur.latest_image is None and wait_time < 10:
        rclpy.spin_once(ur, timeout_sec=0.5)
        wait_time += 0.5
    
    if ur.latest_image is None:
        print("⚠ No camera images - continuing anyway")
    else:
        print(f"✓ Camera ready ({ur.latest_image.width}x{ur.latest_image.height})")
    
    # TESTED POSITION from diagnostic (we know this works!)
    SAFE_TEST = {'x': 0.3, 'y': 0.0, 'z': 0.4, 'rx': math.pi, 'ry': 0.0, 'rz': 0.0}
    
    # HOME position - slightly higher and safer
    HOME = {'x': 0.3, 'y': 0.0, 'z': 0.5, 'rx': math.pi, 'ry': 0.0, 'rz': 0.0}
    
    # Camera positions - CONSERVATIVE positions around object at (0.3, 0.0, 0.2)
    # Using smaller radius to ensure reachability
    positions = [
        {'name': 'Top View', 'x': 0.3, 'y': 0.0, 'z': 0.6},
        {'name': 'Front', 'x': 0.15, 'y': 0.0, 'z': 0.3},
        {'name': 'Right', 'x': 0.3, 'y': 0.2, 'z': 0.3},
        {'name': 'Back', 'x': 0.5, 'y': 0.0, 'z': 0.3},
        {'name': 'Left', 'x': 0.3, 'y': -0.2, 'z': 0.3},
        {'name': 'Front-Angled', 'x': 0.2, 'y': 0.0, 'z': 0.45},
        {'name': 'Right-Angled', 'x': 0.3, 'y': 0.15, 'z': 0.45},
    ]
    
    try:
        # First, test with the known-good position from diagnostic
        print(f"\n{'='*70}")
        print("🧪 Testing with diagnostic-confirmed position first...")
        print(f"{'='*70}")
        
        if not ur.move_smooth(SAFE_TEST['x'], SAFE_TEST['y'], SAFE_TEST['z'], 
                              SAFE_TEST['rx'], SAFE_TEST['ry'], SAFE_TEST['rz']):
            print("✗ Even the test position failed! Something is very wrong.")
            print("💡 Use ur_photo_simple.py instead (no IK dependency)")
            return
        
        print("✓ Test position works! Continuing with photo sequence...")
        time.sleep(1)
        
        # Move to home
        print(f"\n{'='*70}")
        print("🏠 Moving to HOME...")
        print(f"{'='*70}")
        
        ur.move_smooth(HOME['x'], HOME['y'], HOME['z'], 
                      HOME['rx'], HOME['ry'], HOME['rz'])
        time.sleep(1)
        
        # Capture sequence
        photo_count = 0
        
        for i, pos in enumerate(positions, 1):
            print(f"\n{'='*70}")
            print(f"📸 Position {i}/{len(positions)}: {pos['name']}")
            print(f"{'='*70}")
            
            success = ur.move_smooth(
                pos['x'], pos['y'], pos['z'],
                pos.get('rx', math.pi), 
                pos.get('ry', 0.0), 
                pos.get('rz', 0.0),
                steps=40
            )
            
            if success:
                print(f"✓ Position reached")
                time.sleep(0.8)
                
                if ur.capture_photo(pos['name']):
                    photo_count += 1
                
                time.sleep(0.5)
            else:
                print(f"⚠ Skipped - IK failed for this position")
        
        # Return home
        print(f"\n{'='*70}")
        print("🏠 Returning HOME...")
        print(f"{'='*70}")
        ur.move_smooth(HOME['x'], HOME['y'], HOME['z'],
                       HOME['rx'], HOME['ry'], HOME['rz'])
        
        print(f"\n{'='*70}")
        print(f"✓ Complete!")
        print(f"  Photos: {photo_count}/{len(positions)}")
        print(f"  Location: {os.path.abspath(save_dir)}")
        if photo_count > 0:
            print(f"\n💡 View: eog {save_dir}/*.ppm")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()