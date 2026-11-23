#!/usr/bin/env python3
"""
UR5e Photo Capture - Simplified Version
No cv_bridge dependency - saves images directly from ROS messages
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
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"✓ Images will be saved to: {os.path.abspath(self.save_dir)}")
        
        # Subscribe to camera topic
        self.camera_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
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
        
        # Subscribe to current joint states
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # Wait for IK service
        while not self.ik.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for /compute_ik service...")
        
        self.get_logger().info("✓ MoveIt IK service ready!")
    
    def image_callback(self, msg):
        """Store latest camera image as ROS message"""
        self.latest_image = msg
    
    def capture_photo(self, position_name):
        """
        Save the current camera image directly from ROS message
        
        Args:
            position_name: Name for the saved image file
            
        Returns:
            bool: True if image was captured and saved
        """
        if self.latest_image is None:
            self.get_logger().warn("⚠ No camera image available yet")
            return False
        
        msg = self.latest_image
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = position_name.replace(' ', '_').replace('/', '-')
        filename = f"{timestamp}_{safe_name}.ppm"
        filepath = os.path.join(self.save_dir, filename)
        
        try:
            # Convert ROS Image message to numpy array
            if msg.encoding == 'rgb8':
                # RGB8 format: 3 channels, 8 bits per channel
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                # BGR8 format: 3 channels, 8 bits per channel
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
                # Convert BGR to RGB
                img_array = img_array[:, :, ::-1]
            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
                return False
            
            # Save as PPM (simple format, no dependencies)
            with open(filepath, 'wb') as f:
                f.write(f"P6\n{msg.width} {msg.height}\n255\n".encode())
                f.write(img_array.tobytes())
            
            self.get_logger().info(f"  📷 SAVED: {filename}")
            return True
            
        except Exception as e:
            self.get_logger().error(f"  ✗ Failed to save image: {e}")
            return False
    
    def joint_callback(self, msg):
        """Store current joint positions"""
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])
    
    @staticmethod
    def euler_to_quat(rx, ry, rz):
        """Convert Euler angles (radians) to quaternion"""
        cy, sy = math.cos(rz*0.5), math.sin(rz*0.5)
        cp, sp = math.cos(ry*0.5), math.sin(ry*0.5)
        cr, sr = math.cos(rx*0.5), math.sin(rx*0.5)
        
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        
        return x, y, z, w
    
    def compute_ik(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0):
        """Compute IK for given pose"""
        pose = PoseStamped()
        pose.header.frame_id = "world"
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
        req.ik_request.timeout.sec = 2
        req.ik_request.avoid_collisions = True

        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3)
        
        try:
            response = future.result()
            if response.error_code.val == 1:
                return list(response.solution.joint_state.position[:6])
            else:
                return None
        except Exception as e:
            self.get_logger().error(f"IK error: {e}")
            return None
    
    def interpolate_joints(self, start_joints, end_joints, steps=20):
        """Create smooth interpolation between joint positions"""
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
        """Execute a smooth trajectory"""
        for joints in trajectory:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joints
            msg.position = joints
            
            self.pub.publish(msg)
            time.sleep(duration_per_step)
    
    def move_smooth(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0, steps=30):
        """Move to target with smooth interpolated trajectory"""
        self.get_logger().info(f"→ Moving to ({x:.3f}, {y:.3f}, {z:.3f})")
        
        target_joints = self.compute_ik(x, y, z, rx, ry, rz)
        
        if target_joints is None:
            self.get_logger().error("✗ IK failed")
            return False
        
        if self.current_joints is None:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joints
            msg.position = target_joints
            self.pub.publish(msg)
            time.sleep(2)
            return True
        
        trajectory = self.interpolate_joints(self.current_joints, target_joints, steps)
        self.execute_trajectory(trajectory, duration_per_step=0.05)
        self.current_joints = target_joints
        
        self.get_logger().info("✓ Movement complete")
        return True
    
    @staticmethod
    def check_workspace_limits(x, y, z):
        """Validate workspace limits"""
        MIN_HEIGHT = 0.15
        if z < MIN_HEIGHT:
            return f"TOO LOW! Z={z:.3f}m"
        
        X_MIN, X_MAX = -0.05, 0.85
        Y_MIN, Y_MAX = -0.85, 0.85  
        Z_MAX = 1.2
        
        if not (X_MIN <= x <= X_MAX):
            return f"X out of range"
        if not (Y_MIN <= y <= Y_MAX):
            return f"Y out of range"
        if z > Z_MAX:
            return f"TOO HIGH!"
        
        return None


def main():
    rclpy.init()
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"captured_photos_{timestamp}"
    
    ur = URPhotoCapture(save_dir=save_dir)
    
    print("\n" + "="*70)
    print("UR5e Photo Capture System - Simplified Version")
    print("="*70)
    print(f"\n📁 Images will be saved to: {os.path.abspath(save_dir)}")
    print(f"📷 Camera topic: /camera/image_raw")
    print(f"💾 Format: PPM (viewable with any image viewer)")
    
    # Wait for camera
    print("\n⏳ Waiting for camera images...")
    wait_time = 0
    while ur.latest_image is None and wait_time < 10:
        rclpy.spin_once(ur, timeout_sec=0.5)
        wait_time += 0.5
    
    if ur.latest_image is None:
        print("\n⚠ WARNING: No camera images received!")
        print("   Check: ros2 topic hz /camera/image_raw")
    else:
        print(f"✓ Camera connected! ({ur.latest_image.width}x{ur.latest_image.height})")
    
    # Home position
    HOME = {'x': 0.4, 'y': 0.4, 'z': 0.6, 'rx': math.pi, 'ry': 0.0, 'rz': 0.0}
    
    # Camera positions around cube at (0.3, 0.3, 0.35)
    positions = [
        {'name': 'Top Center', 'x': 0.3, 'y': 0.3, 'z': 0.65},
        {'name': 'Front', 'x': 0.1, 'y': 0.3, 'z': 0.35},
        {'name': 'Right', 'x': 0.3, 'y': 0.6, 'z': 0.35},
        {'name': 'Back', 'x': 0.6, 'y': 0.3, 'z': 0.35},
        {'name': 'Left', 'x': 0.3, 'y': 0.1, 'z': 0.35},
        {'name': 'Front-Angled', 'x': 0.1, 'y': 0.3, 'z': 0.50},
        {'name': 'Right-Angled', 'x': 0.3, 'y': 0.55, 'z': 0.50},
        {'name': 'Back-Angled', 'x': 0.55, 'y': 0.3, 'z': 0.50},
    ]
    
    try:
        # Go home
        print(f"\n{'='*70}")
        print("🏠 Moving to HOME position...")
        print(f"{'='*70}")
        ur.move_smooth(HOME['x'], HOME['y'], HOME['z'], 
                       HOME['rx'], HOME['ry'], HOME['rz'])
        time.sleep(1)
        
        # Photo capture sequence
        photo_count = 0
        for i, pos in enumerate(positions, 1):
            print(f"\n{'='*70}")
            print(f"📸 Position {i}/{len(positions)}: {pos['name']}")
            print(f"{'='*70}")
            
            error = ur.check_workspace_limits(pos['x'], pos['y'], pos['z'])
            if error:
                print(f"⚠ Skipping - {error}")
                continue
            
            success = ur.move_smooth(
                pos['x'], pos['y'], pos['z'],
                pos.get('rx', math.pi), 
                pos.get('ry', 0.0), 
                pos.get('rz', 0.0),
                steps=40
            )
            
            if success:
                print(f"✓ Camera positioned")
                time.sleep(0.8)  # Stabilization
                
                # CAPTURE PHOTO
                if ur.capture_photo(pos['name']):
                    photo_count += 1
                
                time.sleep(0.5)
            else:
                print(f"✗ Failed to reach position")
        
        # Return home
        print(f"\n{'='*70}")
        print("🏠 Returning to HOME...")
        print(f"{'='*70}")
        ur.move_smooth(HOME['x'], HOME['y'], HOME['z'],
                       HOME['rx'], HOME['ry'], HOME['rz'])
        
        print(f"\n{'='*70}")
        print(f"✓ Photo capture complete!")
        print(f"  Total photos: {photo_count}/{len(positions)}")
        print(f"  Saved in: {os.path.abspath(save_dir)}")
        print(f"\n💡 View images with:")
        print(f"  eog {save_dir}/*.ppm")
        print(f"  or convert to PNG: mogrify -format png {save_dir}/*.ppm")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()