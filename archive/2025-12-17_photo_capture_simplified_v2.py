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
        self.latest_image = msg
    
    def capture_photo(self, position_name):
        if self.latest_image is None:
            self.get_logger().warn("⚠ No camera image available yet")
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
            self.get_logger().error(f"  ✗ Failed to save image: {e}")
            return False
    
    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])
    
    @staticmethod
    def euler_to_quat(rx, ry, rz):
        cy, sy = math.cos(rz*0.5), math.sin(rz*0.5)
        cp, sp = math.cos(ry*0.5), math.sin(ry*0.5)
        cr, sr = math.cos(rx*0.5), math.sin(rx*0.5)
        
        w = cr*cp*cy + sr*sp*sy
        x = sr*cp*cy - cr*sp*sy
        y = cr*sp*cy + sr*cp*sy
        z = cr*cp*sy - sr*sp*cy
        
        return x, y, z, w
    
    def compute_ik(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0):
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
            return None
        except Exception as e:
            self.get_logger().error(f"IK error: {e}")
            return None
    
    def interpolate_joints(self, start_joints, end_joints, steps=20):
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
        for joints in trajectory:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joints
            msg.position = joints
            self.pub.publish(msg)
            time.sleep(duration_per_step)
    
    def move_smooth(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0, steps=30):
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
        self.execute_trajectory(trajectory)
        self.current_joints = target_joints
        
        self.get_logger().info("✓ Movement complete")
        return True
    
    @staticmethod
    def check_workspace_limits(x, y, z):
        if z < 0.15:
            return "TOO LOW"
        if not (-0.05 <= x <= 0.85):
            return "X out of range"
        if not (-0.85 <= y <= 0.85):
            return "Y out of range"
        if z > 1.2:
            return "TOO HIGH"
        return None


def main():
    rclpy.init()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"captured_photos_{timestamp}"
    
    ur = URPhotoCapture(save_dir=save_dir)
    
    # -----------------------------
    # BOX DEFINITION (EDIT 1)
    # -----------------------------
    BOX_CENTER = np.array([0.3, 0.3, 0.2])
    BOX_SIZE   = 0.15
    BOX_HALF   = BOX_SIZE / 2.0
    
    CAMERA_DISTANCE = 0.30
    TOP_CLEARANCE   = 0.25
    SIDE_CLEARANCE  = 0.10
    
    HOME = {'x': 0.4, 'y': 0.4, 'z': 0.6, 'rx': math.pi}
    
    # -----------------------------
    # CAMERA POSITIONS (EDIT 2)
    # -----------------------------
    positions = [
        {'name': 'Top',
         'x': BOX_CENTER[0],
         'y': BOX_CENTER[1],
         'z': BOX_CENTER[2] + BOX_HALF + TOP_CLEARANCE},

        {'name': 'Front',
         'x': BOX_CENTER[0] - CAMERA_DISTANCE,
         'y': BOX_CENTER[1],
         'z': BOX_CENTER[2] + BOX_HALF + SIDE_CLEARANCE},

        {'name': 'Back',
         'x': BOX_CENTER[0] + CAMERA_DISTANCE,
         'y': BOX_CENTER[1],
         'z': BOX_CENTER[2] + BOX_HALF + SIDE_CLEARANCE},

        {'name': 'Left',
         'x': BOX_CENTER[0],
         'y': BOX_CENTER[1] - CAMERA_DISTANCE,
         'z': BOX_CENTER[2] + BOX_HALF + SIDE_CLEARANCE},

        {'name': 'Right',
         'x': BOX_CENTER[0],
         'y': BOX_CENTER[1] + CAMERA_DISTANCE,
         'z': BOX_CENTER[2] + BOX_HALF + SIDE_CLEARANCE},
    ]
    
    ur.move_smooth(**HOME)
    time.sleep(1)
    
    photo_count = 0
    for pos in positions:
        if ur.move_smooth(pos['x'], pos['y'], pos['z']):
            time.sleep(0.8)
            if ur.capture_photo(pos['name']):
                photo_count += 1
    
    ur.move_smooth(**HOME)
    
    print(f"\n✓ Photo capture complete: {photo_count}/{len(positions)} images")
    print(f"Saved in: {os.path.abspath(save_dir)}")
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()
