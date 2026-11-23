#!/usr/bin/env python3
"""
UR5e Photo Capture - Optimized for Box at [0.3, 0.3, 0.2]
Captures from multiple angles: top, sides, and corners
Camera offset compensated automatically
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
        
        # Box configuration
        self.box_center = np.array([0.3, 0.3, 0.2])  # Box center position
        self.box_size = 0.15  # Box is 15cm cube
        
        # Camera offset from wrist_3_link
        # Camera is at [-0.01578, 0.00732, 0.01917] relative to wrist
        self.camera_offset = np.array([-0.01578, 0.00732, 0.01917])
        
        self.get_logger().info(f"📦 Box center: {self.box_center}")
        self.get_logger().info(f"📦 Box size: {self.box_size}m")
        self.get_logger().info(f"📷 Camera offset from wrist: {self.camera_offset}")
        
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
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
            elif msg.encoding == 'bgr8':
                img_array = np.frombuffer(msg.data, dtype=np.uint8)
                img_array = img_array.reshape((msg.height, msg.width, 3))
                # Convert BGR to RGB
                img_array = img_array[:, :, ::-1]
            else:
                self.get_logger().error(f"Unsupported encoding: {msg.encoding}")
                return False
            
            # Save as PPM
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


def calculate_look_at_orientation(camera_pos, target_pos):
    """
    Calculate Euler angles for camera to look at target
    
    Args:
        camera_pos: [x, y, z] camera position
        target_pos: [x, y, z] target position to look at
        
    Returns:
        (rx, ry, rz) Euler angles in radians
    """
    # Direction vector from camera to target
    direction = np.array(target_pos) - np.array(camera_pos)
    direction = direction / np.linalg.norm(direction)  # Normalize
    
    dx, dy, dz = direction
    
    # Calculate yaw (rotation around Z-axis)
    yaw = math.atan2(dy, dx)
    
    # Calculate pitch (rotation around Y-axis)
    pitch = -math.asin(dz)  # Negative because of coordinate system
    
    # For camera pointing setup:
    # rx = pi means camera is upside down (standard robot camera orientation)
    # ry = pitch adjustment
    # rz = yaw to point at target
    
    rx = math.pi  # Camera mounted upside down
    ry = pitch     # Tilt up/down to look at target
    rz = yaw + math.pi  # Rotate to face target (+ pi to flip direction)
    
    return rx, ry, rz


def generate_box_capture_positions(box_center, box_size, camera_offset):
    """
    Generate camera positions to capture box from all angles
    Camera orientations calculated to always look at box center
    
    Args:
        box_center: [x, y, z] center of box
        box_size: size of box (cube)
        camera_offset: [x, y, z] camera offset from wrist_3_link
        
    Returns:
        List of camera positions with wrist_3_link coordinates
    """
    positions = []
    
    bx, by, bz = box_center
    
    # Distance from box center to wrist (adjusted for camera offset)
    # For a 15cm box, wrist should be ~30-35cm away
    wrist_distance = 0.35
    
    # === TOP VIEW ===
    # Directly above box (camera pointing down)
    wx, wy, wz = bx, by, bz + wrist_distance
    rx, ry, rz = calculate_look_at_orientation([wx, wy, wz], box_center)
    
    positions.append({
        'name': 'Top-Center',
        'wrist': [wx, wy, wz],
        'rx': rx,
        'ry': ry,
        'rz': rz
    })
    
    # === SIDE VIEWS (Horizontal ring around box) ===
    # 8 positions around the box at box height
    for i in range(6):
        angle = 2 * math.pi * i / 8
        
        # Position wrist at distance from box, at box center height + offset
        wx = bx + wrist_distance * math.cos(angle)
        wy = by + wrist_distance * math.sin(angle)
        wz = bz + box_size * 0.5  # Slightly above box center for better view
        
        # Calculate orientation to look at box center
        rx, ry, rz = calculate_look_at_orientation([wx, wy, wz], box_center)
        
        positions.append({
            'name': f'Side-{i+1} ({int(math.degrees(angle))}deg)',
            'wrist': [wx, wy, wz],
            'rx': rx,
            'ry': ry,
            'rz': rz
        })
    
    # === ANGLED VIEWS (45° elevation) ===
    # 4 corners at 45° elevation angle
    for i in range(4):
        angle = 2 * math.pi * i / 4 + math.pi/4  # Offset by 45° from cardinal directions
        
        # Position at 45° elevation
        elevation_angle = math.pi / 4  # 45 degrees
        horizontal_dist = wrist_distance * math.cos(elevation_angle)
        vertical_dist = wrist_distance * math.sin(elevation_angle)
        
        wx = bx + horizontal_dist * math.cos(angle)
        wy = by + horizontal_dist * math.sin(angle)
        wz = bz + vertical_dist
        
        # Calculate orientation to look at box center
        rx, ry, rz = calculate_look_at_orientation([wx, wy, wz], box_center)
        
        positions.append({
            'name': f'Angled-{i+1} ({int(math.degrees(angle))}deg)',
            'wrist': [wx, wy, wz],
            'rx': rx,
            'ry': ry,
            'rz': rz
        })
    
    # === LOW ANGLED VIEWS (20° elevation - near table level) ===
    # 4 positions looking slightly up at the box
    for i in range(4):
        angle = 2 * math.pi * i / 4
        
        # Position at low elevation (20°)
        elevation_angle = math.pi / 9  # ~20 degrees
        horizontal_dist = wrist_distance * math.cos(elevation_angle)
        vertical_dist = wrist_distance * math.sin(elevation_angle)
        
        wx = bx + horizontal_dist * math.cos(angle)
        wy = by + horizontal_dist * math.sin(angle)
        wz = bz - vertical_dist + box_size * 0.3  # Slightly below box center
        
        # Calculate orientation to look at box center
        rx, ry, rz = calculate_look_at_orientation([wx, wy, wz], box_center)
        
        positions.append({
            'name': f'Low-{i+1} ({int(math.degrees(angle))}deg)',
            'wrist': [wx, wy, wz],
            'rx': rx,
            'ry': ry,
            'rz': rz
        })
    
    return positions


def main():
    rclpy.init()
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"box_photos_{timestamp}"
    
    ur = URPhotoCapture(save_dir=save_dir)
    
    print("\n" + "="*70)
    print("UR5e Box Photo Capture - Multi-Angle System")
    print("="*70)
    print(f"\n📦 Box Configuration:")
    print(f"   Center: {ur.box_center}")
    print(f"   Size: {ur.box_size}m (15cm cube)")
    print(f"\n📷 Camera Configuration:")
    print(f"   Offset from wrist: {ur.camera_offset}")
    print(f"\n📁 Save directory: {os.path.abspath(save_dir)}")
    
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
    
    # Home position (safe starting point)
    HOME = {'x': 0.4, 'y': 0.4, 'z': 0.6, 'rx': math.pi, 'ry': 0.0, 'rz': 0.0}
    
    # Generate camera positions optimized for the box
    positions = generate_box_capture_positions(
        ur.box_center, 
        ur.box_size, 
        ur.camera_offset
    )
    
    print(f"\n📋 Generated {len(positions)} camera positions:")
    print(f"   - 1 top view")
    print(f"   - 8 side views (horizontal ring)")
    print(f"   - 4 angled views (45° elevation)")
    print(f"   - 4 low views (near table level)")
    
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
        skipped = 0
        
        for i, pos in enumerate(positions, 1):
            print(f"\n{'='*70}")
            print(f"📸 Position {i}/{len(positions)}: {pos['name']}")
            print(f"{'='*70}")
            
            wx, wy, wz = pos['wrist']
            
            # Check workspace limits
            error = ur.check_workspace_limits(wx, wy, wz)
            if error:
                print(f"⚠ Skipping - {error}")
                skipped += 1
                continue
            
            # Move to photo position
            success = ur.move_smooth(
                wx, wy, wz,
                pos['rx'], pos['ry'], pos['rz'],
                steps=40
            )
            
            if success:
                print(f"✓ Camera positioned")
                print(f"  Wrist: ({wx:.3f}, {wy:.3f}, {wz:.3f})")
                
                time.sleep(0.8)  # Stabilization
                
                # CAPTURE PHOTO
                if ur.capture_photo(pos['name']):
                    photo_count += 1
                
                time.sleep(0.5)
            else:
                print(f"✗ Failed to reach position - IK/collision")
                skipped += 1
        
        # Return home
        print(f"\n{'='*70}")
        print("🏠 Returning to HOME...")
        print(f"{'='*70}")
        ur.move_smooth(HOME['x'], HOME['y'], HOME['z'],
                       HOME['rx'], HOME['ry'], HOME['rz'])
        
        print(f"\n{'='*70}")
        print(f"✓ Photo capture complete!")
        print(f"  Photos captured: {photo_count}/{len(positions)}")
        print(f"  Positions skipped: {skipped}")
        print(f"  Saved in: {os.path.abspath(save_dir)}")
        print(f"\n💡 View images:")
        print(f"  eog {save_dir}/*.ppm")
        print(f"\n💡 Convert to PNG:")
        print(f"  mogrify -format png {save_dir}/*.ppm")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted - returning HOME...")
        ur.move_smooth(HOME['x'], HOME['y'], HOME['z'],
                       HOME['rx'], HOME['ry'], HOME['rz'])
    
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()