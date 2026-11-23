def generate_circular_ring_waypoints(object_center, radius, elevation_deg, n_points, robot):
    """
    Generate evenly spaced, reachable waypoints in a circle around the object at a given elevation.
    Returns a list of dicts with x, y, z, qx, qy, qz, qw.
    """
    waypoints = []
    elevation_rad = math.radians(elevation_deg)
    min_clearance = 0.25
    for i in range(n_points):
        azimuth_deg = (360.0 / n_points) * i
        azimuth_rad = math.radians(azimuth_deg)
        x_offset = radius * math.cos(azimuth_rad) * math.cos(elevation_rad)
        y_offset = radius * math.sin(azimuth_rad) * math.cos(elevation_rad)
        z_offset = radius * math.sin(elevation_rad)
        camera_pos = {
            'x': object_center['x'] + x_offset,
            'y': object_center['y'] + y_offset,
            'z': max(object_center['z'] + z_offset + 0.1, min_clearance)
        }
        qx, qy, qz, qw = URControlWithDetection.look_at_orientation(camera_pos, object_center)
        # Check reachability
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = float(camera_pos['x'])
        pose.pose.position.y = float(camera_pos['y'])
        pose.pose.position.z = float(camera_pos['z'])
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 2
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state.name = robot.joints
        req.ik_request.robot_state.joint_state.position = robot.current_joints
        req.ik_request.robot_state.joint_state.header.stamp = robot.get_clock().now().to_msg()
        future = robot.ik.call_async(req)
        rclpy.spin_until_future_complete(robot, future, timeout_sec=3)
        try:
            response = future.result()
            if response and response.error_code.val == 1:
                waypoints.append({
                    'x': camera_pos['x'],
                    'y': camera_pos['y'],
                    'z': camera_pos['z'],
                    'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw
                })
        except Exception:
            continue
    return waypoints

def execute_ring_capture(robot, waypoints, save_dir=None):
    """
    Move to each waypoint and capture a photo. Only move to valid, reachable positions.
    """
    for i, wp in enumerate(waypoints):
        success = robot.goto_smooth(wp['x'], wp['y'], wp['z'], wp['qx'], wp['qy'], wp['qz'], wp['qw'], steps=50)
        if success and save_dir:
            robot.capture_photo(f"ring_{i:03d}", save_dir)
#!/usr/bin/env python3
"""
UR5e Robot with Automatic Object Detection and Multi-View Capture
OPTIMIZED FOR 3D RECONSTRUCTION: Multi-ring capture with proper camera orientation
"""

import rclpy
from rclpy.node import Node
import time, math, sys, os
import numpy as np
from datetime import datetime
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import CollisionObject, PlanningScene
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import JointState, Image
from shape_msgs.msg import SolidPrimitive
import cv2
from cv_bridge import CvBridge

# Import modular detection system
from modular_object_detection import create_detection_pipeline


class URControlWithDetection(Node):
    def move_to_joint_positions(self, joint_positions, steps=50):
        """Move robot directly to specified joint positions (no IK)."""
        if len(joint_positions) != 6:
            self.get_logger().error("✗ Joint positions must have 6 values.")
            return False
        trajectory = self.interpolate_joints(self.current_joints, joint_positions, steps)
        self.execute_trajectory(trajectory, duration_per_step=0.04)
        self.current_joints = list(joint_positions)
        self.get_logger().info("  ✓ Moved to pre-home joint state")
        return True
    def __init__(self, detection_method='color'):
        super().__init__('ur_control_detection')
        
        # Robot setup
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.current_joints = None
        self.joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        # Camera setup
        self.latest_image = None
        self.camera_topic = '/camera/image_raw'
        self.bridge = CvBridge()
        
        self.camera_sub = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )
        
        # Joint state subscription
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        # Object detection pipeline
        sam_params = {}
        if detection_method == 'sam':
            sam_params = {
                'model_type': 'vit_h',
                'checkpoint_path': None
            }
        
        self.detection_pipeline = create_detection_pipeline(
            method=detection_method,
            camera_height=0.7,
            image_width=1280,
            image_height=720,
            fov_deg=60,
            **sam_params
        )
        
        self.get_logger().info(f"📷 Detection method: {detection_method}")
        self.get_logger().info(f"📷 Camera topic: {self.camera_topic}")
        
        # Planning Scene Publisher (for ground plane)
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        time.sleep(0.5)
        
        # Add ground plane
        self.add_ground_plane()
        
        # Wait for IK service
        while not self.ik.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for IK service...")
        
        # Wait for initial state
        wait_time = 0
        while self.current_joints is None and wait_time < 10:
            rclpy.spin_once(self, timeout_sec=0.5)
            wait_time += 0.5
        
        if self.current_joints:
            self.get_logger().info("✓ Robot ready!")
        
        time.sleep(1.0)
    
    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])
    
    def image_callback(self, msg):
        self.latest_image = msg
    
    def get_latest_image_as_cv(self):
        if self.latest_image is None:
            return None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, "rgb8")
            return cv_image
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return None
    
    def detect_object(self, visualize=False):
        cv_image = self.get_latest_image_as_cv()
        if cv_image is None:
            self.get_logger().warn("No camera image available")
            return {'detected': False}
        
        results = self.detection_pipeline.detect_and_measure(cv_image, visualize=visualize)
        
        if isinstance(results, tuple):
            results, vis_image = results
            if visualize:
                cv2.imwrite('detection_result.jpg', cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return results
    
    def capture_photo(self, position_name, save_dir="captured_photos"):
        if self.latest_image is None:
            return False
        
        os.makedirs(save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = position_name.replace(' ', '_')
        filename = f"{timestamp}_{safe_name}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        try:
            cv_image = self.get_latest_image_as_cv()
            cv2.imwrite(filepath, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f"  📷 Photo saved: {filename}")
            return True
        except Exception as e:
            self.get_logger().error(f"  ✗ Failed to save: {e}")
            return False
    
    def add_ground_plane(self):
        """Add ground plane to planning scene"""
        self.get_logger().info("Adding ground plane to planning scene...")
        
        scene_msg = PlanningScene()
        scene_msg.is_diff = True
        
        ground = CollisionObject()
        ground.header.frame_id = "world"  # Isaac Sim frame
        ground.header.stamp = self.get_clock().now().to_msg()
        ground.id = "ground_plane"
        
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [3.0, 3.0, 0.01]
        
        ground_pose = Pose()
        ground_pose.position.x = 0.0
        ground_pose.position.y = 0.0
        ground_pose.position.z = -0.005
        ground_pose.orientation.w = 1.0
        
        ground.primitives.append(box)
        ground.primitive_poses.append(ground_pose)
        ground.operation = CollisionObject.ADD
        
        scene_msg.world.collision_objects.append(ground)
        
        for i in range(10):
            self.scene_pub.publish(scene_msg)
            time.sleep(0.1)
        
        self.get_logger().info("✓ Ground plane published to planning scene")
        time.sleep(0.5)
    
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
    
    @staticmethod
    def look_at_orientation(camera_pos, target_pos):
        """
        Calculate quaternion for camera to look at target
        Camera points along -Z axis (standard camera convention)
        """
        # Direction from camera to target
        dx = target_pos['x'] - camera_pos['x']
        dy = target_pos['y'] - camera_pos['y']
        dz = target_pos['z'] - camera_pos['z']
        
        # Horizontal angle (azimuth)
        yaw = math.atan2(dy, dx)
        
        # Vertical angle (elevation)
        horizontal_dist = math.sqrt(dx**2 + dy**2)
        pitch = math.atan2(-dz, horizontal_dist)
        
        # Camera roll (keep level)
        roll = 0.0
        
        # Convert to quaternion
        qx, qy, qz, qw = URControlWithDetection.euler_to_quat(roll, pitch, yaw)
        
        return qx, qy, qz, qw
    
    def goto_smooth(self, x, y, z, qx, qy, qz, qw, steps=50):
        """Move robot with quaternion orientation - USING YOUR WORKING PARAMETERS"""
        
        if z < 0.25:
            self.get_logger().warn(f"⚠ Target z={z:.3f}m too low, adjusting to 0.25m")
            z = 0.25
        
        self.get_logger().info(f"→ Moving to ({x:.3f}, {y:.3f}, {z:.3f})")
        
        pose = PoseStamped()
        pose.header.frame_id = "world"  # Isaac Sim frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 5
        req.ik_request.avoid_collisions = False
        req.ik_request.robot_state.joint_state.name = self.joints
        req.ik_request.robot_state.joint_state.position = self.current_joints
        req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()
        
        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=7)
        
        try:
            response = future.result()
            if response.error_code.val != 1:
                self.get_logger().error(f"✗ IK failed (code: {response.error_code.val})")
                return False
            
            target_joints = list(response.solution.joint_state.position[:6])
            trajectory = self.interpolate_joints(self.current_joints, target_joints, steps)
            self.execute_trajectory(trajectory, duration_per_step=0.04)
            self.current_joints = target_joints
            self.get_logger().info("  ✓ Movement complete")
            return True
        except Exception as e:
            self.get_logger().error(f"✗ Error: {e}")
            return False
    
    def interpolate_joints(self, start, end, steps=50):
        trajectory = []
        for i in range(steps + 1):
            t = i / steps
            # Smooth S-curve interpolation
            smooth_t = 3*t**2 - 2*t**3
            interpolated = [s + smooth_t * (e - s) for s, e in zip(start, end)]
            trajectory.append(interpolated)
        return trajectory
    
    def execute_trajectory(self, trajectory, duration_per_step=0.04):
        for joints in trajectory:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = self.joints
            msg.position = joints
            self.pub.publish(msg)
            time.sleep(duration_per_step)


def generate_multiview_for_3d_reconstruction(object_center, camera_distance=0.35, robot=None):
    """
    Generate SIMPLE circular viewpoints for 3D reconstruction
    Based on working reference code patterns
    """
    viewpoints = []
    rings = [
        {'name': 'Low', 'elevation_deg': 25, 'n_views': 8},
        {'name': 'Mid', 'elevation_deg': 45, 'n_views': 8},
        {'name': 'High', 'elevation_deg': 65, 'n_views': 8},
    ]
    view_id = 0
    min_z = 0.25
    for ring in rings:
        elevation_rad = math.radians(ring['elevation_deg'])
        n_views = ring['n_views']
        for i in range(n_views):
            azimuth_deg = (360.0 / n_views) * i
            azimuth_rad = math.radians(azimuth_deg)
            x_offset = camera_distance * math.cos(azimuth_rad) * math.cos(elevation_rad)
            y_offset = camera_distance * math.sin(azimuth_rad) * math.cos(elevation_rad)
            z_offset = camera_distance * math.sin(elevation_rad)
            camera_x = object_center['x'] + x_offset
            camera_y = object_center['y'] + y_offset
            camera_z = max(object_center['z'] + z_offset + 0.05, min_z)
            rx = math.pi
            ry = 0.0
            rz = azimuth_rad
            qx, qy, qz, qw = URControlWithDetection.euler_to_quat(rx, ry, rz)
            viewpoints.append({
                'name': f'{view_id:02d}-{ring["name"]}-{azimuth_deg:03.0f}deg',
                'x': camera_x,
                'y': camera_y,
                'z': camera_z,
                'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw
            })
            view_id += 1
    viewpoints.append({
        'name': f'{view_id:02d}-Top',
        'x': object_center['x'],
        'y': object_center['y'],
        'z': object_center['z'] + 0.35,
        'qx': 0.0, 'qy': 0.0, 'qz': 0.0, 'qw': 1.0
    })
    return viewpoints


def main():
    try:
        rclpy.init()
        
        detection_method = 'sam'
        robot = URControlWithDetection(detection_method=detection_method)
        
        print("\n" + "="*70)
        print("UR5e Multi-View Capture for 3D Reconstruction")
        print("Optimized: 3 elevation rings + top view = 25 images")
        print("="*70)
        
        # Wait for camera
        print("\n⏳ Waiting for camera...")
        wait_time = 0
        while robot.latest_image is None and wait_time < 10:
            rclpy.spin_once(robot, timeout_sec=0.5)
            wait_time += 0.5
        
        if robot.latest_image is None:
            print("✗ No camera feed")
            return
        
        print("✓ Camera ready")
        
        # Simple HOME (based on working code)
        HOME = {'x': 0.4, 'y': 0.1, 'z': 0.6}
        rx = math.pi
        ry = 0.0
        rz = 0.0
        qx, qy, qz, qw = URControlWithDetection.euler_to_quat(rx, ry, rz)
        HOME_QUAT = {'qx': qx, 'qy': qy, 'qz': qz, 'qw': qw}

        print(f"\n{'='*70}")
        print("STEP 1: Moving to HOME")
        print(f"{'='*70}")

        if not robot.goto_smooth(HOME['x'], HOME['y'], HOME['z'], 
                                 HOME_QUAT['qx'], HOME_QUAT['qy'], 
                                 HOME_QUAT['qz'], HOME_QUAT['qw']):
            print("✗ Failed to reach HOME")
            return
        time.sleep(1.5)

        # (STEP 3 and onward: Detect object, etc...)
        
        time.sleep(1.5)
        
        # Detect object
        print(f"\n{'='*70}")
        print("STEP 2: Detecting object")
        print(f"{'='*70}")
        
        rclpy.spin_once(robot, timeout_sec=0.5)
        detection_result = robot.detect_object(visualize=True)
        
        if not detection_result['detected']:
            print("✗ No object detected!")
            return
        
        size_info = detection_result['size']
        print(f"✓ Object detected!")
        print(f"  Size: {size_info['width_m']*100:.1f} × {size_info['height_m']*100:.1f} cm")
        
        # Calculate object center in world frame
        object_center_world = {
            'x': HOME['x'] + size_info['center_x_m'],
            'y': HOME['y'] + size_info['center_y_m'],
            'z': 0.05
        }
        
        # Simple multi-view circular capture
        print(f"\n{'='*70}")
        print("STEP 3: Executing robust circular multi-view capture")
        print(f"{'='*70}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"captures/3d_recon_simple_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        n_points = 24  # Fewer, evenly spaced views per ring
        rings = [
            {'name': 'Low', 'elevation_deg': 20, 'radius': 0.32},
            {'name': 'Mid', 'elevation_deg': 45, 'radius': 0.32},
            {'name': 'High', 'elevation_deg': 70, 'radius': 0.32},
        ]
        for ring in rings:
            print(f"\n-- {ring['name']} ring at {ring['elevation_deg']}° --")
            waypoints = generate_circular_ring_waypoints(
                object_center=object_center_world,
                radius=ring['radius'],
                elevation_deg=ring['elevation_deg'],
                n_points=n_points,
                robot=robot
            )
            print(f"  Reachable views: {len(waypoints)}/{n_points}")
            execute_ring_capture(robot, waypoints, save_dir)
        # Top view
        print(f"\n-- Top view --")
        top_pos = {
            'x': object_center_world['x'],
            'y': object_center_world['y'],
            'z': object_center_world['z'] + 0.42
        }
        robot.goto_smooth(top_pos['x'], top_pos['y'], top_pos['z'], 0.0, 0.0, 0.0, 1.0)
        robot.capture_photo("top_view", save_dir)
        # Return HOME
        print(f"\n🏠 Returning to HOME...")
        robot.goto_smooth(HOME['x'], HOME['y'], HOME['z'], HOME_QUAT['qx'], HOME_QUAT['qy'], HOME_QUAT['qz'], HOME_QUAT['qw'])
        print(f"\n{'='*70}")
        print("✓ SIMPLE CIRCULAR CAPTURE COMPLETE")
        print(f"{'='*70}")
        print(f"  Save location: {os.path.abspath(save_dir)}")
        print(f"\n💡 Ready for 3D reconstruction!")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()