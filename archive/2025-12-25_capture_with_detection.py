#!/usr/bin/env python3
"""
UR5e Robot with Automatic Object Detection and Multi-View Capture

Base: original working code (doc 5) — unchanged except one addition:
  FIX 4 - Per-viewpoint orientation so camera faces the object instead
           of always pointing straight down. This is the only meaningful
           change needed for better 3D reconstruction images.

Everything else (HOME, IK call, interpolate_joints, execute_trajectory,
frame_id, detection, planning scene) is 100% identical to the working script.
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

from modular_object_detection import create_detection_pipeline


# =============================================================================
# FIX 4 helper — only new code added to the original script
# =============================================================================
def orientation_pointing_at(cam_pos, target_pos):
    """
    Compute (rx, ry, rz) Euler angles so the camera Z-axis points from
    cam_pos toward target_pos.
    cam_pos / target_pos are (x, y, z) tuples in world frame.
    """
    dx = target_pos[0] - cam_pos[0]
    dy = target_pos[1] - cam_pos[1]
    dz = target_pos[2] - cam_pos[2]
    dist_horiz = math.sqrt(dx**2 + dy**2)
    yaw   =  math.atan2(dy, dx)
    pitch = -math.atan2(-dz, dist_horiz)
    return 0.0, pitch, yaw   # rx, ry, rz


# =============================================================================
# Robot node — identical to working original
# =============================================================================
class URControlWithDetection(Node):
    def __init__(self, detection_method='color'):
        super().__init__('ur_control_detection')

        # Robot setup — UNCHANGED
        self.ik  = self.create_client(GetPositionIK, '/compute_ik')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.current_joints = None
        self.joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        # Camera setup — UNCHANGED
        self.latest_image  = None
        self.camera_topic  = '/camera/image_raw'
        self.bridge        = CvBridge()
        self.camera_sub    = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10)

        # Joint state subscription — UNCHANGED
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Detection pipeline — UNCHANGED
        sam_params = {}
        if detection_method == 'sam':
            sam_params = {'model_type': 'vit_h', 'checkpoint_path': None}

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

        # Planning scene — UNCHANGED
        self.scene_pub = self.create_publisher(PlanningScene, '/planning_scene', 10)
        time.sleep(0.5)
        self.add_ground_plane()

        # Wait for IK service — UNCHANGED
        while not self.ik.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for IK service...")

        # Wait for initial joint state — UNCHANGED
        wait_time = 0
        while self.current_joints is None and wait_time < 10:
            rclpy.spin_once(self, timeout_sec=0.5)
            wait_time += 0.5

        if self.current_joints:
            self.get_logger().info("✓ Robot ready!")

        time.sleep(1.0)

    # ── Callbacks — UNCHANGED ─────────────────────────────────────────────
    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])

    def image_callback(self, msg):
        self.latest_image = msg

    def get_latest_image_as_cv(self):
        if self.latest_image is None:
            return None
        try:
            return self.bridge.imgmsg_to_cv2(self.latest_image, "rgb8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return None

    # ── Detection — UNCHANGED ─────────────────────────────────────────────
    def detect_object(self, visualize=False):
        cv_image = self.get_latest_image_as_cv()
        if cv_image is None:
            self.get_logger().warn("No camera image available")
            return {'detected': False}

        results = self.detection_pipeline.detect_and_measure(cv_image, visualize=visualize)

        if isinstance(results, tuple):
            results, vis_image = results
            if visualize:
                cv2.imwrite('detection_result.jpg',
                            cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        return results

    # ── Photo capture — UNCHANGED ─────────────────────────────────────────
    def capture_photo(self, position_name, save_dir="captured_photos"):
        if self.latest_image is None:
            return False

        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = position_name.replace(' ', '_')
        filename  = f"{timestamp}_{safe_name}.jpg"
        filepath  = os.path.join(save_dir, filename)

        try:
            cv_image = self.get_latest_image_as_cv()
            cv2.imwrite(filepath, cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f"  📷 Photo saved: {filename}")
            return True
        except Exception as e:
            self.get_logger().error(f"  ✗ Failed to save: {e}")
            return False

    # ── Planning scene — UNCHANGED ────────────────────────────────────────
    def add_ground_plane(self):
        self.get_logger().info("Adding ground plane to planning scene...")

        scene_msg = PlanningScene()
        scene_msg.is_diff = True

        ground = CollisionObject()
        ground.header.frame_id = "World"        # capital W — unchanged
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
        self.get_logger().info("  To verify: ros2 topic echo /planning_scene --once")
        time.sleep(0.5)

    # ── Motion — UNCHANGED (exactly as in working original) ───────────────
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

    def goto_smooth(self, x, y, z, rx=math.pi, ry=0.0, rz=0.0, steps=40):
        """Exactly the original working goto_smooth — UNCHANGED."""
        if z < 0.3:
            self.get_logger().warn(f"⚠ Target z={z:.3f}m too low, adjusting to 0.3m")
            z = 0.3

        self.get_logger().info(f"→ Moving to ({x:.3f}, {y:.3f}, {z:.3f})")

        pose = PoseStamped()
        pose.header.frame_id = "World"          # capital W — unchanged
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
        req.ik_request.group_name       = "arm"
        req.ik_request.ik_link_name     = "wrist_3_link"
        req.ik_request.pose_stamped     = pose
        req.ik_request.timeout.sec      = 5
        req.ik_request.avoid_collisions = True
        req.ik_request.robot_state.joint_state.name     = self.joints
        req.ik_request.robot_state.joint_state.position = self.current_joints
        req.ik_request.robot_state.joint_state.header.stamp = (
            self.get_clock().now().to_msg())

        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=7)

        try:
            response = future.result()
            if response.error_code.val != 1:
                self.get_logger().error(f"✗ IK failed (code: {response.error_code.val})")
                return False

            target_joints = list(response.solution.joint_state.position[:6])
            trajectory    = self.interpolate_joints(self.current_joints, target_joints, steps)
            self.execute_trajectory(trajectory, duration_per_step=0.05)
            self.current_joints = target_joints
            self.get_logger().info("  ✓ Movement complete")
            return True

        except Exception as e:
            self.get_logger().error(f"✗ Error: {e}")
            return False

    def interpolate_joints(self, start, end, steps=30):
        """UNCHANGED from original."""
        trajectory = []
        for i in range(steps + 1):
            t = i / steps
            smooth_t = 3*t**2 - 2*t**3
            interpolated = [s + smooth_t * (e - s) for s, e in zip(start, end)]
            trajectory.append(interpolated)
        return trajectory

    def execute_trajectory(self, trajectory, duration_per_step=0.05):
        """UNCHANGED from original."""
        for joints in trajectory:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name         = self.joints
            msg.position     = joints
            self.pub.publish(msg)
            time.sleep(duration_per_step)


# =============================================================================
# FIX 4: generate_viewpoints_around_object — only this function is changed
# Original had rx=pi, ry=0, rz=0 for every viewpoint (camera always down).
# Now each ring viewpoint gets a unique orientation facing the object.
# Top view keeps rx=pi (straight down) since that is correct for top view.
# =============================================================================
def generate_viewpoints_around_object(object_center, object_size,
                                       n_views=12, camera_distance=0.3):
    viewpoints = []

    # Top view — UNCHANGED (straight down is correct for top)
    viewpoints.append({
        'name': '0-Top',
        'x':  object_center['x'],
        'y':  object_center['y'],
        'z':  object_center['z'] + 0.5,
        'rx': math.pi, 'ry': 0.0, 'rz': 0.0
    })

    # Ring at 50° elevation — positions UNCHANGED, only orientation is new
    elevation_angle = math.radians(50)
    n_ring_views    = n_views - 1

    for i in range(n_ring_views):
        azimuth_deg = (360.0 / n_ring_views) * i
        azimuth_rad = math.radians(azimuth_deg)

        x_offset = camera_distance * math.cos(azimuth_rad) * math.cos(elevation_angle)
        y_offset = camera_distance * math.sin(azimuth_rad) * math.cos(elevation_angle)
        z_offset = camera_distance * math.sin(elevation_angle)

        final_z = object_center['z'] + z_offset + 0.35
        if final_z < 0.2:
            final_z = 0.2

        cam_x = object_center['x'] + x_offset
        cam_y = object_center['y'] + y_offset

        # FIX 4: compute orientation so camera faces the object
        # Original was always: rx=math.pi, ry=0.0, rz=0.0  (straight down)
        rx, ry, rz = orientation_pointing_at(
            cam_pos    = (cam_x, cam_y, final_z),
            target_pos = (object_center['x'], object_center['y'], object_center['z'])
        )

        viewpoints.append({
            'name': f'{i+1}-Ring{azimuth_deg:.0f}deg',
            'x': cam_x, 'y': cam_y, 'z': final_z,
            'rx': rx, 'ry': ry, 'rz': rz
        })

    return viewpoints


# =============================================================================
# Main — UNCHANGED from original
# =============================================================================
def main():
    try:
        rclpy.init()

        detection_method = 'color'   # change to 'sam' once torch is installed
        robot = URControlWithDetection(detection_method=detection_method)

        print("\n" + "="*70)
        print("UR5e Multi-View Capture")
        print("Method: Direct IK + Ground Plane in Planning Scene")
        print("="*70)

        # Wait for camera — UNCHANGED
        print("\n⏳ Waiting for camera...")
        wait_time = 0
        while robot.latest_image is None and wait_time < 10:
            rclpy.spin_once(robot, timeout_sec=0.5)
            wait_time += 0.5

        if robot.latest_image is None:
            print("✗ No camera feed")
            return
        print("✓ Camera ready")

        # HOME — UNCHANGED (exactly as in working original)
        HOME = {'x': 0.45, 'y': 0.15, 'z': 0.7,
                'rx': math.pi, 'ry': 0.0, 'rz': 0.0}

        print(f"\n{'='*70}")
        print("STEP 1: Moving to HOME")
        print(f"{'='*70}")

        if not robot.goto_smooth(HOME['x'], HOME['y'], HOME['z'],
                                  HOME['rx'], HOME['ry'], HOME['rz']):
            print("✗ Failed to reach HOME")
            return

        time.sleep(1.5)

        # Detect — UNCHANGED
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

        # Object world position — UNCHANGED
        object_center_world = {
            'x': HOME['x'] + size_info['center_x_m'],
            'y': HOME['y'] + size_info['center_y_m'],
            'z': 0.05
        }

        # Generate viewpoints — calls updated function (Fix 4 only)
        print(f"\n{'='*70}")
        print("STEP 3: Generating viewpoints")
        print(f"{'='*70}")

        viewpoints = generate_viewpoints_around_object(
            object_center=object_center_world,
            object_size=size_info,
            n_views=12,
            camera_distance=0.3
        )

        print(f"✓ {len(viewpoints)} viewpoints generated")
        print(f"  Distance: 30cm, Elevation: 50°")
        print(f"  Camera orientation: faces object at each ring position")

        # Capture loop — UNCHANGED
        print(f"\n{'='*70}")
        print("STEP 4: Capturing images")
        print(f"{'='*70}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir  = (f"captures/"
                     f"multiview_capture_{timestamp}")

        success_count = 0
        photo_count   = 0

        for i, vp in enumerate(viewpoints):
            print(f"\n📸 {i+1}/{len(viewpoints)}: {vp['name']}")

            success = robot.goto_smooth(vp['x'], vp['y'], vp['z'],
                                         vp['rx'], vp['ry'], vp['rz'])

            if success:
                success_count += 1
                time.sleep(0.8)

                old_img    = robot.latest_image
                frame_wait = 0
                while frame_wait < 2.0:
                    rclpy.spin_once(robot, timeout_sec=0.1)
                    if robot.latest_image != old_img:
                        break
                    frame_wait += 0.1

                if robot.capture_photo(vp['name'], save_dir):
                    photo_count += 1
            else:
                print(f"✗ Skipped (collision or unreachable)")

        # Return HOME — UNCHANGED
        print(f"\n🏠 Returning to HOME...")
        robot.goto_smooth(HOME['x'], HOME['y'], HOME['z'],
                           HOME['rx'], HOME['ry'], HOME['rz'])

        # Summary — UNCHANGED
        print(f"\n{'='*70}")
        print("✓ COMPLETE")
        print(f"{'='*70}")
        print(f"  Reached: {success_count}/{len(viewpoints)}")
        print(f"  Photos:  {photo_count}/{len(viewpoints)}")
        print(f"  Folder:  {os.path.abspath(save_dir)}")
        print(f"{'='*70}\n")

    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()