#!/usr/bin/env python3
"""
UR5e Photo Capture - Full Box Coverage
- Captures top + all 4 sides of a box
- Camera automatically looks at box center
- Smooth joint-space motion
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

        self.latest_image = None
        self.current_joints = None
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f"✓ Saving images to: {os.path.abspath(self.save_dir)}")

        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10
        )

        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )

        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')

        self.joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]

        while not self.ik.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for /compute_ik service...")

        self.get_logger().info("✓ MoveIt IK service ready!")

    # -------------------------------------------------------
    # CALLBACKS
    # -------------------------------------------------------
    def image_callback(self, msg):
        self.latest_image = msg

    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            self.current_joints = list(msg.position[:6])

    # -------------------------------------------------------
    # CAMERA ORIENTATION (LOOK-AT)
    # -------------------------------------------------------
    @staticmethod
    def look_at_orientation(camera_pos, target_pos):
        dx, dy, dz = target_pos - camera_pos
        yaw = math.atan2(dy, dx)
        dist_xy = math.sqrt(dx**2 + dy**2)
        pitch = math.atan2(dist_xy, -dz)
        return pitch, 0.0, yaw

    @staticmethod
    def euler_to_quat(rx, ry, rz):
        cy, sy = math.cos(rz/2), math.sin(rz/2)
        cp, sp = math.cos(ry/2), math.sin(ry/2)
        cr, sr = math.cos(rx/2), math.sin(rx/2)
        return (
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy
        )

    # -------------------------------------------------------
    # IK + MOTION
    # -------------------------------------------------------
    def compute_ik(self, x, y, z, rx, ry, rz):
        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z

        qx, qy, qz, qw = self.euler_to_quat(rx, ry, rz)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = True
        req.ik_request.timeout.sec = 2

        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3)

        if future.result().error_code.val == 1:
            return list(future.result().solution.joint_state.position[:6])
        return None

    def interpolate(self, start, end, steps):
        return [
            [s + (e-s)*(3*t**2 - 2*t**3)
             for s, e in zip(start, end)]
            for t in np.linspace(0, 1, steps)
        ]

    def move_smooth(self, x, y, z, rx, ry, rz, steps=40):
        self.get_logger().info(f"→ Moving to ({x:.2f}, {y:.2f}, {z:.2f})")

        target = self.compute_ik(x, y, z, rx, ry, rz)
        if target is None:
            self.get_logger().error("✗ IK failed")
            return False

        if self.current_joints is None:
            msg = JointState(name=self.joints, position=target)
            self.pub.publish(msg)
            time.sleep(2)
            return True

        traj = self.interpolate(self.current_joints, target, steps)
        for q in traj:
            self.pub.publish(JointState(name=self.joints, position=q))
            time.sleep(0.05)

        self.current_joints = target
        return True

    # -------------------------------------------------------
    # IMAGE SAVE
    # -------------------------------------------------------
    def capture_photo(self, name):
        if self.latest_image is None:
            return False

        msg = self.latest_image
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)

        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.ppm"
        path = os.path.join(self.save_dir, filename)

        with open(path, "wb") as f:
            f.write(f"P6\n{msg.width} {msg.height}\n255\n".encode())
            f.write(img.tobytes())

        self.get_logger().info(f"📷 Saved: {filename}")
        return True


# ===========================================================
# MAIN
# ===========================================================
def main():
    rclpy.init()
    save_dir = f"captured_photos_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ur = URPhotoCapture(save_dir)

    # -------------------------
    # BOX DEFINITION
    # -------------------------
    BOX_CENTER = np.array([0.3, 0.3, 0.2])
    BOX_SIZE = 0.15
    BOX_HALF = BOX_SIZE / 2

    RADIUS = 0.30
    TOP_Z = BOX_CENTER[2] + BOX_HALF + 0.25
    SIDE_Z = BOX_CENTER[2] + BOX_HALF + 0.05

    HOME = (0.4, 0.4, 0.6, math.pi, 0.0, 0.0)

    # -------------------------
    # VIEWPOINTS
    # -------------------------
    poses = []

    # Top
    poses.append(("Top", BOX_CENTER[0], BOX_CENTER[1], TOP_Z, math.pi, 0, 0))

    # Sides
    offsets = [(-RADIUS, 0), (RADIUS, 0), (0, -RADIUS), (0, RADIUS)]
    names = ["Front", "Back", "Left", "Right"]

    for (dx, dy), name in zip(offsets, names):
        cam = np.array([BOX_CENTER[0]+dx, BOX_CENTER[1]+dy, SIDE_Z])
        rx, ry, rz = ur.look_at_orientation(cam, BOX_CENTER)
        poses.append((name, cam[0], cam[1], cam[2], rx, ry, rz))

    # -------------------------
    # EXECUTION
    # -------------------------
    ur.move_smooth(*HOME)
    time.sleep(1)

    for name, x, y, z, rx, ry, rz in poses:
        if ur.move_smooth(x, y, z, rx, ry, rz):
            time.sleep(0.8)
            ur.capture_photo(name)

    ur.move_smooth(*HOME)

    print(f"\n✓ Finished. Images saved in:\n{os.path.abspath(save_dir)}")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
