#!/usr/bin/env python3
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
        
        # --- BOX AND CAMERA CALIBRATION DATA ---
        self.box_pos = np.array([0.3, 0.3, 0.2])  # Center of box
        self.box_scale = 0.15
        
        # Camera offset relative to wrist_3_link/tool0
        # We subtract this from our target to put the camera lens where the tool0 would be
        self.cam_offset = np.array([-0.01578, 0.00732, 0.01917])
        
        self.latest_image = None
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # ROS Setup
        self.camera_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.pub = self.create_publisher(JointState, '/joint_command', 10)
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        
        self.current_joints = None
        self.joints = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                       'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

    def image_callback(self, msg):
        self.latest_image = msg

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

    def compute_ik(self, x, y, z, rx, ry, rz):
        # APPLY CAMERA OFFSET CORRECTION
        # To place camera at (x,y,z), the robot flange must be at (target - offset)
        target_x = x - self.cam_offset[0]
        target_y = y - self.cam_offset[1]
        target_z = z - self.cam_offset[2]

        pose = PoseStamped()
        pose.header.frame_id = "world"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = float(target_x), float(target_y), float(target_z)
        
        qx, qy, qz, qw = self.euler_to_quat(rx, ry, rz)
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = qx, qy, qz, qw
        
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ur_manipulator"
        req.ik_request.ik_link_name = "tool0"
        req.ik_request.pose_stamped = pose
        req.ik_request.avoid_collisions = True
        
        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2)
        res = future.result()
        return list(res.solution.joint_state.position[:6]) if res and res.error_code.val == 1 else None

    def move_smooth(self, x, y, z, rx, ry, rz):
        target_joints = self.compute_ik(x, y, z, rx, ry, rz)
        if not target_joints: return False
        
        steps = 40
        start_joints = self.current_joints if self.current_joints else target_joints
        
        for i in range(steps + 1):
            t = i / steps
            smooth_t = 3*t**2 - 2*t**3
            interp = [s + smooth_t * (e - s) for s, e in zip(start_joints, target_joints)]
            msg = JointState(name=self.joints, position=interp)
            msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(msg)
            time.sleep(0.05)
        self.current_joints = target_joints
        return True

    def capture_photo(self, name):
        if not self.latest_image: return False
        path = os.path.join(self.save_dir, f"{name}.ppm")
        img = np.frombuffer(self.latest_image.data, dtype=np.uint8).reshape((self.latest_image.height, self.latest_image.width, 3))
        if self.latest_image.encoding == 'bgr8': img = img[:, :, ::-1]
        with open(path, 'wb') as f:
            f.write(f"P6\n{self.latest_image.width} {self.latest_image.height}\n255\n".encode())
            f.write(img.tobytes())
        return True

def main():
    rclpy.init()
    ur = URPhotoCapture()
    
    # Orientation Constants (Radians)
    DOWN = math.pi         # Camera looking straight down
    SIDE_TILT = 2.35       # ~135 degrees (angled looking inwards and down)
    
    # Trajectory: [Name, x, y, z, rx, ry, rz]
    # Centered around box [0.3, 0.3, 0.2]
    traj = [
        # TOP VIEW
        ["Top_View", 0.3, 0.3, 0.55, DOWN, 0.0, 0.0],
        
        # SIDE VIEWS (Moved back and tilted)
        # Looking from Front (positive X)
        ["Front_Side", 0.55, 0.3, 0.35, SIDE_TILT, 0.0, 0.0],
        # Looking from Left (negative Y)
        ["Left_Side", 0.3, 0.05, 0.35, SIDE_TILT, 0.0, -math.pi/2],
        # Looking from Right (positive Y)
        ["Right_Side", 0.3, 0.55, 0.35, SIDE_TILT, 0.0, math.pi/2],
    ]

    try:
        for p in traj:
            print(f"Moving to {p[0]}...")
            if ur.move_smooth(p[1], p[2], p[3], p[4], p[5], p[6]):
                time.sleep(1.0)
                ur.capture_photo(p[0])
                print(f"Captured {p[0]}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()