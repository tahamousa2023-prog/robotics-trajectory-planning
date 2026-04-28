#!/usr/bin/env python3
"""
ur5e_orbit_capture_detect.py
===========================

What it does (keeps your working pipeline):
  1) Move to HOME (IK + smooth interpolation + /joint_command)
  2) Run object detection on the HOME camera image
  3) Convert detection result into a world-frame object center (x,y) and pick orbit params
  4) Generate orbit poses that look at the detected object center (fixed-roll "human" feel)
  5) Move + capture images around the object

Notes:
  - Uses the SAME IK service /compute_ik and the SAME manual interpolation
  - Keeps your stable look_at_quaternion() (fixed roll)
  - Keeps your wrist_2 flip + continuity fix
  - Home pose is taken from your older script

Requires:
  - modular_object_detection.py providing create_detection_pipeline()
"""

import os
import math
import time
import threading
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped
from moveit_msgs.srv import GetPositionIK

from cv_bridge import CvBridge
import cv2

from modular_object_detection import create_detection_pipeline


# ==============================================================================
#  CONFIGURATION
# ==============================================================================

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# IK parameters
IK_GROUP   = "arm"
IK_LINK    = "wrist_3_link"
IK_FRAME   = "World"     # Isaac Sim uses "World"
IK_TIMEOUT = 5           # seconds

# Topics / services
CAMERA_TOPIC  = "/camera/image_raw"
COMMAND_TOPIC = "/joint_command"
IK_SERVICE    = "/compute_ik"

# Output
CAPTURE_BASE = "/home/AP_PathMatters/path_matters/trajectory/captures"

# Timing
INTERP_STEPS         = 40
INTERP_STEP_TIME     = 0.05
WAIT_AT_POSE         = 2.0
JOINT_STATE_TIMEOUT  = 15.0
CAMERA_READY_TIMEOUT = 10.0

# Orbit sampling
NUM_POSES = 8

# Camera roll fix: wrist_2_joint (index 4)
FLIP_JOINT_INDEX = 4
FLIP_ANGLE = math.pi

# HOME (from your detection script)
HOME_POS = np.array([0.45, 0.15, 0.70], dtype=float)
# Where HOME "looks" (just into the table area; doesn't need to be exact)
HOME_LOOK_TARGET = np.array([0.45, 0.15, 0.05], dtype=float)

# Detection method
DETECTION_METHOD = "sam"   # "sam" or "color"
DETECTION_VISUALIZE = True

# Assumed table/object base Z if you don't have depth
DEFAULT_OBJECT_Z = 0.05

# Camera model params for your detection pipeline (match your sim camera)
CAMERA_HEIGHT = 0.7
IMAGE_WIDTH   = 1280
IMAGE_HEIGHT  = 720
FOV_DEG       = 60

# Wait-until-reached parameters
REACH_TOL = 0.02          # rad, ~1.1°
REACH_STABLE_COUNT = 5    # how many consecutive checks within tol
REACH_TIMEOUT = 8.0       # seconds max wait after trajectory


# ==============================================================================
#  MATH / GEOMETRY
# ==============================================================================

def smoothstep(t: float) -> float:
    return 3.0 * t * t - 2.0 * t * t * t


def interpolate_joints(start: List[float], end: List[float], steps: int) -> List[List[float]]:
    traj = []
    for i in range(steps + 1):
        t = smoothstep(i / steps)
        traj.append([s + t * (e - s) for s, e in zip(start, end)])
    return traj


def make_continuous(target: float, reference: float) -> float:
    """Pick an equivalent angle (±2πk) closest to reference to avoid wrap spins."""
    while target - reference > math.pi:
        target -= 2.0 * math.pi
    while target - reference < -math.pi:
        target += 2.0 * math.pi
    return target


def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=float)
    return q / np.linalg.norm(q)


def look_at_quaternion(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Look-at with FIXED ROLL (stable "human taking photos" feel).

    - forward (local -Z) points at target
    - roll is fixed by forcing tool local +X ("right") to align with a
      constant world reference projected onto the view plane.
    """
    forward = target - eye
    nf = np.linalg.norm(forward)
    if nf < 1e-9:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    forward = forward / nf

    # local -Z points forward => local +Z points backwards
    z_axis = -forward  # tool local +Z in world coords

    # choose constant world reference for "right"
    world_ref = np.array([0.0, 1.0, 0.0], dtype=float)  # world +Y

    # project onto plane perpendicular to z_axis
    x_axis = world_ref - np.dot(world_ref, z_axis) * z_axis
    nx = np.linalg.norm(x_axis)

    if nx < 1e-6:
        world_ref = np.array([1.0, 0.0, 0.0], dtype=float)  # fallback world +X
        x_axis = world_ref - np.dot(world_ref, z_axis) * z_axis
        nx = np.linalg.norm(x_axis)
        if nx < 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    x_axis = x_axis / nx
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    return rotation_matrix_to_quat_xyzw(R)


def compute_orbit_poses(center: np.ndarray, radius: float, z: float, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    poses = []
    for i in range(n):
        angle = i * (2.0 * math.pi / n)
        pos = np.array([
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            z,
        ], dtype=float)
        quat = look_at_quaternion(pos, center)
        poses.append((pos, quat))
    return poses


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def orbit_params_from_detection(det: dict) -> Tuple[float, float]:
    """
    Returns (radius, height_above_object_z) with HARD caps:
      radius <= 0.40
      height <= 0.45
    """
    size = det.get("size", {}) if isinstance(det, dict) else {}
    w = float(size.get("width_m", 0.20))
    h = float(size.get("height_m", 0.20))
    base = max(w, h, 0.10)

    # Scale gently with object size, but NEVER exceed limits
    radius = 1.2 * base
    height = 0.70 + 0.35 * base

    # HARD caps (cannot exceed)
    radius = min(max(radius, 0.22), 0.35)
    height = min(max(height, 0.25), 0.5)

    return radius, height



# ==============================================================================
#  ROS NODE
# ==============================================================================

class RobotNode(Node):
    def __init__(self):
        super().__init__("robot_node")

        # IK client
        self.ik_client = self.create_client(GetPositionIK, IK_SERVICE)

        # Command publisher
        self.cmd_pub = self.create_publisher(JointState, COMMAND_TOPIC, qos_profile=10)

        # Joint state subscriber
        self.current_joints: Optional[List[float]] = None
        self.create_subscription(JointState, "/joint_states", self._js_callback, qos_profile=10)

        # Camera subscriber
        self._bridge = CvBridge()
        self._latest_img = None
        self._img_lock = threading.Lock()
        self.create_subscription(Image, CAMERA_TOPIC, self._img_callback, qos_profile=10)

        # Detection pipeline
        sam_params = {}
        if DETECTION_METHOD == "sam":
            sam_params = {"model_type": "vit_h", "checkpoint_path": None}

        self.detection_pipeline = create_detection_pipeline(
            method=DETECTION_METHOD,
            camera_height=CAMERA_HEIGHT,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            fov_deg=FOV_DEG,
            **sam_params
        )

    def _js_callback(self, msg: JointState):
        positions = []
        for jn in JOINT_NAMES:
            if jn not in msg.name:
                return
            positions.append(msg.position[list(msg.name).index(jn)])
        self.current_joints = positions

    def _img_callback(self, msg: Image):
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self._img_lock:
                self._latest_img = img
        except Exception as e:
            self.get_logger().error(f"CvBridge: {e}")

    def capture(self) -> Optional[np.ndarray]:
        with self._img_lock:
            return None if self._latest_img is None else self._latest_img.copy()

    def get_latest_image_rgb(self) -> Optional[np.ndarray]:
        img_bgr = self.capture()
        if img_bgr is None:
            return None
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def detect_object(self, save_dir: str, visualize: bool = True) -> dict:
        """
        Run detection on the latest camera frame and save:
        - home_raw.png (always)
        - home_detection_vis.png (if detector returns visualization)
        """
        img_bgr = self.capture()
        if img_bgr is None:
            return {"detected": False}

        os.makedirs(save_dir, exist_ok=True)

        # Save raw HOME frame (what detector sees)
        cv2.imwrite(os.path.join(save_dir, "home_raw.png"), img_bgr)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = self.detection_pipeline.detect_and_measure(img_rgb, visualize=visualize)

        # pipeline might return (result_dict, vis_img_rgb)
        if isinstance(res, tuple) and len(res) == 2:
            result, vis_img = res
            if visualize and vis_img is not None:
                cv2.imwrite(
                    os.path.join(save_dir, "home_detection_vis.png"),
                    cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
                )
            return result if isinstance(result, dict) else {"detected": False}

        return res if isinstance(res, dict) else {"detected": False}



    def compute_ik(self, position: np.ndarray, quat_xyzw: np.ndarray) -> Optional[List[float]]:
        if self.current_joints is None:
            self.get_logger().error("compute_ik: no current_joints available")
            return None

        pose = PoseStamped()
        pose.header.frame_id = IK_FRAME
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(position[0])
        pose.pose.position.y = float(position[1])
        pose.pose.position.z = float(position[2])
        pose.pose.orientation.x = float(quat_xyzw[0])
        pose.pose.orientation.y = float(quat_xyzw[1])
        pose.pose.orientation.z = float(quat_xyzw[2])
        pose.pose.orientation.w = float(quat_xyzw[3])

        req = GetPositionIK.Request()
        req.ik_request.group_name = IK_GROUP
        req.ik_request.ik_link_name = IK_LINK
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = IK_TIMEOUT
        req.ik_request.avoid_collisions = True
        req.ik_request.robot_state.joint_state.name = JOINT_NAMES
        req.ik_request.robot_state.joint_state.position = self.current_joints
        req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

        future = self.ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=IK_TIMEOUT + 2)

        if not future.done():
            self.get_logger().error("IK service timed out")
            return None

        response = future.result()
        if response.error_code.val != 1:
            self.get_logger().error(f"IK failed, error code: {response.error_code.val}")
            return None

        sol = response.solution.joint_state
        result = []
        for jn in JOINT_NAMES:
            if jn not in sol.name:
                self.get_logger().error(f"IK solution missing joint '{jn}'")
                return None
            result.append(sol.position[list(sol.name).index(jn)])
        return result

    def wait_until_reached(self, target_joints: List[float],
                        tol: float = REACH_TOL,
                        stable_count: int = REACH_STABLE_COUNT,
                        timeout: float = REACH_TIMEOUT) -> bool:
        """
        Wait until /joint_states is close to target_joints.
        Requires consecutive 'stable_count' hits within tol to avoid capturing mid-swing.
        """
        t0 = time.time()
        ok_hits = 0

        while time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)

            if self.current_joints is None:
                continue

            err = np.max(np.abs(np.array(self.current_joints) - np.array(target_joints)))
            if err <= tol:
                ok_hits += 1
                if ok_hits >= stable_count:
                    return True
            else:
                ok_hits = 0

        self.get_logger().warn(f"wait_until_reached: timeout, max_err={err:.4f} rad")
        return False

    def execute_trajectory(self, trajectory: List[List[float]]):
        for positions in trajectory:
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = JOINT_NAMES
            msg.position = positions
            self.cmd_pub.publish(msg)
            time.sleep(INTERP_STEP_TIME)

    def move_to_pose(self, pos: np.ndarray, quat: np.ndarray, steps: int = INTERP_STEPS) -> bool:
        rclpy.spin_once(self, timeout_sec=0.1)

        target_joints = self.compute_ik(pos, quat)
        if target_joints is None:
            return False

        # camera roll fix: flip wrist_2 and keep it continuous
        target_joints[FLIP_JOINT_INDEX] = make_continuous(
            target_joints[FLIP_JOINT_INDEX] + FLIP_ANGLE,
            self.current_joints[FLIP_JOINT_INDEX],
        )

        trajectory = interpolate_joints(self.current_joints, target_joints, steps)
        self.execute_trajectory(trajectory)

        # IMPORTANT: do NOT assume we're there yet — wait for /joint_states to converge
        reached = self.wait_until_reached(target_joints)

        # keep internal state consistent with reality
        if reached:
            self.current_joints = target_joints
        else:
            # refresh current_joints from sim one last time
            rclpy.spin_once(self, timeout_sec=0.1)

        return reached



# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    rclpy.init()

    # Output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(CAPTURE_BASE, f"detect_orbit_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[OUTPUT] {output_dir}")

    robot = RobotNode()

    # Wait for IK service
    print("[INITીINIT] Waiting for IK service …")
    while not robot.ik_client.wait_for_service(timeout_sec=1.0):
        print("[INIT]   still waiting …")
    print("[INIT] ✓ IK service available")

    # Wait for /joint_states
    print("[INIT] Waiting for /joint_states …")
    deadline = time.time() + JOINT_STATE_TIMEOUT
    while robot.current_joints is None and time.time() < deadline:
        rclpy.spin_once(robot, timeout_sec=0.1)
    if robot.current_joints is None:
        print("[ERROR] No /joint_states. Is Isaac Sim playing?")
        robot.destroy_node()
        rclpy.shutdown()
        return
    print(f"[INIT] ✓ joint_states: [{', '.join(f'{v:.4f}' for v in robot.current_joints)}]")

    # Wait for camera
    print("[INIT] Waiting for camera …")
    deadline = time.time() + CAMERA_READY_TIMEOUT
    while robot.capture() is None and time.time() < deadline:
        rclpy.spin_once(robot, timeout_sec=0.1)
    if robot.capture() is None:
        print("[ERROR] No camera image. Check /camera/image_raw")
        robot.destroy_node()
        rclpy.shutdown()
        return
    print("[INIT] ✓ camera ready")

    # ----------------------------------------------------------------------
    # STEP 1: Go HOME
    # ----------------------------------------------------------------------
    print("[STEP 1] Going to HOME …")
    home_quat = look_at_quaternion(HOME_POS, HOME_LOOK_TARGET)
    if not robot.move_to_pose(HOME_POS, home_quat):
        print("[ERROR] Failed to reach HOME")
        robot.destroy_node()
        rclpy.shutdown()
        return

    time.sleep(1.0)
    rclpy.spin_once(robot, timeout_sec=0.2)

    # ----------------------------------------------------------------------
    # STEP 2: Detect object (from HOME image)
    # ----------------------------------------------------------------------
    print("[STEP 2] Detecting object …")
    det = robot.detect_object(save_dir=output_dir, visualize=True)
    print(f"[DETECT] saved: {os.path.join(output_dir, 'home_raw.png')}")
    print(f"[DETECT] saved: {os.path.join(output_dir, 'home_raw.png')}")
    print(f"[DETECT] saved: {os.path.join(output_dir, 'home_detection_vis.png')} (if available)")


    if not det.get("detected", False):
        print("[ERROR] No object detected (detected=False).")
        robot.destroy_node()
        rclpy.shutdown()
        return

    size = det.get("size", {})
    cx_m = float(size.get("center_x_m", 0.0))
    cy_m = float(size.get("center_y_m", 0.0))
    w_m  = float(size.get("width_m", 0.20))
    h_m  = float(size.get("height_m", 0.20))

    print(f"[DETECT] center_x_m={cx_m:.3f}  center_y_m={cy_m:.3f}  "
          f"width_m={w_m:.3f}  height_m={h_m:.3f}")

    # Convert to world object center (same assumption as your older script)
    object_center = np.array([
        HOME_POS[0] + cx_m,
        HOME_POS[1] + cy_m,
        DEFAULT_OBJECT_Z
    ], dtype=float)

    # Decide orbit params based on size
    orbit_radius, orbit_height = orbit_params_from_detection(det)
    orbit_z = object_center[2] + orbit_height

    print(f"[ORBIT] center=({object_center[0]:.3f},{object_center[1]:.3f},{object_center[2]:.3f}) "
          f"radius={orbit_radius:.3f}  orbit_z={orbit_z:.3f}  poses={NUM_POSES}")

    # ----------------------------------------------------------------------
    # STEP 3: Compute orbit poses around detected object center
    # ----------------------------------------------------------------------
    poses = compute_orbit_poses(center=object_center, radius=orbit_radius, z=orbit_z, n=NUM_POSES)

    # ----------------------------------------------------------------------
    # STEP 4: Orbit capture (same as your working loop)
    # ----------------------------------------------------------------------
    captured = 0
    for i, (pos, quat) in enumerate(poses):
        print(f"\n[MOVE {i+1}/{NUM_POSES}] pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        ok = robot.move_to_pose(pos, quat)
        if not ok:
            print(f"[MOVE {i+1}/{NUM_POSES}] IK FAILED — skipping")
            continue

        # give the camera 2-3 frames AFTER we are stable at pose
        for _ in range(3):
            rclpy.spin_once(robot, timeout_sec=0.1)

        img = robot.capture()

        if img is None:
            print(f"[WARN] No image at pose {i}")
            continue

        path = os.path.join(output_dir, f"pose_{i:02d}.png")
        cv2.imwrite(path, img)
        print(f"[CAPTURE] pose_{i:02d}.png  ({img.shape[1]}x{img.shape[0]})")
        captured += 1

    # ----------------------------------------------------------------------
    # STEP 5: Return HOME
    # ----------------------------------------------------------------------
    print("\n[STEP 5] Returning to HOME …")
    robot.move_to_pose(HOME_POS, home_quat)

    # Summary
    print("\n" + "=" * 60)
    print(f"  DONE  {captured}/{NUM_POSES} images captured")
    print(f"  {output_dir}")
    print("=" * 60)

    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# ur5e_orbit_capture.py
# =====================
# Moves the UR5e end-effector to 8 positions around a cube, captures an image
# at each position.

# HOW IT WORKS:
#   - GetPositionIK service computes joint angles for each target pose.
#   - Joint trajectory is smoothly interpolated (smoothstep) between current
#     and target joint positions.
#   - Each interpolated step is published as a JointState to /joint_command —
#     exactly what Isaac Sim's articulation_controller listens to.
#   - No MoveIt planning pipeline, no controller manager, no action servers.

# WHY NOT MoveIt plan()/move_to_pose():
#   Isaac Sim does not run a FollowJointTrajectory action server.
#   MoveIt's controller manager cannot execute anything.
#   The target_cube collision mesh OBJ causes false collisions with
#   upper_arm_link that cannot be resolved by the OMPL planner.
#   Direct IK + manual interpolation is the correct approach for Isaac Sim.

# HOW TO RUN:
#   Terminal 1:  cd isaacsim/_build/linux-x86_64/release && ./isaac-sim.sh
#                -> Open ur5e_w_cam_w_plane.usd -> Press PLAY
#   Terminal 2:  conda deactivate && ros2 launch ur5e_wr_moveit_config move_group.launch.py
#   Terminal 3:  conda deactivate && python3 ur5e_orbit_capture.py
# """

# import os
# import sys
# import math
# import time
# import threading
# import numpy as np
# from datetime import datetime
# from typing import List, Tuple, Optional

# import rclpy
# from rclpy.node import Node
# from rclpy.executors import SingleThreadedExecutor
# from sensor_msgs.msg import Image, JointState
# from geometry_msgs.msg import PoseStamped, Pose
# from moveit_msgs.srv import GetPositionIK
# from cv_bridge import CvBridge
# import cv2


# # ==============================================================================
# #  CONFIGURATION
# # ==============================================================================

# JOINT_NAMES = [
#     "shoulder_pan_joint",
#     "shoulder_lift_joint",
#     "elbow_joint",
#     "wrist_1_joint",
#     "wrist_2_joint",
#     "wrist_3_joint",
# ]

# # IK parameters (must match working code)
# IK_GROUP      = "arm"
# IK_LINK       = "wrist_3_link"
# IK_FRAME      = "World"          # critical: Isaac Sim uses "World", not "ur5e_base_link"
# IK_TIMEOUT    = 5                # seconds

# # --- Environment geometry ---
# CUBE_CENTER = np.array([0.43092, 0.12914, 0.05145])
# CUBE_SIZE   = np.array([0.2, 0.2, 0.2])
# CUBE_TOP_Z  = CUBE_CENTER[2] + CUBE_SIZE[2] / 2   # 0.15145

# # --- Orbit ---
# ORBIT_RADIUS = 0.30
# ORBIT_HEIGHT = 0.45
# ORBIT_Z      = CUBE_TOP_Z + ORBIT_HEIGHT   # 0.45145
# NUM_POSES    = 8

# # --- Topics / services ---
# CAMERA_TOPIC  = "/camera/image_raw"
# COMMAND_TOPIC = "/joint_command"
# IK_SERVICE    = "/compute_ik"

# # --- Output ---
# CAPTURE_BASE = "/home/AP_PathMatters/path_matters/trajectory/captures"

# # --- Timing ---
# INTERP_STEPS       = 40      # waypoints between current and target
# INTERP_STEP_TIME   = 0.05    # seconds per step (40 steps × 0.05s = 2s per move)
# WAIT_AT_POSE       = 2.0     # seconds to hold still before capture
# JOINT_STATE_TIMEOUT = 15.0

# # which joint to flip to fix camera roll (your case: wrist_2_joint)
# FLIP_JOINT_INDEX = 4  # wrist_2_joint
# FLIP_ANGLE = math.pi  # 180°


# # ==============================================================================
# #  MATH / GEOMETRY
# # ==============================================================================

# def smoothstep(t: float) -> float:
#     return 3.0 * t * t - 2.0 * t * t * t


# def interpolate_joints(start: List[float], end: List[float], steps: int) -> List[List[float]]:
#     traj = []
#     for i in range(steps + 1):
#         t = smoothstep(i / steps)
#         traj.append([s + t * (e - s) for s, e in zip(start, end)])
#     return traj


# def make_continuous(target: float, reference: float) -> float:
#     """
#     Pick an equivalent angle (±2πk) closest to reference to avoid wrap spins.
#     """
#     while target - reference > math.pi:
#         target -= 2.0 * math.pi
#     while target - reference < -math.pi:
#         target += 2.0 * math.pi
#     return target


# def rotation_matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
#     trace = R[0, 0] + R[1, 1] + R[2, 2]
#     if trace > 0:
#         s = 0.5 / math.sqrt(trace + 1.0)
#         w = 0.25 / s
#         x = (R[2, 1] - R[1, 2]) * s
#         y = (R[0, 2] - R[2, 0]) * s
#         z = (R[1, 0] - R[0, 1]) * s
#     elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
#         s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
#         w = (R[2, 1] - R[1, 2]) / s
#         x = 0.25 * s
#         y = (R[0, 1] + R[1, 0]) / s
#         z = (R[0, 2] + R[2, 0]) / s
#     elif R[1, 1] > R[2, 2]:
#         s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
#         w = (R[0, 2] - R[2, 0]) / s
#         x = (R[0, 1] + R[1, 0]) / s
#         y = 0.25 * s
#         z = (R[1, 2] + R[2, 1]) / s
#     else:
#         s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
#         w = (R[1, 0] - R[0, 1]) / s
#         x = (R[0, 2] + R[2, 0]) / s
#         y = (R[1, 2] + R[2, 1]) / s
#         z = 0.25 * s
#     q = np.array([x, y, z, w])
#     return q / np.linalg.norm(q)


# def look_at_quaternion(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
#     """
#     Look-at with FIXED ROLL (stable "human taking photos" feel).

#     - forward (local -Z) points at target
#     - roll is fixed by forcing the tool's local +X ("right") to align with a
#       constant world reference projected onto the view plane.
#     """
#     forward = target - eye
#     nf = np.linalg.norm(forward)
#     if nf < 1e-9:
#         return np.array([0.0, 0.0, 0.0, 1.0])
#     forward = forward / nf

#     # local -Z points forward => local +Z points backwards
#     z_axis = -forward  # this is the tool's local +Z in world coords

#     # choose a constant world reference for "right"
#     # (world +Y works well for orbiting; you can switch to +X if you prefer)
#     world_ref = np.array([0.0, 1.0, 0.0])

#     # project world_ref onto plane perpendicular to z_axis (to define roll)
#     x_axis = world_ref - np.dot(world_ref, z_axis) * z_axis
#     nx = np.linalg.norm(x_axis)

#     # if degenerate (z almost parallel to world_ref), fall back to world +X
#     if nx < 1e-6:
#         world_ref = np.array([1.0, 0.0, 0.0])
#         x_axis = world_ref - np.dot(world_ref, z_axis) * z_axis
#         nx = np.linalg.norm(x_axis)
#         if nx < 1e-6:
#             return np.array([0.0, 0.0, 0.0, 1.0])

#     x_axis = x_axis / nx
#     y_axis = np.cross(z_axis, x_axis)
#     y_axis = y_axis / np.linalg.norm(y_axis)

#     R = np.column_stack([x_axis, y_axis, z_axis])
#     return rotation_matrix_to_quat_xyzw(R)



# def compute_orbit_poses(center: np.ndarray, radius: float, z: float, n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
#     poses = []
#     for i in range(n):
#         angle = i * (2.0 * math.pi / n)
#         pos = np.array([
#             center[0] + radius * math.cos(angle),
#             center[1] + radius * math.sin(angle),
#             z,
#         ])
#         quat = look_at_quaternion(pos, center)
#         poses.append((pos, quat))
#     return poses

# # ==============================================================================
# #  ROBOT NODE — single node that does IK, publishes commands, subscribes to
# #               joint_states and camera. Spun by ONE executor in main thread
# #               via rclpy.spin_once().  No background threads on this node.
# # ==============================================================================

# class RobotNode(Node):
#     def __init__(self):
#         super().__init__("robot_node")

#         # --- IK client ---
#         self.ik_client = self.create_client(GetPositionIK, IK_SERVICE)

#         # --- command publisher ---
#         self.cmd_pub = self.create_publisher(JointState, COMMAND_TOPIC, qos_profile=10)

#         # --- joint state subscriber ---
#         self.current_joints: Optional[List[float]] = None
#         self._js_sub = self.create_subscription(
#             JointState, "/joint_states", self._js_callback, qos_profile=10
#         )

#         # --- camera subscriber ---
#         self._bridge     = CvBridge()
#         self._latest_img = None
#         self._img_lock   = threading.Lock()
#         self._img_sub = self.create_subscription(
#             Image, CAMERA_TOPIC, self._img_callback, qos_profile=10
#         )

#     # ── callbacks ──────────────────────────────────────────────────────────
#     def _js_callback(self, msg: JointState):
#         # Extract only our 6 joints in order
#         positions = []
#         for jn in JOINT_NAMES:
#             if jn in msg.name:
#                 positions.append(msg.position[list(msg.name).index(jn)])
#             else:
#                 return   # message doesn't have all our joints, skip
#         self.current_joints = positions

#     def _img_callback(self, msg: Image):
#         try:
#             img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
#             with self._img_lock:
#                 self._latest_img = img
#         except Exception as e:
#             self.get_logger().error(f"CvBridge: {e}")

#     def capture(self) -> Optional[np.ndarray]:
#         with self._img_lock:
#             return self._latest_img.copy() if self._latest_img is not None else None

#     # ── IK ─────────────────────────────────────────────────────────────────
#     def compute_ik(
#         self, position: np.ndarray, quat_xyzw: np.ndarray
#     ) -> Optional[List[float]]:
#         """
#         Call /compute_ik service. Returns 6 joint positions or None.
#         Caller must have already spun this node so current_joints is fresh.
#         """
#         if self.current_joints is None:
#             self.get_logger().error("compute_ik: no current_joints available")
#             return None

#         pose = PoseStamped()
#         pose.header.frame_id = IK_FRAME
#         pose.header.stamp    = self.get_clock().now().to_msg()
#         pose.pose.position.x = float(position[0])
#         pose.pose.position.y = float(position[1])
#         pose.pose.position.z = float(position[2])
#         pose.pose.orientation.x = float(quat_xyzw[0])
#         pose.pose.orientation.y = float(quat_xyzw[1])
#         pose.pose.orientation.z = float(quat_xyzw[2])
#         pose.pose.orientation.w = float(quat_xyzw[3])

#         req = GetPositionIK.Request()
#         req.ik_request.group_name  = IK_GROUP
#         req.ik_request.ik_link_name = IK_LINK
#         req.ik_request.pose_stamped = pose
#         req.ik_request.timeout.sec  = IK_TIMEOUT
#         req.ik_request.avoid_collisions = True
#         req.ik_request.robot_state.joint_state.name     = JOINT_NAMES
#         req.ik_request.robot_state.joint_state.position = self.current_joints
#         req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

#         future = self.ik_client.call_async(req)
#         # Spin until the future completes (this pumps all callbacks on this node)
#         rclpy.spin_until_future_complete(self, future, timeout_sec=IK_TIMEOUT + 2)

#         if not future.done():
#             self.get_logger().error("IK service timed out")
#             return None

#         response = future.result()
#         if response.error_code.val != 1:   # 1 = SUCCESS in MoveItErrorCodes
#             self.get_logger().error(f"IK failed, error code: {response.error_code.val}")
#             return None

#         # Extract the 6 joints we care about from the IK solution
#         sol = response.solution.joint_state
#         result = []
#         for jn in JOINT_NAMES:
#             if jn in sol.name:
#                 result.append(sol.position[list(sol.name).index(jn)])
#             else:
#                 self.get_logger().error(f"IK solution missing joint '{jn}'")
#                 return None
#         return result

#     # ── motion ─────────────────────────────────────────────────────────────
#     def execute_trajectory(self, trajectory: List[List[float]]):
#         """Publish each waypoint as JointState to /joint_command."""
#         for positions in trajectory:
#             msg = JointState()
#             msg.header.stamp = self.get_clock().now().to_msg()
#             msg.name     = JOINT_NAMES
#             msg.position = positions
#             self.cmd_pub.publish(msg)
#             time.sleep(INTERP_STEP_TIME)


# # ==============================================================================
# #  MAIN
# # ==============================================================================

# def main():
#     rclpy.init()

#     # ── output folder ──────────────────────────────────────────────────────
#     timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = os.path.join(CAPTURE_BASE, timestamp)
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"[OUTPUT] {output_dir}")

#     # ── single robot node ──────────────────────────────────────────────────
#     robot = RobotNode()

#     # ── wait for IK service ────────────────────────────────────────────────
#     print("[INIT] Waiting for IK service …")
#     while not robot.ik_client.wait_for_service(timeout_sec=1.0):
#         print("[INIT]   still waiting …")
#     print("[INIT] ✓ IK service available")

#     # ── wait for joint_states ──────────────────────────────────────────────
#     print("[INIT] Waiting for /joint_states …")
#     deadline = time.time() + JOINT_STATE_TIMEOUT
#     while robot.current_joints is None and time.time() < deadline:
#         rclpy.spin_once(robot, timeout_sec=0.1)
#     if robot.current_joints is None:
#         print("[ERROR] No /joint_states. Is Isaac Sim playing?")
#         robot.destroy_node()
#         rclpy.shutdown()
#         return
#     print(f"[INIT] ✓ joint_states: [{', '.join(f'{v:.4f}' for v in robot.current_joints)}]")

#     # ── wait for camera ────────────────────────────────────────────────────
#     print("[INIT] Waiting for camera …")
#     deadline = time.time() + 10.0
#     while robot.capture() is None and time.time() < deadline:
#         rclpy.spin_once(robot, timeout_sec=0.1)
#     if robot.capture() is None:
#         print("[ERROR] No camera image. Check /camera/image_raw")
#         robot.destroy_node()
#         rclpy.shutdown()
#         return
#     print("[INIT] ✓ camera ready")

#     # ── compute orbit poses ────────────────────────────────────────────────
#     poses = compute_orbit_poses(
#         center=CUBE_CENTER, radius=ORBIT_RADIUS, z=ORBIT_Z, n=NUM_POSES
#     )
#     print(f"\n[POSES] {NUM_POSES} poses at z={ORBIT_Z:.4f} m")
#     for i, (pos, quat) in enumerate(poses):
#         print(f"  {i}: {i*(360//NUM_POSES):>4d} deg  "
#               f"pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})  "
#               f"q=({quat[0]:.3f},{quat[1]:.3f},{quat[2]:.3f},{quat[3]:.3f})")

#     # ── main loop: IK → interpolate → publish → capture ───────────────────
#     captured = 0
#     for i, (pos, quat) in enumerate(poses):
#         print(f"\n[MOVE {i+1}/{NUM_POSES}] pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

#         # Spin to make sure current_joints is fresh
#         rclpy.spin_once(robot, timeout_sec=0.1)

#         # --- IK ---
#         print(f"[MOVE {i+1}/{NUM_POSES}] Computing IK …")
#         target_joints = robot.compute_ik(pos, quat)
#         if target_joints is None:
#             print(f"[MOVE {i+1}/{NUM_POSES}] IK FAILED — skipping")
#             continue
#         print(f"[MOVE {i+1}/{NUM_POSES}] IK OK: [{', '.join(f'{v:.4f}' for v in target_joints)}]")


#         # 🔥 FIX: flip camera via wrist_2_joint (index 4) and keep it continuous
#         target_joints[4] = target_joints[4] + math.pi
#         target_joints[4] = make_continuous(target_joints[4], robot.current_joints[4])
        

#         # --- interpolate ---
#         trajectory = interpolate_joints(robot.current_joints, target_joints, INTERP_STEPS)

#         # --- execute ---
#         print(f"[MOVE {i+1}/{NUM_POSES}] Executing {len(trajectory)} waypoints …")
#         robot.execute_trajectory(trajectory)

#         # Update current_joints to the target (we just commanded it there)
#         robot.current_joints = target_joints
#         print(f"[MOVE {i+1}/{NUM_POSES}] Done")

#         # --- wait for settling ---
#         time.sleep(WAIT_AT_POSE)

#         # Spin to get a fresh image
#         rclpy.spin_once(robot, timeout_sec=0.1)

#         # --- capture ---
#         img = robot.capture()
#         if img is None:
#             print(f"[WARN] No image at pose {i}")
#             continue

#         path = os.path.join(output_dir, f"pose_{i:02d}.png")
#         cv2.imwrite(path, img)
#         print(f"[CAPTURE] pose_{i:02d}.png  ({img.shape[1]}x{img.shape[0]})")
#         captured += 1

#     # ── summary ────────────────────────────────────────────────────────────
#     print("\n" + "=" * 60)
#     print(f"  DONE  {captured}/{NUM_POSES} images captured")
#     print(f"  {output_dir}")
#     print("=" * 60)

#     robot.destroy_node()
#     rclpy.shutdown()


# if __name__ == "__main__":
#     main()