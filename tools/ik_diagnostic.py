#!/usr/bin/env python3
"""
MoveIt IK Diagnostic - Find the correct configuration
"""

import rclpy
from rclpy.node import Node
import math
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class IKDiagnostic(Node):
    def __init__(self):
        super().__init__('ik_diagnostic')
        
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        
        self.joints = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        
        self.current_joints = None
        
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10
        )
        
        self.get_logger().info("Waiting for IK service...")
        self.ik.wait_for_service(timeout_sec=10)
        self.get_logger().info("✓ IK service ready")
    
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
    
    def test_ik(self, x, y, z, rx, ry, rz, frame_id, description):
        """Test IK with specific parameters"""
        pose = PoseStamped()
        pose.header.frame_id = frame_id
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
        req.ik_request.avoid_collisions = False
        
        if self.current_joints:
            req.ik_request.robot_state.joint_state.name = self.joints
            req.ik_request.robot_state.joint_state.position = self.current_joints
        
        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=6)
        
        try:
            response = future.result()
            if response.error_code.val == 1:
                return True, "SUCCESS"
            else:
                return False, f"ERROR {response.error_code.val}"
        except Exception as e:
            return False, f"EXCEPTION: {e}"


def main():
    rclpy.init()
    diag = IKDiagnostic()
    
    print("\n" + "="*70)
    print("MoveIt IK Diagnostic Tool")
    print("="*70)
    
    # Wait for joint states
    print("\n⏳ Waiting for joint states...")
    import time
    timeout = 5
    while diag.current_joints is None and timeout > 0:
        rclpy.spin_once(diag, timeout_sec=0.5)
        timeout -= 0.5
    
    if diag.current_joints:
        print(f"✓ Current joints: {[f'{j:.2f}' for j in diag.current_joints]}")
    else:
        print("⚠ No joint states received")
    
    # Test different configurations
    print("\n" + "="*70)
    print("Testing different reference frames and orientations:")
    print("="*70)
    
    # Simple reachable position in front of robot
    test_position = (0.3, 0.0, 0.4)
    
    tests = [
        # (frame_id, rx, ry, rz, description)
        ("base_link", math.pi, 0, 0, "base_link, tool pointing down"),
        ("base_link", 0, math.pi/2, 0, "base_link, tool pointing forward"),
        ("base_link", 0, 0, 0, "base_link, tool pointing up"),
        ("world", math.pi, 0, 0, "world, tool pointing down"),
        ("world", 0, 0, 0, "world, tool pointing up"),
        ("base", math.pi, 0, 0, "base, tool pointing down"),
        ("base", 0, 0, 0, "base, tool pointing up"),
    ]
    
    print(f"\nTest position: ({test_position[0]}, {test_position[1]}, {test_position[2]})\n")
    
    success_count = 0
    for frame_id, rx, ry, rz, desc in tests:
        success, result = diag.test_ik(
            test_position[0], test_position[1], test_position[2],
            rx, ry, rz, frame_id, desc
        )
        
        status = "✓ SUCCESS" if success else f"✗ {result}"
        print(f"{status:20} | {desc}")
        
        if success:
            success_count += 1
            print(f"                     → Use: frame='{frame_id}', rx={rx:.2f}, ry={ry:.2f}, rz={rz:.2f}")
    
    print("\n" + "="*70)
    print(f"Results: {success_count}/{len(tests)} configurations worked")
    print("="*70)
    
    if success_count == 0:
        print("\n⚠ WARNING: No IK solutions found!")
        print("\nPossible issues:")
        print("1. MoveIt robot model doesn't match Isaac Sim")
        print("2. Wrong planning group name (try checking with: ros2 param list /move_group)")
        print("3. URDF/robot description mismatch")
        print("\n💡 RECOMMENDATION: Use direct joint control (ur_photo_simple.py)")
    else:
        print("\n✓ At least one configuration works!")
        print("Update your code to use the working frame_id and orientation.")
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()