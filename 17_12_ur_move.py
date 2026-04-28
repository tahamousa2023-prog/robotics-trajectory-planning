#!/usr/bin/env python3
"""
UR5e Cartesian Control with FULL Collision Avoidance
"""

import rclpy
from rclpy.node import Node
import time, math
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest, RobotState
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class URControl(Node):
    def __init__(self):
        super().__init__('ur_control')
        
        # IK service
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        
        # Joint command publisher
        self.pub = self.create_publisher(JointState, '/joint_commands', 10)
        
        # UR5e joint names
        self.joints = [
            'shoulder_pan_joint', 
            'shoulder_lift_joint', 
            'elbow_joint',
            'wrist_1_joint', 
            'wrist_2_joint', 
            'wrist_3_joint'
        ]
        
        # Current joint state
        self.current_state = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        
        # Wait for IK service
        while not self.ik.wait_for_service(timeout_sec=1):
            self.get_logger().info("Waiting for /compute_ik service...")
        
        self.get_logger().info("✓ MoveIt IK service ready!")
        time.sleep(1.0)
    
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
    
    def goto(self, x, y, z, rx=0, ry=0, rz=0):
        """Move end-effector to XYZ position with orientation"""
        
        constraint_error = self.check_workspace_limits(x, y, z)
        if constraint_error:
            self.get_logger().error(f"✗ Position blocked: {constraint_error}")
            self.get_logger().info(f"  Requested: ({x:.3f}, {y:.3f}, {z:.3f})")
            return False
        
        self.get_logger().info(f"→ Planning move to ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Create target pose
        pose = PoseStamped()
        pose.header.frame_id = "World"
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        qx, qy, qz, qw = self.euler_to_quat(rx, ry, rz)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw
        
        # Create IK request
        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"  # FIXED: Changed from ur_manipulator to arm
        req.ik_request.ik_link_name = "wrist_3_link"  # TEMPORARY: Try wrist_3_link first
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 2
        req.ik_request.avoid_collisions = True
        
        # Provide robot state seed
        req.ik_request.robot_state.joint_state.name = self.joints
        req.ik_request.robot_state.joint_state.position = self.current_state
        req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()

        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5)
        
        try:
            response = future.result()
            
            if response.error_code.val == 1:  # SUCCESS
                angles = list(response.solution.joint_state.position[:6])
                
                self.get_logger().info("  ✓ IK solution found")
                self.get_logger().info(f"  ✓ Collision check: PASSED")
                self.get_logger().info(f"  ✓ Joint angles: {[f'{math.degrees(a):.1f}°' for a in angles]}")
                
                self.current_state = angles
                
                # Send to robot
                msg = JointState()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.name = self.joints
                msg.position = angles
                
                self.pub.publish(msg)
                self.get_logger().info("  ✓ Command sent to Isaac Sim")
                
                return True
                
            else:
                error_codes = {
                    -1: "FAILURE",
                    -2: "PLANNING_FAILED", 
                    -3: "INVALID_GOAL_CONSTRAINTS",
                    -10: "COLLISION_DETECTED",
                    -11: "START_STATE_IN_COLLISION",
                    -12: "GOAL_IN_COLLISION",
                    -18: "INVALID_ROBOT_STATE",
                    -31: "NO_IK_SOLUTION"
                }
                
                error_name = error_codes.get(response.error_code.val, f"UNKNOWN({response.error_code.val})")
                self.get_logger().error(f"  ✗ IK failed: {error_name}")
                
                if response.error_code.val in [-10, -12]:
                    self.get_logger().error("  → Position would cause COLLISION")
                elif response.error_code.val == -31:
                    self.get_logger().error("  → Position is UNREACHABLE")
                elif response.error_code.val == -18:
                    self.get_logger().error("  → Robot state is invalid")
                
                return False
                
        except Exception as e:
            self.get_logger().error(f"  ✗ IK service error: {e}")
            return False
    
    @staticmethod
    def check_workspace_limits(x, y, z):
        """Pre-check workspace limits"""
        MIN_HEIGHT = 0.15
        if z < MIN_HEIGHT:
            return f"TOO LOW! Z={z:.3f}m (minimum: {MIN_HEIGHT}m)"
        
        X_MIN, X_MAX = -0.05, 0.85
        Y_MIN, Y_MAX = -0.85, 0.85  
        Z_MAX = 1.2
        
        if not (X_MIN <= x <= X_MAX):
            return f"X out of range: {x:.3f}m"
        if not (Y_MIN <= y <= Y_MAX):
            return f"Y out of range: {y:.3f}m"
        if z > Z_MAX:
            return f"TOO HIGH! Z={z:.3f}m"
        
        return None


def main():
    try:
        rclpy.init()
        ur = URControl()
        
        print("\n" + "="*70)
        print("UR5e Cartesian Control - Full Collision Avoidance")
        print("="*70)
        
        HOME = {'x': 0.3, 'y': 0.0, 'z': 0.5}
        print(f"\n🏠 HOME: ({HOME['x']:.2f}, {HOME['y']:.2f}, {HOME['z']:.2f})")
        
        print("\n→ Moving to HOME position...")
        if not ur.goto(HOME['x'], HOME['y'], HOME['z']):
            print("✗ Failed to reach HOME")
            return
        
        time.sleep(2)
        
        positions = [
            {'name': 'Safe High', 'x': 0.4, 'y': 0.2, 'z': 0.6},
            {'name': 'Safe Low', 'x': 0.35, 'y': -0.15, 'z': 0.25},
            {'name': 'TOO LOW', 'x': 0.3, 'y': 0.0, 'z': 0.05},
        ]
        
        for pos in positions:
            print(f"\n{'='*70}")
            print(f"🎯 Testing: {pos['name']}")
            print(f"{'='*70}")
            
            success = ur.goto(pos['x'], pos['y'], pos['z'])
            
            if success:
                print(f"✓ Reached target")
                time.sleep(2)
                print("→ Returning to HOME...")
                ur.goto(HOME['x'], HOME['y'], HOME['z'])
                time.sleep(2)
            else:
                print(f"✗ Target rejected")
                time.sleep(1)
        
        print(f"\n{'='*70}")
        print("✓ Demo complete!")
        print(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3
# """
# UR5e Cartesian Control with FULL Collision Avoidance
# - Ground plane collision checking
# - Self-collision checking  
# - Workspace limits
# """

# import rclpy
# from rclpy.node import Node
# import time, math
# from moveit_msgs.srv import GetPositionIK
# from moveit_msgs.msg import PositionIKRequest, RobotState, Constraints, JointConstraint
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import JointState


# class URControl(Node):
#     def __init__(self):
#         super().__init__('ur_control')
        
#         # IK service
#         self.ik = self.create_client(GetPositionIK, '/compute_ik')
        
#         # Joint command publisher
#         self.pub = self.create_publisher(JointState, '/isaac_joint_command', 10)
        
#         # UR5e joint names
#         self.joints = [
#             'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
#             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
#         ]
        
#         # Wait for IK service
#         while not self.ik.wait_for_service(timeout_sec=1):
#             self.get_logger().info("Waiting for /compute_ik service...")
        
#         self.get_logger().info("✓ MoveIt IK service ready!")
    
#     @staticmethod
#     def euler_to_quat(rx, ry, rz):
#         """Convert Euler angles (radians) to quaternion"""
#         cy, sy = math.cos(rz*0.5), math.sin(rz*0.5)
#         cp, sp = math.cos(ry*0.5), math.sin(ry*0.5)
#         cr, sr = math.cos(rx*0.5), math.sin(rx*0.5)
        
#         w = cr*cp*cy + sr*sp*sy
#         x = sr*cp*cy - cr*sp*sy
#         y = cr*sp*cy + sr*cp*sy
#         z = cr*cp*sy - sr*sp*cy
        
#         return x, y, z, w
    
#     def goto(self, x, y, z, rx=0, ry=0, rz=0):
#         """
#         Move end-effector to XYZ position with orientation
        
#         Args:
#             x, y, z: Position in meters (relative to world frame)
#             rx, ry, rz: Orientation in radians (Euler angles)
        
#         Returns:
#             bool: True if successful, False if failed
#         """
        
#         # Basic workspace validation
#         constraint_error = self.check_workspace_limits(x, y, z)
#         if constraint_error:
#             self.get_logger().error(f"✗ Position blocked: {constraint_error}")
#             self.get_logger().info(f"  Requested: ({x:.3f}, {y:.3f}, {z:.3f})")
#             return False
        
#         self.get_logger().info(f"→ Planning move to ({x:.3f}, {y:.3f}, {z:.3f})")
        
#         # Create target pose
#         pose = PoseStamped()
#         pose.header.frame_id = "world"
#         pose.header.stamp = self.get_clock().now().to_msg()
        
#         pose.pose.position.x = float(x)
#         pose.pose.position.y = float(y)
#         pose.pose.position.z = float(z)
        
#         qx, qy, qz, qw = self.euler_to_quat(rx, ry, rz)
#         pose.pose.orientation.x = qx
#         pose.pose.orientation.y = qy
#         pose.pose.orientation.z = qz
#         pose.pose.orientation.w = qw
        
#         # Create IK request
#         req = GetPositionIK.Request()
#         req.ik_request.group_name = "ur_manipulator"
#         req.ik_request.ik_link_name = "tool0"
#         req.ik_request.pose_stamped = pose
#         req.ik_request.timeout.sec = 2
#         req.ik_request.avoid_collisions = True  # Enable collision checking

#         future = self.ik.call_async(req)
#         rclpy.spin_until_future_complete(self, future, timeout_sec=3)
        
#         try:
#             response = future.result()
            
#             # Check if IK found a solution
#             if response.error_code.val == 1:  # SUCCESS
#                 angles = list(response.solution.joint_state.position[:6])
                
#                 self.get_logger().info("  ✓ IK solution found")
#                 self.get_logger().info(f"  ✓ Collision check: PASSED")
#                 self.get_logger().info(f"  ✓ Joint angles: {[f'{math.degrees(a):.1f}°' for a in angles]}")
                
#                 # Send to robot
#                 msg = JointState()
#                 msg.header.stamp = self.get_clock().now().to_msg()
#                 msg.name = self.joints
#                 msg.position = angles
                
#                 self.pub.publish(msg)
#                 self.get_logger().info("  ✓ Command sent to Isaac Sim")
                
#                 return True
                
#             else:
#                 error_codes = {
#                     -1: "FAILURE",
#                     -2: "PLANNING_FAILED", 
#                     -3: "INVALID_GOAL_CONSTRAINTS",
#                     -4: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE",
#                     -5: "CONTROL_FAILED",
#                     -10: "COLLISION_DETECTED",
#                     -11: "START_STATE_IN_COLLISION",
#                     -12: "GOAL_IN_COLLISION",
#                     -31: "NO_IK_SOLUTION"
#                 }
                
#                 error_name = error_codes.get(response.error_code.val, f"UNKNOWN({response.error_code.val})")
                
#                 self.get_logger().error(f"  ✗ IK failed: {error_name}")
                
#                 if response.error_code.val in [-10, -12]:
#                     self.get_logger().error("  → Position would cause COLLISION")
#                 elif response.error_code.val == -31:
#                     self.get_logger().error("  → Position is UNREACHABLE (outside workspace)")
                
#                 return False
                
#         except Exception as e:
#             self.get_logger().error(f"  ✗ IK service error: {e}")
#             return False
    
#     @staticmethod
#     def check_workspace_limits(x, y, z):
#         """Pre-check workspace limits (before calling IK)"""
        
#         # Ground collision (must be above ground)
#         MIN_HEIGHT = 0.15  # 15cm clearance above ground
#         if z < MIN_HEIGHT:
#             return f"TOO LOW! Z={z:.3f}m (minimum: {MIN_HEIGHT}m) - GROUND COLLISION"
        
#         # UR5e workspace limits (conservative)
#         X_MIN, X_MAX = -0.05, 0.85
#         Y_MIN, Y_MAX = -0.85, 0.85  
#         Z_MAX = 1.2
        
#         if not (X_MIN <= x <= X_MAX):
#             return f"X out of range: {x:.3f}m (valid: {X_MIN} to {X_MAX}m)"
        
#         if not (Y_MIN <= y <= Y_MAX):
#             return f"Y out of range: {y:.3f}m (valid: {Y_MIN} to {Y_MAX}m)"
        
#         if z > Z_MAX:
#             return f"TOO HIGH! Z={z:.3f}m (maximum: {Z_MAX}m)"
        
#         return None


# def main():
#     rclpy.init()
#     ur = URControl()
    
#     print("\n" + "="*70)
#     print("UR5e Cartesian Control - Full Collision Avoidance")
#     print("="*70)
#     print("\n📋 COLLISION CHECKING ENABLED:")
#     print("   ✓ Ground plane collision")
#     print("   ✓ Self-collision (robot links)")
#     print("   ✓ Workspace limits")
#     print("   ✓ MoveIt planning scene objects")
    
#     print("\n📍 WORKSPACE LIMITS:")
#     print("   Ground clearance: Z >= 0.15m")
#     print("   Reachable area:   X: -0.05 to 0.85m")
#     print("                     Y: -0.85 to 0.85m")
#     print("                     Z:  0.15 to 1.20m")
    
#     # Home position
#     HOME = {'x': 0.3, 'y': 0.0, 'z': 0.5}
    
#     print(f"\n🏠 HOME: ({HOME['x']:.2f}, {HOME['y']:.2f}, {HOME['z']:.2f})")
    
#     try:
#         # Go home
#         print("\n→ Moving to HOME position...")
#         if not ur.goto(HOME['x'], HOME['y'], HOME['z']):
#             print("✗ Failed to reach HOME")
#             rclpy.shutdown()
#             return
        
#         time.sleep(2)
        
#         # Test positions
#         positions = [
#             {'name': 'Safe High', 'x': 0.4, 'y': 0.2, 'z': 0.6},
#             {'name': 'Safe Low', 'x': 0.35, 'y': -0.15, 'z': 0.25},
#             {'name': 'TOO LOW (should fail)', 'x': 0.3, 'y': 0.0, 'z': 0.05},  # Below ground clearance
#         ]
        
#         for pos in positions:
#             print(f"\n{'='*70}")
#             print(f"🎯 Testing: {pos['name']}")
#             print(f"{'='*70}")
            
#             success = ur.goto(pos['x'], pos['y'], pos['z'])
            
#             if success:
#                 print(f"✓ Reached target successfully")
#                 time.sleep(2)
                
#                 # Return home
#                 print("→ Returning to HOME...")
#                 ur.goto(HOME['x'], HOME['y'], HOME['z'])
#                 time.sleep(2)
#             else:
#                 print(f"✗ Target rejected (collision or unreachable)")
#                 time.sleep(1)
        
#         print(f"\n{'='*70}")
#         print("✓ Demo complete!")
#         print(f"{'='*70}\n")
        
#     except KeyboardInterrupt:
#         print("\n\n⚠ Interrupted - returning to HOME...")
#         ur.goto(HOME['x'], HOME['y'], HOME['z'])
#         time.sleep(1)
    
#     finally:
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()