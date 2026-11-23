#!/usr/bin/env python3
"""
Quick Position Tester - Find a good HOME position
"""

import rclpy
from rclpy.node import Node
import math
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState


class PositionTester(Node):
    def __init__(self):
        super().__init__('position_tester')
        
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.current_joints = None
        
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', 
            lambda msg: setattr(self, 'current_joints', list(msg.position[:6])),
            10
        )
        
        while not self.ik.wait_for_service(timeout_sec=1):
            pass
        
        # Wait for joint state
        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.5)
            if self.current_joints:
                break
        
        if not self.current_joints:
            self.current_joints = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
    
    def test_position(self, x, y, z):
        """Test if position is reachable"""
        pose = PoseStamped()
        pose.header.frame_id = "World"
        pose.header.stamp = self.get_clock().now().to_msg()
        
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.position.z = float(z)
        
        # Pointing down
        pose.pose.orientation.x = 1.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 0.0
        
        req = GetPositionIK.Request()
        req.ik_request.group_name = "arm"
        req.ik_request.ik_link_name = "wrist_3_link"
        req.ik_request.pose_stamped = pose
        req.ik_request.timeout.sec = 5
        req.ik_request.avoid_collisions = True
        req.ik_request.robot_state.joint_state.name = [
            'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
        ]
        req.ik_request.robot_state.joint_state.position = self.current_joints
        req.ik_request.robot_state.joint_state.header.stamp = self.get_clock().now().to_msg()
        
        future = self.ik.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=7)
        
        try:
            response = future.result()
            return response.error_code.val == 1
        except:
            return False


def main():
    rclpy.init()
    tester = PositionTester()
    
    print("\n" + "="*70)
    print("TESTING REACHABLE POSITIONS")
    print("="*70)
    
    # Test various positions
    test_positions = [
        {'name': 'Original HOME', 'x': 0.4, 'y': 0.0, 'z': 0.5},
        {'name': 'Higher HOME', 'x': 0.4, 'y': 0.0, 'z': 0.6},
        {'name': 'Closer HOME', 'x': 0.3, 'y': 0.0, 'z': 0.5},
        {'name': 'Safe HIGH', 'x': 0.3, 'y': 0.0, 'z': 0.7},
        {'name': 'Front-Low', 'x': 0.25, 'y': 0.0, 'z': 0.3},
        {'name': 'Front-Mid', 'x': 0.3, 'y': 0.0, 'z': 0.4},
        {'name': 'Side-Right', 'x': 0.2, 'y': 0.3, 'z': 0.4},
        {'name': 'Side-Left', 'x': 0.2, 'y': -0.3, 'z': 0.4},
    ]
    
    reachable = []
    
    for pos in test_positions:
        result = tester.test_position(pos['x'], pos['y'], pos['z'])
        status = "✓ REACHABLE" if result else "✗ UNREACHABLE"
        print(f"  {pos['name']:20s} ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}): {status}")
        
        if result:
            reachable.append(pos)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    if reachable:
        print(f"\n✓ Found {len(reachable)} reachable position(s)\n")
        print("Use one of these as HOME:")
        for pos in reachable[:3]:  # Show top 3
            print(f"  HOME = {{'x': {pos['x']}, 'y': {pos['y']}, 'z': {pos['z']}}}")
    else:
        print("\n✗ No reachable positions found!")
        print("   Check your MoveIt configuration")
    
    print("\n" + "="*70)
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()