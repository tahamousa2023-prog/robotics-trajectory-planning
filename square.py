import numpy as np
import time
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.types import ArticulationAction

# ============================================================
# CONFIGURATION
# ============================================================
ROBOT_PATH = "/World/ur5e"
DWELL_TIME = 2.0               # seconds to wait at each corner
STEPS_PER_MOVE = 120           # smoothness between corners

# Square trajectory around object (in meters, relative to object center)
# Adjust these based on your object's position in the scene
OBJECT_CENTER = np.array([0.5, 0.0, 0.0])  # [x, y, z] position of object
SQUARE_SIZE = 0.2              # size of square (meters)
CAMERA_HEIGHT = -0.4            # height above table to position camera

# Define 4 corners of square around the object
SQUARE_OFFSETS = [
    np.array([SQUARE_SIZE/2, SQUARE_SIZE/2, 0]),    # Corner 1: front-right
    np.array([-SQUARE_SIZE/2, SQUARE_SIZE/2, 0]),   # Corner 2: front-left
    np.array([-SQUARE_SIZE/2, -SQUARE_SIZE/2, 0]),  # Corner 3: back-left
    np.array([SQUARE_SIZE/2, -SQUARE_SIZE/2, 0]),   # Corner 4: back-right
]

# ============================================================
# INITIALIZATION
# ============================================================
print("\n--- Initializing World ---")
world = World()
app = omni.kit.app.get_app()

print(f"Checking robot at: {ROBOT_PATH}")

# ------------------------------------------------------------
# SAFE ROBOT LOADING
# ------------------------------------------------------------
existing_prim = get_prim_at_path(ROBOT_PATH)
if existing_prim is None:
    raise RuntimeError(f"❌ Robot not found at {ROBOT_PATH}. Add robot to the stage first.")
else:
    print("✓ Robot exists in stage. Attaching Articulation...")

ur5e = Articulation(prim_path=ROBOT_PATH, name="ur5e_sfdszwesddfgffghdfgasfvsssdfot")
world.scene.add(ur5e)
world.reset()
time.sleep(1)

# ------------------------------------------------------------
# Confirm joints work
# ------------------------------------------------------------
joints = ur5e.get_joint_positions()
dof = len(joints)
print(f"✓ Robot has {dof} DOF")

# ============================================================
# HELPER FUNCTIONS FOR INVERSE KINEMATICS
# ============================================================

def compute_ik_target(position, look_at_point):
    """
    Compute target joint positions to reach a Cartesian position
    while looking at a specific point.
    
    This is a simplified approach. For better results, consider using
    Isaac Sim's built-in IK solver or a proper IK library.
    """
    # Simple analytical IK for UR5e (simplified)
    # You may need to adjust this based on your specific setup
    
    # Calculate direction vector from camera to object
    direction = look_at_point - position
    direction = direction / np.linalg.norm(direction)
    
    # Basic joint angles (this is simplified - you may need proper IK)
    x, y, z = position
    
    # Joint 0: rotation around base (pointing towards object)
    j0 = np.arctan2(y, x)
    
    # Distance in XY plane
    r = np.sqrt(x**2 + y**2)
    
    # Simple shoulder and elbow calculation (simplified IK)
    # UR5e link lengths (approximate)
    L1 = 0.425  # upper arm
    L2 = 0.392  # forearm
    
    # Adjust z for shoulder height
    z_adj = z - 0.089  # shoulder offset
    
    # Distance from shoulder to target
    d = np.sqrt(r**2 + z_adj**2)
    
    # Check if target is reachable
    if d > (L1 + L2):
        print(f"⚠️  Warning: Target may be out of reach (d={d:.3f}m)")
        d = L1 + L2 - 0.01
    
    # Elbow angle using cosine law
    cos_j2 = (d**2 - L1**2 - L2**2) / (2 * L1 * L2)
    cos_j2 = np.clip(cos_j2, -1, 1)
    j2 = np.arccos(cos_j2)
    
    # Shoulder angle
    alpha = np.arctan2(z_adj, r)
    beta = np.arctan2(L2 * np.sin(j2), L1 + L2 * np.cos(j2))
    j1 = alpha - beta
    
    # Wrist angles to point camera down at object
    j3 = -(j1 + j2) - np.pi/2  # keeps end-effector pointing down
    j4 = -np.pi/2  # wrist rotation
    j5 = 0.0  # wrist rotation
    
    return np.array([j0, j1, j2, j3, j4, j5], dtype=np.float32)

def move_to_cartesian_smooth(start_joints, target_position, look_at, steps):
    """Smooth interpolation in Cartesian space."""
    # Get start position (approximate forward kinematics)
    start_pos = get_end_effector_position(ur5e)
    
    for t in np.linspace(0, 1, steps):
        # Interpolate position
        current_pos = start_pos * (1 - t) + target_position * t
        
        # Compute IK for current position
        target_joints = compute_ik_target(current_pos, look_at)
        
        # Pad if necessary
        if dof > len(target_joints):
            target_joints = np.pad(target_joints, (0, dof - len(target_joints)), constant_values=0)
        
        ur5e.set_joint_positions(target_joints)
        app.update()

def get_end_effector_position(robot):
    """
    Get approximate end-effector position using forward kinematics.
    For better accuracy, use Isaac Sim's built-in FK functions.
    """
    # This is a placeholder - you should use proper FK
    # For now, return a reasonable estimate
    joints = robot.get_joint_positions()
    
    # Simplified FK for UR5e
    L1 = 0.425
    L2 = 0.392
    
    j0, j1, j2 = joints[0], joints[1], joints[2]
    
    # XY position
    r = L1 * np.cos(j1) + L2 * np.cos(j1 + j2)
    x = r * np.cos(j0)
    y = r * np.sin(j0)
    
    # Z position
    z = 0.089 + L1 * np.sin(j1) + L2 * np.sin(j1 + j2)
    
    return np.array([x, y, z])

# ============================================================
# MOVE TO HOME POSITION
# ============================================================
print("Moving to home position...")
HOME_POSE = np.deg2rad(np.array([0.0, -90.0, 90.0, -90.0, -90.0, 0.0], dtype=np.float32))

if dof > len(HOME_POSE):
    HOME_POSE = np.pad(HOME_POSE, (0, dof - len(HOME_POSE)), constant_values=0)

for _ in range(100):
    ur5e.set_joint_positions(HOME_POSE)
    app.update()

time.sleep(1)

# ============================================================
# SQUARE TRAJECTORY AROUND OBJECT
# ============================================================
print("\n--- Starting Square Trajectory Around Object ---\n")
print(f"Object center: {OBJECT_CENTER}")
print(f"Square size: {SQUARE_SIZE}m")
print(f"Camera height: {CAMERA_HEIGHT}m\n")

# Calculate absolute positions for each corner
square_positions = []
for offset in SQUARE_OFFSETS:
    pos = OBJECT_CENTER.copy()
    pos[:2] += offset[:2]  # Add XY offset
    pos[2] = CAMERA_HEIGHT  # Set camera height
    square_positions.append(pos)

# Move to first corner
print(f"→ Moving to starting position (Corner 1)")
current_joints = ur5e.get_joint_positions()
move_to_cartesian_smooth(current_joints, square_positions[0], OBJECT_CENTER, STEPS_PER_MOVE)

# Main loop - traverse the square
for idx in range(len(square_positions)):
    current_pos = square_positions[idx]
    
    # Hold at current position
    print(f"   Holding at Corner {idx+1} for {DWELL_TIME}s")
    print(f"   Position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    
    # Compute target joints for this position
    target_joints = compute_ik_target(current_pos, OBJECT_CENTER)
    if dof > len(target_joints):
        target_joints = np.pad(target_joints, (0, dof - len(target_joints)), constant_values=0)
    
    hold_start = time.time()
    while time.time() - hold_start < DWELL_TIME:
        # -----------------------------------------------------
        # CAMERA SNAPSHOT LOGIC HERE
        #
        # Example:
        # synthetic_cam.capture_rgb(f"/path/to/object_view_{idx+1}.png")
        #
        # -----------------------------------------------------
        ur5e.set_joint_positions(target_joints)
        app.update()
    
    # Move to next corner (wraps around to first corner after last)
    next_idx = (idx + 1) % len(square_positions)
    print(f"→ Moving to Corner {next_idx+1}")
    
    current_joints = ur5e.get_joint_positions()
    move_to_cartesian_smooth(current_joints, square_positions[next_idx], 
                            OBJECT_CENTER, STEPS_PER_MOVE)

print("\n✓ Square trajectory complete\n")