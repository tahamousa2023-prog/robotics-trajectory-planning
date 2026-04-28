import numpy as np
import time
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from omni.isaac.core.utils.prims import get_prim_at_path
from datetime import datetime
import asyncio

# ============================================================
# CONFIGURATION
# ============================================================
ROBOT_PATH = "/World/RobotHand/ur5e"

# Square corner positions (in degrees)
SQUARE_POSITIONS = [
    [45.0, -90.0, -90.0, -90.0, 0.0, -90.0, 0.0, 0.0],   # Corner 1
    [90.0, -90.0, -90.0, -90.0, 0.0, -90.0, 0.0, 0.0],   # Corner 2
    [135.0, -90.0, -90.0, -90.0, 0.0, -90.0, 0.0, 0.0],  # Corner 3
    [180.0, -90.0, -90.0, -90.0, 0.0, -90.0, 0.0, 0.0],  # Corner 4
]

# Home position (in degrees)
HOME_POSITION = [60.0, -90.0, -90.0, -90.0, 0.0, -90.0, 0.0, 0.0]

# Timing
DWELL_TIME = 2.0        
STEPS_PER_MOVE = 120    

# ============================================================
# INITIALIZATION
# ============================================================
print("\n" + "="*60)
print("   SQUARE TRAJECTORY - ROBOT MOVEMENT")
print("="*60)

world = World()
app = omni.kit.app.get_app()

# Check robot exists
existing_prim = get_prim_at_path(ROBOT_PATH)
if existing_prim is None:
    raise RuntimeError(f"❌ Robot not found at {ROBOT_PATH}")

# Attach articulation
session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ur5e = Articulation(prim_path=ROBOT_PATH, name=f"ur5e_{session_timestamp}")
world.scene.add(ur5e)
world.reset()

# Allow time for initialization
print("⏱️  Initializing robot...")
for _ in range(60):  # More stable initialization
    app.update()

#dof = len(ur5e.get_joint_positions())
#print(f"✓ Robot has {dof} DOF")
dof = 6

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_update():
    """Safely update the app without triggering asyncio errors."""
    try:
        app.update()
    except (RuntimeError, IndexError):
        # Ignore asyncio errors from background tasks
        pass

def move_to_position(joint_angles_deg, steps=120, description=""):
    """Move robot to specified joint angles smoothly."""
    if description:
        print(f"→ {description}")
    
    # Get current position with retry
    current = None
    for attempt in range(5):
        current = ur5e.get_joint_positions()
        if current is not None and len(current) > 0:
            break
        for _ in range(10):
            safe_update()
    
    if current is None or len(current) == 0:
        print("   ⚠️  Could not get current position, skipping move")
        return
    
    # Target position in radians
    target = np.deg2rad(np.array(joint_angles_deg, dtype=np.float32))
    
    # Pad if necessary
    if dof > len(target):
        target = np.pad(target, (0, dof - len(target)), constant_values=0)
    
    # Smooth interpolation
    for t in np.linspace(0, 1, steps):
        interpolated = current * (1 - t) + target * t
        ur5e.set_joint_positions(interpolated)
        safe_update()

def hold_position(joint_angles_deg, duration):
    """Hold robot at position for specified duration."""
    target = np.deg2rad(np.array(joint_angles_deg, dtype=np.float32))
    if dof > len(target):
        target = np.pad(target, (0, dof - len(target)), constant_values=0)
    
    start_time = time.time()
    while time.time() - start_time < duration:
        ur5e.set_joint_positions(target)
        safe_update()

def return_to_home():
    """Return robot to home position."""
    print("\n" + "="*60)
    print("   RETURNING TO HOME POSITION")
    print("="*60)
    
    move_to_position(HOME_POSITION, steps=STEPS_PER_MOVE, description=f"Moving to home: {HOME_POSITION[:6]}")
    print("✓ Robot at home position")

# ============================================================
# MAIN TRAJECTORY EXECUTION
# ============================================================

print("\n" + "="*60)
print("   STARTING SQUARE TRAJECTORY")
print("="*60)
print(f"Corners to visit: {len(SQUARE_POSITIONS)}")
print(f"Dwell time: {DWELL_TIME}s")
print(f"Movement steps: {STEPS_PER_MOVE}\n")

# Move to home position first
print("Moving to home position...")
move_to_position(HOME_POSITION, steps=STEPS_PER_MOVE)
for _ in range(30):
    safe_update()

# Execute square trajectory
for idx, position in enumerate(SQUARE_POSITIONS):
    corner_num = idx + 1
    
    print(f"\n{'─'*60}")
    print(f"📍 CORNER {corner_num}/{len(SQUARE_POSITIONS)}")
    print(f"{'─'*60}")
    
    # Move to corner
    move_to_position(position, steps=STEPS_PER_MOVE, description=f"Moving to corner {corner_num}: {position[:6]}")
    
    # Hold at corner
    print(f"   ⏱️  Holding for {DWELL_TIME}s...")
    hold_position(position, DWELL_TIME)

# Return to home
return_to_home()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("   🎉 TRAJECTORY COMPLETE")
print("="*60)
print(f"Corners visited: {len(SQUARE_POSITIONS)}")
print("\n✓ All operations complete!")
print("="*60 + "\n")