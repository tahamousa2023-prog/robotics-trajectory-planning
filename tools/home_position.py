import numpy as np
import omni.kit.app
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
ROBOT_PATH = "/World/RobotHand/ur5e"  # Adjust if needed

# Target position (in degrees)
TARGET_POSITION = [45.0, -90.0, -90.0, -90.0, 90.0, -90.0, 0.0, 0.0]

# ============================================================
# MOVE ROBOT
# ============================================================
world = World()
app = omni.kit.app.get_app()

# Create unique name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
robot_name = f"ur5e_robot_{timestamp}"

ur5e = Articulation(prim_path=ROBOT_PATH, name=robot_name)
world.scene.add(ur5e)
world.reset()

# Convert to radians
target = np.deg2rad(np.array(TARGET_POSITION, dtype=np.float32))

# Move to position
print(f"Robot name: {robot_name}")
print("Moving to target position...")
for _ in range(120):
    ur5e.set_joint_positions(target)
    app.update()

print("✓ Done!")