import numpy as np
from omni.isaac.core import World
from omni.isaac.core.articulations import Articulation
import omni.kit.app

# Initialize
world = World()
franka = world.scene.add(
    Articulation(prim_path="/Franka", name="my_fsgsgska")
)
world.reset()

# Get the app update interface
app = omni.kit.app.get_app()

# Define positions
home = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785, 0.04, 0.04])
target = np.array([01.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.785, 0.04, 0.04])

print("Moving smoothly...")

# Smooth movement
num_steps = 150
for i in range(num_steps):
    t = i / num_steps
    position = home + t * (target - home)
    franka.set_joint_positions(position)
    
    # Update one frame
    app.update()

print("Movement complete!")
