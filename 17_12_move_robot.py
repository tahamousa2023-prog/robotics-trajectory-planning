#####################################################################
########### Loading the robot into isaacsim #########################
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
import carb

# Add UR5e to scene
ur5e_usd = carb.settings.get_settings().get("/persistent/isaac/asset_root/default") + "/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"

add_reference_to_stage(usd_path=ur5e_usd, prim_path="/World/ur5e")

print("UR5e loaded successfully!")
###############################################################
