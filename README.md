# UR5e Trajectory Planning for Multi-View 3D Reconstruction

Camera trajectory planning for a UR5e robot arm in NVIDIA Isaac Sim, controlled through ROS 2 and MoveIt2. The scripts move a wrist-mounted camera along scan patterns around an object and capture images for 2D-to-3D reconstruction (VGGT, Fast3R, SAM3D).

This is my part of the [Path Matters](https://github.com/tahamousa2023-prog/path-matters-robotics) project at TU Berlin (WiSe 2025/26), where I was responsible for trajectory generation and evaluation.

## Key finding

Pointing the camera at the detected object at every waypoint, instead of keeping it fixed downward, raised mean ICP fitness from **0.68 to 0.79** and reduced mean inlier RMSE from 0.035 m to 0.022 m across five scan patterns, without changing the trajectory geometry or the number of viewpoints. The best configuration was a hemisphere pattern with object-pointing orientation (fitness 0.86, RMSE 0.015 m).

## Repository structure

| Folder | Contents |
|---|---|
| `capture/` | Main pipelines: orbit capture with object detection, object-pointing capture, multi-view capture, photo capture, and the object detection module |
| `patterns/` | Scan pattern generation and visualisation: hemisphere, circular, square, smooth and cubic-eased motion |
| `tools/` | Utilities: home position, position tester, IK diagnostics, robot loading in Isaac Sim |
| `archive/` | Earlier development versions, kept for reference (named by date) |

**Main entry points**

| Script | Purpose |
|---|---|
| `capture/orbit_capture_detect.py` | Detects the object from the home pose, generates orbit poses that look at its centre, moves and captures images |
| `capture/object_pointing_capture.py` | Capture with the camera orientation computed at each waypoint to face the object |
| `capture/multiview_capture_detection.py` | Multi-view capture with automatic object detection and size estimation |
| `capture/modular_object_detection.py` | Object detection and size estimation (used by the capture scripts) |

## Requirements

- NVIDIA Isaac Sim with a UR5e scene and wrist camera
- ROS 2 Humble, MoveIt2 (`ur_moveit_config`), `cv_bridge`
- Python 3.10, NumPy, OpenCV; Segment Anything checkpoints in `checkpoints/` for SAM-based detection

## How to run

Use three terminals:

```bash
# 1) Isaac Sim: start it and open the UR5e scene
./isaac-sim.sh

# 2) MoveIt2
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e

# 3) Capture pipeline
python3 capture/orbit_capture_detect.py
```

Captured images are written to `captures/`.

## License

MIT License, see [LICENSE](LICENSE).

## Contact

**Taha Mohammed** · [Website](https://tahamousa2023-prog.github.io) · [LinkedIn](https://www.linkedin.com/in/taha-mahmoud)
