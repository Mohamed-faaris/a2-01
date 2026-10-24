# Hand-Eye-0: Multi-Pose Eye-to-Hand Calibration & World Scene Mapping

Comprehensive eye-to-hand calibration and static scene reconstruction pipeline using multi-pose AprilTag observations and robot forward kinematics.

## Overview

In this system:
1. **Stationary Elements**:
   - Depth camera (`d415`) and wide USB camera (`usb`) are rigidly mounted in the workspace.
   - Base cluster tags (tags `0, 1, 4, 5, 41`) are rigidly affixed to or near the robot base.
   - Workpiece box tags (tags `42, 43, 44, 45`) are rigidly affixed to the workspace table / container.
2. **Moving Elements**:
   - Robot arm (DOBOT CR5A) executes multiple distinct poses (`pose01_ready` to `pose18_ready_return`).
   - Tool tags (`20, 21, 40`) are mounted on the robot Tool Center Point (TCP / flange).

## Problem Formulation

Given:
- Robot forward kinematics for each pose: $T_{\text{base}}^{\text{tcp}}$ (from `joint_angles.yaml` FK).
- AprilTag detections in camera frame $c$: $T_{c}^{\text{tag}}$ (computed directly from raw images using `pupil_apriltags` + OpenCV IPPE-Square).

The solver estimates:
1. **Camera Extrinsics**: $T_{\text{base}}^{\text{d415}}$ and $T_{\text{base}}^{\text{usb}}$.
2. **Tool Mounts**: Rigid transforms from TCP to each mounted tag $T_{\text{tcp}}^{\text{tag}}$.
3. **Static World Tags**: 3D poses $T_{\text{base}}^{\text{tag}}$ for all stationary tags (`0, 1, 4, 5, 41, 42, 43, 44, 45`).

### Calibration Equation
For any tool tag $i$ observed by camera $c$ at pose $k$:
$$T_{\text{base}}^{\text{tcp}, k} \cdot T_{\text{tcp}}^{\text{tag}, i} = T_{\text{base}}^{c} \cdot T_{c}^{\text{tag}, i, k}$$

Rearranging:
$$T_{c}^{\text{tag}, i, k} = \left( T_{\text{base}}^{c} \right)^{-1} \cdot T_{\text{base}}^{\text{tcp}, k} \cdot T_{\text{tcp}}^{\text{tag}, i}$$

The non-linear optimizer minimizes 3D position and rotation residuals across all poses and all tool tags using Levenberg-Marquardt with a Cauchy robust loss function.

## Directory Structure

```
src/demos/hand-eye-0/
├── config.py           # Tag geometries, tag roles, camera calibration loaders
├── detector.py         # Raw image tag detection & IPPE_SQUARE PnP solver
├── solver.py           # Closed-form SVD init + non-linear joint hand-eye optimizer
├── evaluator.py        # K-fold cross-validation & residual reporting
├── viz.py              # Annotated 2D/3D image reprojection and gallery generation
├── kinematics.py       # DOBOT CR5A forward kinematics & rod skeleton builder
├── viewer3d.py         # Interactive Viser 3D WebGL viewer & pose player
├── main.py             # Calibration CLI runner
├── outputs/            # Generated calibrations, report, and annotated imagery
└── README.md
```

## Running the Pipeline

```bash
# 1. Run full calibration on dataset (tmp/eye-hand/008)
.venv/bin/python src/demos/hand-eye-0/main.py

# 2. Launch Interactive 3D WebGL Viewer (Rods & CAD meshes)
.venv/bin/python src/demos/hand-eye-0/viewer3d.py --port 8085
# Then open: http://localhost:8085
```

## 3D Viewer Features

- **DOBOT CR5A Kinematics**:
  - **Rods / Skeleton Mode**: High-performance, clean cylindrical link rods and joint spheres following exact URDF dimensions (`/home/mfk01/Downloads/Robotics_Resources/cr5/urdf/cr5a.urdf`).
  - **CAD Mesh Mode**: Loads the real STL link meshes from `/home/mfk01/Downloads/Robotics_Resources/cr5`.
- **Calibrated Cameras**: Solved D415 and USB camera bodies and optical viewing frustums in `base_link`.
- **Static Scene**: Stationary Base tags (`4`, `5`) and Workpiece box tags (`42`, `43`, `44`, `45`) + fitted 3D plane.
- **Dynamic Tool Tags**: Mounted to the moving TCP (`20`, `21`, `40`), with predicted vs observed discrepancy vectors.
- **Interactive GUI**:
  - Pose selector (Pose 01 to Pose 18)
  - Play / Pause animation loop with adjustable playback speed
  - Display layer toggles (Rods, CAD meshes, Joint frames, Tags, Sighting rays, Ground grid)
  - Camera POV shortcuts (Home, Top-down, D415 POV, USB POV, Follow TCP)
  - Real-time telemetry card with joint angles, TCP coordinates, and residual errors

## Outputs

1. **`outputs/handeye_calibration.yaml`**: Full ROS2/robotics compatible calibration configuration.
2. **`outputs/handeye_calibration.json`**: Machine-readable transform trees and metadata.
3. **`outputs/summary_report.md`**: Detailed quantitative evaluation including K-fold held-out validation.
4. **`outputs/annotated/`**: Visual verification gallery showing:
   - Green bounding boxes: observed tags.
   - Magenta axes & boxes: forward-kinematics predicted tool tags in camera view.
   - Red dashed line: residual discrepancy vector.
