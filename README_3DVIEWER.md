# 🌐 3D Hand-Eye Visualizer (`viewer3d.py`)

Interactive 3D WebGL digital twin and scene mapping visualizer for multi-pose hand-eye calibration with the **DOBOT CR5A** 6-DOF industrial robot arm, **Intel RealSense D415**, and global context **USB camera**.

Powered by **[Viser](https://viser.studio/)** (Three.js WebGL engine running directly in your browser).

---

## 📖 Table of Contents
1. [How It Works (Architecture & Coordinate Frames)](#-how-it-works)
2. [Quick Start (Default Execution)](#-quick-start-default-run)
3. [Command-Line Arguments & Default Values](#-command-line-arguments--defaults)
4. [How to Load Different Datasets & Calibration Files](#-how-to-load-different-datasets--files)
5. [Interactive Web UI Controls & Features](#-interactive-web-ui-controls--features)
6. [Visual Elements Breakdown](#-visual-elements-breakdown)
7. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🧠 How It Works

### 1. Global World Origin (`base_link`)
All visual and geometric entities are registered relative to the robot's physical base (`base_link` frame at $[0, 0, 0]$):
- **Robot Arm**: The DOBOT CR5A forward kinematics chain is computed using the exact joint offsets and rotation axes defined in `/home/mfk01/Downloads/Robotics_Resources/cr5/urdf/cr5a.urdf`.
- **Dual Visual Modes**:
  - **Rods / Skeleton Mode**: Renders precise cylindrical links connecting joint hubs, color-coded with RGB coordinate axes ($X=\text{Red}, Y=\text{Green}, Z=\text{Blue}$).
  - **CAD Mesh Mode**: Loads high-fidelity STL CAD meshes for the base, joints 1–6, connector slabs, and gripper.
- **Calibrated Cameras**:
  - **Intel RealSense D415**: 3D frustum with realistic focal ratio positioned at the calibrated $T_{\text{base}\to\text{d415}}$ extrinsic pose ($\approx [1.91, -0.32, 0.80]\text{ m}$).
  - **USB Global Camera**: Placed at $T_{\text{base}\to\text{usb}}$ ($\approx [1.97, -0.21, 0.32]\text{ m}$).
- **Dynamic Tool Assembly**:
  - Tag 20 (Cube Front face, $41\text{ mm}$)
  - Tag 21 (Cube Side face, $41\text{ mm}$)
  - Tag 40 (Spindle / Tool Tip, $41\text{ mm}$)  
  These move rigidly with the robot end-effector ($T_{\text{base}\to\text{tcp}} \cdot T_{\text{tcp}\to\text{tag}}$).
- **Static World Landmarks**:
  - Base Markers (Tags 4, 5, $66\text{ mm}$) bolted to the baseplate.
  - Workpiece Platform (Tags 42, 43, 44, 45, $66\text{ mm}$) with its fitted 3D best-fit plane ($1.82\text{ mm}$ planarity RMSE).
- **Optical Verification Rays & Residual Lines**:
  - Yellow rays project from the camera optical centers to detected tags.
  - Red residual error lines visually connect the model's forward-kinematic predicted tag center with the camera's observed tag center.

```
                  [Intel RealSense D415] (Fixed in base_link)
                            │  \
              Sighting Rays │   \ Sighting Rays
                            ▼    ▼
[Base Tag 4 & 5] ◄── [Robot Base (0,0,0)] ──► [Workpiece Plane (Tags 42-45)]
                            │
                       (CR5A FK)
                            │
                            ▼
                      [Robot TCP]
                            │ (Rigid Mounts)
                            ▼
                   [Tool Tags 20, 21, 40]
```

---

## 🚀 Quick Start (Default Run)

To run the visualizer with default parameters:

```bash
# From the repository root:
.venv/bin/python src/demos/hand-eye-0/viewer3d.py
```

Once started, open your web browser to:
👉 **[http://localhost:8085](http://localhost:8085)**

*(Or `http://<machine-ip>:8085` if accessing remotely across your local network).*

---

## ⚙️ Command-Line Arguments & Defaults

| Argument | Type | Default Value | Description |
| :--- | :---: | :--- | :--- |
| **`--calib`** | Path / Dir | `src/demos/hand-eye-0/outputs/handeye_calibration.json` | Path to `handeye_calibration.json`, `scene.json`, or an output directory (e.g. `outputs/dobot-vision-011`). |
| **`--dataset`** | Path | `tmp/eye-hand/008` | Path to the directory containing pose subfolders with `joint_angles.yaml` and camera images. |
| **`--resources`**| Path | `/home/mfk01/Downloads/Robotics_Resources/cr5` | Directory containing DOBOT CR5A STL CAD meshes and URDF files. |
| **`--points`** | Path | Auto (`001.jsonl`) | Path to additional points or waypoints file (`.jsonl` or `.json`), e.g. `output-dobots/pose-above-box/001.jsonl`. |
| **`--port`** | int | `8085` | Web server port for the browser UI. |
| **`--host`** | str | `0.0.0.0` | Server bind IP (`0.0.0.0` enables remote LAN access). |

---

## 📂 How to Load Different Datasets & Files

### 1. Load the New 41-Pose Dataset (`dobot-vision-011`)
To view the latest calibration results and arm trajectory for dataset `011`:

```bash
.venv/bin/python src/demos/hand-eye-0/viewer3d.py \
  --calib outputs/dobot-vision-011/handeye_calibration.json \
  --dataset inputs/dobot-vision/011
```

*(You can also pass the directory directly: `--calib outputs/dobot-vision-011`)*

### 2. Compare Different Calibration Solvers
To inspect how different optimization algorithms align the scene:

- **Direct 2D Corner BA (Optimal)**:
  ```bash
  .venv/bin/python src/demos/hand-eye-0/viewer3d.py \
    --calib outputs/dobot-vision-011/algo_2d_corner_ba/handeye_calibration.json \
    --dataset inputs/dobot-vision/011
  ```

- **Rigid 6-Faced Cube Constraint**:
  ```bash
  .venv/bin/python src/demos/hand-eye-0/viewer3d.py \
    --calib outputs/dobot-vision-011/algo_rigid_cube_ba/handeye_calibration.json \
    --dataset inputs/dobot-vision/011
  ```

- **Baseline 3D SE(3) Cauchy**:
  ```bash
  .venv/bin/python src/demos/hand-eye-0/viewer3d.py \
    --calib outputs/dobot-vision-011/algo_3d_pose_cauchy/handeye_calibration.json \
    --dataset inputs/dobot-vision/011
  ```

### 3. Load Dataset with Additional Box Corner Waypoints
```bash
.venv/bin/python src/demos/hand-eye-0/viewer3d.py \
  --calib outputs/dobot-vision-024 \
  --dataset inputs/dobot-vision/024 \
  --points output-dobots/pose-above-box/001.jsonl
```

### 4. Change Port If Port 8085 Is Busy
```bash
.venv/bin/python src/demos/hand-eye-0/viewer3d.py --port 8090
```

---

## 🎛️ Interactive Web UI Controls & Features

When viewing in the browser, the left-hand panel provides interactive GUI controls:

### 📍 Pose Inspection & Animation Playback
- **Select Pose Dropdown**: Instantly jump to any recorded pose (e.g. `pose01` through `pose41`). The robot smoothly updates its FK, and the TCP tool tags snap to their exact 3D coordinates.
- **▶️ Play / ⏸️ Pause Animation**: Automatically cycle through all poses to preview the robot's physical trajectory.
- **Playback Speed (fps)**: Slider from $0.5\text{ fps}$ to $5.0\text{ fps}$ for smooth trajectory playback.

### 🎨 Robot Visualization Mode
- **Display Mode**:
  - `Rods (Skeleton)`: Minimalist cylindrical rods with joint sphere hubs.
  - `CAD Meshes (STLs)`: High-resolution industrial CAD representation.
  - `Both`: Overlays the kinematic skeleton on top of the CAD meshes.
- **Rod Thickness Multiplier**: Slider ($0.5\times$ to $2.0\times$) to adjust skeleton rod diameters.

### 👁️ Display Layers Toggles
You can toggle individual 3D elements on or off:
- `[x] Robot Arm`: Toggle arm visualization.
- `[x] Joint Axes Frames`: RGB coordinate frames at each joint hub ($J_1$ to $J_6$).
- `[x] Tool Tags (TCP)`: Tags 20, 21, and 40 attached to the end-effector.
- `[x] Static Tags (Base & Box)`: Tags 4, 5 and the workpiece plane tags 42–45.
- `[x] Camera Frustums`: Dual D415 and USB camera pyramid frustums.
- `[x] Camera Sighting Rays`: Yellow sighting lines from camera sensors to tags.
- `[x] Ground Grid`: Metric ground reference grid at $Z=0$.

### 🎥 Camera POV View Presets
Buttons to snap the 3D viewport to standard viewpoints:
- **Home 3D View**: Standard isometric bird's-eye view.
- **Top-Down View**: Looking directly down the $+Z$ axis at the table.
- **Look from D415 Camera**: First-person perspective looking out from the RealSense D415 sensor.
- **Look from USB Camera**: First-person perspective looking out from the USB context camera.
- **Follow Robot TCP**: Third-person chase camera locked to the end-effector.

### 📊 Live Telemetry Card
Displays real-time telemetry for the selected pose:
- Active pose name and index.
- 6-DOF joint angles in both degrees and radians ($J_1 \dots J_6$).
- Metric TCP coordinates in `base_link` frame ($[X, Y, Z]\text{ m}$).
- List of visible tags in D415 and USB camera views.
- Calibrated camera positions and residual RMS metrics.

---

## 🧩 Visual Elements Breakdown

| Visual Element | Color / Appearance | Description |
| :--- | :--- | :--- |
| **Robot Kinematics** | Silver/Blue Meshes or Cylinders | DOBOT CR5A 6-DOF arm articulated via exact URDF FK. |
| **Joint Coordinate Frames** | Red ($+X$), Green ($+Y$), Blue ($+Z$) | Orientation frame at each joint center. |
| **Tool Tags (20, 21, 40)** | Cyan square with border | Dynamic markers rigidly fixed to TCP. |
| **Workpiece Tags (42–45)** | Yellow squares + semi-transparent plane | Fixture platform defining the workpiece coordinate plane. |
| **Base Tags (4, 5)** | Orange squares | Stationary reference markers on the table baseplate. |
| **Camera Frustums** | Blue (D415) / Green (USB) | 3D visual cones showing camera field-of-view and optical center. |
| **Optical Rays** | Thin yellow lines | Sighting vectors connecting cameras to visible AprilTags. |
| **Residual Error Lines** | Thin red lines | Difference between FK prediction and camera measurement. |

---

## 🛠️ Troubleshooting & FAQs

### Q1: `Missing calibration file: src/demos/hand-eye-0/outputs/handeye_calibration.json`
**Solution**: Run the calibration benchmark first to generate the outputs:
```bash
.venv/bin/python src/demos/hand-eye-0/benchmark.py --dataset inputs/dobot-vision/011
```
Or specify the existing calibration folder with `--calib outputs/dobot-vision-011`.

### Q2: `CR5 resource dir not found`
**Solution**: If CAD meshes fail to load, verify the robotics resources path:
```bash
.venv/bin/python src/demos/hand-eye-0/viewer3d.py --resources /home/mfk01/Downloads/Robotics_Resources/cr5
```
*(If resources are missing, the visualizer automatically falls back to clean Rods/Skeleton mode without error).*

### Q3: How do I view across different machines on the same network?
**Solution**: The viewer binds to `0.0.0.0` by default. Simply open `http://<robot-pc-ip>:8085` in any browser on your laptop or tablet on the same Wi-Fi/Ethernet network.
