# Multi-Camera Multi-Feature Realtime Tracker Demo

Real-time multi-camera tracker supporting arbitrary $N$ cameras ($N \ge 1$, e.g. 2, 3, 4, or more) for AprilTags:
- **Base Reference Frame**: Tag 0 (origin and coordinate basis loaded from [`calibrations/markers.yaml`](../../../calibrations/markers.yaml) and [`calibrations/base_link.yaml`](../../../calibrations/base_link.yaml)).
- **Plane A (Line / Pair Plane)**: Tags 40 & 41 (66 mm each).
- **Plane B (Quad Platform Plane)**: Tags 42, 43, 44, 45 (66 mm each, consensus normal & 3D plane coordinates across all cameras).
- **6-Faced AprilTag Cube**: Tags 20, 21, 22, 23, 24, 25 (41 mm each, RANSAC 3D internal centroid tracking with support plane constraints).
- **Multi-Camera PnP Fusion & Auto-Survey**: Seamlessly fuses detections from all available cameras. Static tags are automatically surveyed into a unified Base-Frame world map so cameras that cannot directly see Tag 0 can still track in the exact same world coordinate system.

---

## Architecture & Layout

The UI displays a unified composite layout ($1920 \times 720$ when the 3D scene panel is enabled):
- **Left Column ($640 \times 720$)**:
  - Dynamically tiles all $N$ camera annotated feeds into the column canvas:
    * $1$ Camera: $640 \times 360$ centered with letterboxing.
    * $2$ Cameras: Two $640 \times 360$ feeds vertically stacked (identical to the 2-cam layout).
    * $3$ or $4$ Cameras: $2 \times 2$ grid ($320 \times 360$ each) with per-camera color badges.
    * $>4$ Cameras: Dynamic multi-column grid scaled to fit $640 \times 720$.
- **Middle Column ($640 \times 720$, toggle with `V`)**:
  - Software-rendered 3D perspective view of the Base frame: ground grid, Base axes, Plane A segment, Plane B quad + normal, Cube face quads, the custom `H` frame, and 3D frustums for all active cameras with color-coded badges.
- **Right Column ($640 \times 720$)**:
  - Telemetry HUD dashboard displaying:
    * Multi-Camera Status Table (Cam 0, Cam 1, Cam 2... with lock status and PnP method).
    * World Map surveyed tag count.
    * Cube 3D centroid $[X, Y, Z]$, visible faces, observation count, and residual spread.
    * Plane A baseline distance, midpoint, and height.
    * Plane B surface normal $[n_x, n_y, n_z]$, center, and coplanarity RMSE.
    * `base_link` reference frame status and origins.

---

## Code Layout

| File | Contents |
|:---|:---|
| [`main.py`](./main.py) | CLI entrypoint: supports `--cams` (space-separated list) or `--cam0`, `--cam1`..., `--calibs`, inspection modes (`--list-cams`, `--list-calibs`), and session setup. |
| [`tracker.py`](./tracker.py) | `CameraCalibration`, `VideoFeedSource`, `MultiCameraPlanesCubePipeline` (`process_frames` + `run`), multi-camera auto-survey world map, multi-camera PnP fusion, and geometry math. |
| [`viz.py`](./viz.py) | Multi-camera view tiling (`layout_camera_views`), 3D overlays (`annotate_camera_view`, `draw_base_frame_overlay`), multi-camera frustum 3D renderer (`render_scene3d_panel`), multi-camera telemetry HUD (`render_hud_dashboard`), and `SessionLogger`. |
| [`viewer3d.py`](./viewer3d.py) | Interactive WebGL 3D viewer (Viser) with multi-camera frustums, dynamic "Cam N POV" shortcuts, and live file reloading. |
| [`checks/check_overlays.py`](./checks/check_overlays.py) | Offline verification suite: synthetic projection round-trip, multi-camera tiling layout, HUD dashboard rendering, scene panel with frustums, composite UI, and session logging. |

---

## Interactive Controls (Realtime Loop)

| Key | Action |
|:---|:---|
| `Q` or `ESC` | Quit tracking loop cleanly and close cameras. |
| `H` | Toggle Custom Baseline `H` coordinate frame mode on/off. |
| `V` | Toggle 3D Scene Panel on/off (expands/contracts UI width). |
| `S` | Save full snapshot bundle: raw frames (`cam0.png`, `cam1.png`...), annotated frames, `combined_ui.png`, and `live_tracking_results.json`. |

---

## Usage Examples

### 1. Two Cameras (Standard Setup)
```bash
# Live cameras with auto device node resolution
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/main.py \
  --calibs see3cam rl720
```

### 2. Three or More Cameras
```bash
# Using --cams and --calibs
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/main.py \
  --cams /dev/video0 /dev/video2 /dev/video4 \
  --calibs see3cam rl720 astra720

# Or using explicit flags:
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/main.py \
  --cam0 0 --calib0 see3cam \
  --cam1 2 --calib1 rl720 \
  --cam2 4 --calib2 astra720
```

### 3. Inspect Connected Video Streams and Discovered Profiles
```bash
# Scan and print all connected V4L2 device nodes and card names
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/main.py --list-cams

# Scan and print all discovered camera calibration profiles (.npz, .json, .yaml)
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/main.py --list-calibs
```

### 4. Headless Execution / Benchmark Run
```bash
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/main.py \
  --cams path/to/cam0.png path/to/cam1.png path/to/cam2.png \
  --no-display \
  --name my_experiment
```

### 5. Interactive 3D WebGL Viewer
```bash
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/viewer3d.py
# Open http://localhost:8080 in your browser
```

### 6. Run Offline Verification Suite
```bash
PYTHONPATH=src .venv/bin/python src/demos/multi-cam-planes-cube-tracker/checks/check_overlays.py
```
