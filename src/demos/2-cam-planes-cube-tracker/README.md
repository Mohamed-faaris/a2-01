# Two-Camera Multi-Feature Realtime Tracker Demo

Real-time 2-camera tracker for AprilTags:
- **Base Reference Frame**: Tag 0 (origin and coordinate basis loaded from [`calibrations/markers.yaml`](../../../calibrations/markers.yaml)).
- **Plane A (Line / Pair Plane)**: Tags 40 & 41 (66 mm each).
- **Plane B (Quad Platform Plane)**: Tags 42, 43, 44, 45 (66 mm each, RANSAC normal & 3D plane coordinates).
- **6-Faced AprilTag Cube**: Tags 20, 21, 22, 23, 24, 25 (41 mm each, RANSAC 3D internal centroid tracking).

---

## Architecture & Layout

The UI displays a unified interface (up to $1920 \times 720$ when the 3D scene panel is enabled):
- **Left Column ($640 \times 720$)**:
  - Top: Camera 0 annotated feed with polygon bounding boxes, tag IDs, and classification badges.
  - Bottom: Camera 1 annotated feed.
- **Middle Column ($640 \times 720$, toggle with `V`)**:
  - Software-rendered 3D perspective view of the Base frame: ground grid, Base axes, Plane A segment, Plane B quad + normal, Cube face quads, the custom `H` frame, and both camera frustums.
- **Right Column ($640 \times 720$)**:
  - Telemetry HUD dashboard with live 3D metric coordinates (mm) for the Base reference, Cube centroid, Plane A baseline and midpoint, and Plane B normal and coplanarity RMSE.

### 3D Overlay (array verification)

Each camera view re-projects the *fused metric results* (the exact `[X, Y, Z]` arrays shown in the HUD and exported JSON) back into the image: Base axes at Tag 0, the Plane A segment with per-tag coordinate readouts, the Plane B convex hull + normal arrow, Cube face quads + centroid crosshair, and the custom `H` frame. Any misalignment against the raw 2D tag detections exposes a bad coordinate array.

---

## Code Layout

| File | Contents |
|:---|:---|
| `main.py` | CLI entrypoint: argument parsing, `--list-cams` / `--list-calibs` inspection, session setup. |
| `tracker.py` | `CameraCalibration` (NPZ/JSON/YAML/profile loading + live V4L2 node matching), `VideoFeedSource`, tag detection & PnP geometry, and `TwoCameraPlanesCubePipeline` (`process_frames` + `run`). |
| `viz.py` | 3D overlays (`annotate_camera_view`, `draw_base_frame_overlay`), the `render_scene3d_panel` software renderer, `render_hud_dashboard`, and `SessionLogger` (time-series CSV + frame photo collection). |
| `checks/check_overlays.py` | Offline verification: base→camera projection round-trip (< 1e-6 px), overlay pixels at projected positions, scene panel contents, composite UI size, and session logging. |

```bash
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/checks/check_overlays.py
```

### Files handled in the restructure

- [`main.py`](./main.py) — CLI entrypoint (was a 1,743-line monolith, now 168 lines)
- [`tracker.py`](./tracker.py) — calibration loading, video feeds, tag geometry, tracking pipeline
- [`viz.py`](./viz.py) — 3D overlays, 3D scene panel, HUD dashboard, session logger
- [`checks/check_overlays.py`](./checks/check_overlays.py) — offline verification of projection math + overlays
- [`README.md`](./README.md) — this document
- Generated at runtime: `outputs/session_<timestamp>/history.csv` + `outputs/session_<timestamp>/frames/*.png` (time series + photo collection) and `outputs/live_tracking_results.json` (default snapshot path)

### Session logs (time series + photo collection)

Every run writes `outputs/session_<timestamp>/` (root configurable with `--session-dir`):
- `history.csv` — one timestamped row per frame (`--history-interval`, default `0.1 s`) with Cube centroid, Plane A/B centers, RMSE/spread and custom-frame coordinates for offline plotting.
- `frames/*.png` — the annotated UI composite saved every `--photo-interval` seconds (default `10`, `0` disables) plus a `frame_final.png` on exit.

Disable both with `--no-session`. The UI shape and overlay projection are covered by the offline check in `tmp/check_overlays.py` (round-trip projection error < 1e-6 px).

---

## Custom Calibrations & Device Node Matching

Adheres strictly to the calibration specifications in [`custom-calibrator-npz.md`](/home/mfk01/Projects/econ-systems/tools/calibrator/docs/custom-calibrator-npz.md):
- **Universal Loading**: Loads calibrations from `.npz`, `.json`, `.yaml`, or named profile directories (e.g. `--calib0 see3cam --calib1 rl720`).
- **Dynamic V4L2 Node Matching**: Scans connected video capture cards via V4L2 (`enumerate_v4l2_streams`) and automatically remaps to the active RGB streaming node (e.g. automatically binds RealSense RGB to `/dev/video6` instead of the 16-bit depth stream `/dev/video0`).
- **Stream-Type Verification**: Verifies `stream_type` (e.g. `RGB / Color`) against camera capabilities to prevent selecting metadata or depth nodes.

---

## Usage

### 1. Live USB / V4L2 Webcams (Auto Device Node Resolution)
```bash
# Automatically finds /dev/video8 for See3CAM and /dev/video6 for RealSense RGB
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --calib0 see3cam \
  --calib1 rl720
```

### 2. Inspect Connected Cameras & Discovered Calibrations
```bash
# List all connected V4L2 video devices, cards, and capture capabilities
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py --list-cams

# List all discovered calibration profiles across standard output directories
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py --list-calibs
```

### 3. Static Images / Synthetic Frames
```bash
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --cam0 path/to/cam0.png \
  --cam1 path/to/cam1.png \
  --active-ratio 0.8
```

### 4. Headless Mode (CLI / JSON Export)
```bash
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --cam0 path/to/cam0.png \
  --cam1 path/to/cam1.png \
  --no-display \
  --output-json outputs/tracking_snapshot.json
```

### 5. Save Outputs Under Experiment Subfolder
```bash
# Saves into outputs/<experiment-name>/session_<timestamp>/ (history.csv, frames/)
# and updates outputs/<experiment-name>/live_tracking_results.json + .png on 'S' keypress
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --calib0 see3cam \
  --calib1 oak \
  --exp 21_cube_on_plane_platform

# Save directly into outputs/<experiment-name>/ without timestamp subfolders:
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --calib0 see3cam \
  --calib1 oak \
  --exp 21_cube_on_plane_platform \
  --no-timestamp
```

### 6. Interactive 3D WebGL Viewer & Snapshot Tool
Launch the interactive 3D WebGL viewer powered by `viser` (Three.js WebGL with orbit, pan, zoom):
```bash
# Automatically finds latest live_tracking_results.json and serves on http://localhost:8080
uv run src/demos/2-cam-planes-cube-tracker/viewer3d.py

# Or inspect a specific session or snapshot:
uv run src/demos/2-cam-planes-cube-tracker/viewer3d.py \
  --input outputs/session_20261021_104048/live_tracking_results.json \
  --port 8080
```
- **Live Auto-Reload**: Automatically updates in real time as the tracker writes new telemetry.
- **Session Dropdown**: Seamlessly switch between different session JSON files.
- **`📸 Save Snapshot (JSON + PNG)`**: Saves both the active JSON state and a high-resolution 3D client render image into the session folder with a single click.

---

## Interactive Keyboard Shortcuts

| Key | Action |
|:---|:---|
| **`H`** | **Toggle Custom Baseline Reference Frame**: <br>• Baseline Top $\rightarrow$ $+X$ axis (from Tag 40 to 41)<br>• Baseline Right $\rightarrow$ $+Y$ axis<br>• Camera view ray $\rightarrow$ $+Z$ axis<br>• Rotates $90^\circ$ around $Y$-axis so $+Z$ faces **UP**<br>• Offsets origin by **$+2\text{ cm}$ along $X$** and **$+5\text{ cm}$ along $Z$**<br>• Live computes the 6-faced Cube Centroid expressed in this custom frame |
| **`V`** | Toggle the 3D scene panel (middle column) |
| **`S`** | Export current telemetry snapshot to JSON file and save annotated frame photo |
| **`Q`** / **`ESC`** | Quit application |
