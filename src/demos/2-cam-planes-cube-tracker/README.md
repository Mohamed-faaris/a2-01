# Two-Camera Multi-Feature Realtime Tracker Demo

Real-time 2-camera tracker for AprilTags:
- **Base Reference Frame**: Tag 0 (origin and coordinate basis loaded from [`calibrations/markers.yaml`](../../../calibrations/markers.yaml)).
- **Plane A (Line / Pair Plane)**: Tags 40 & 41 (66 mm each).
- **Plane B (Quad Platform Plane)**: Tags 42, 43, 44, 45 (66 mm each, RANSAC normal & 3D plane coordinates).
- **6-Faced AprilTag Cube**: Tags 20, 21, 22, 23, 24, 25 (41 mm each, RANSAC 3D internal centroid tracking).

---

## Architecture & Layout

The UI displays a side-by-side $1280 \times 720$ unified interface:
- **Left Column ($640 \times 720$)**:
  - Top: Camera 0 annotated feed with polygon bounding boxes, tag IDs, and classification badges.
  - Bottom: Camera 1 annotated feed.
- **Right Column ($640 \times 720$)**:
  - Telemetry HUD dashboard with live 3D metric coordinates (mm) for the Base reference, Cube centroid, Plane A baseline and midpoint, and Plane B normal and coplanarity RMSE.

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

---

## Interactive Keyboard Shortcuts

| Key | Action |
|:---|:---|
| **`H`** | **Toggle Custom Baseline Reference Frame**: <br>• Baseline Top $\rightarrow$ $+X$ axis (from Tag 40 to 41)<br>• Baseline Right $\rightarrow$ $+Y$ axis<br>• Camera view ray $\rightarrow$ $+Z$ axis<br>• Rotates $90^\circ$ around $Y$-axis so $+Z$ faces **UP**<br>• Offsets origin by **$+2\text{ cm}$ along $X$** and **$+5\text{ cm}$ along $Z$**<br>• Live computes the 6-faced Cube Centroid expressed in this custom frame |
| **`S`** | Export current telemetry snapshot to JSON file |
| **`Q`** / **`ESC`** | Quit application |
