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

## Usage

### 1. Live USB / V4L2 Webcams
```bash
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --cam0 /dev/video8 \
  --cam1 /dev/video0 \
  --calib0 see3cam \
  --calib1 rl720
```

### 2. Static Images / Synthetic Frames
```bash
PYTHONPATH=src uv run python src/demos/2-cam-planes-cube-tracker/main.py \
  --cam0 path/to/cam0.png \
  --cam1 path/to/cam1.png \
  --active-ratio 0.8
```

### 3. Headless Mode (CLI / JSON Export)
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
| **`S`** | Export current telemetry snapshot to JSON file |
| **`Q`** / **`ESC`** | Quit application |
