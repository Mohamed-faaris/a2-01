# RealSense Object Surface Grid Triangulator & Measurer Demo

Real-time demo based on `realsense_click_triangulate` with automated object detection (cans, bottles, cups, carton boxes) and dense 2D-to-3D sampling grids across each object's surface, anchored to calibrated AprilTag coordinate frames.

---

## 🌟 Key Features

1. **Multi-Object Detection (YOLO26s):**
   - Automatically detects cans, bottles, cups, cartons, and boxes using `/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt`.
   - Distinguishes multiple object types with unique color-coded bounding boxes and mesh wireframes.

2. **Dense 3D Object Surface Grids:**
   - Instead of single-point estimation, it generates configurable $N \times M$ surface grids ($3 \times 3$, $5 \times 5$, $7 \times 7$, or $9 \times 9$) across each detected object's surface.
   - Computes 3D camera coordinates $(X_c, Y_c, Z_c)$ and tabletop AprilTag relative coordinates $(X_t, Y_t, Z_t)$ for every grid node.
   - Evaluates object centroid and spatial 3D spread (width, height, depth).

3. **Interactive Click & Inspection (`realsense_click_triangulate` style):**
   - Left-click anywhere on the Color feed to immediately sample metric depth and measure the exact 3D distance and coordinate relative to the table tag origin.
   - Press `G` to dynamically cycle grid resolutions ($3\times3 \rightarrow 5\times5 \rightarrow 7\times7 \rightarrow 9\times9$).
   - Press `M` to toggle surface grid connecting mesh lines.

4. **Centralized AprilTag Integration:**
   - Loads marker definitions from `calibrations/markers.yaml` via `common.markers` (supports Tag 0 @ 134mm, Tag 1 @ 124mm, Tags 20–31 @ 41mm, Tags 40–45 @ 66mm).
   - Projects object locations onto a top-down Bird's-Eye View (BEV) radar map.

5. **Hardware + Mock Stream Support:**
   - Detects physical Intel RealSense D415/D435 automatically.
   - Falls back to simulated photorealistic mock tabletop scenes when running offline or testing.

---

## 🚀 How to Run

### 1. Live RealSense Camera
```bash
uv run python -m demos.realsense_object_grid_triangulate.main
```
*(or via the registered CLI script)*
```bash
uv run realsense-object-grid
```

### 2. Simulated Mock Mode (Offline Playback)
```bash
uv run python -m demos.realsense_object_grid_triangulate.main --mock
```

### 3. Headless Snapshot Capture
```bash
uv run python -m demos.realsense_object_grid_triangulate.main --mock --headless --snapshot grid_demo_snapshot.png
```

---

## 🎮 Interactive Controls

| Key | Action |
|:---:|:---|
| **Left Click** | Click any pixel on Color feed (Top-Left) to inspect its exact 3D coordinate and table distance |
| **Right Click** / `c` | Clear clicked inspection point |
| `g` | Cycle grid resolution ($3\times3$, $5\times5$, $7\times7$, $9\times9$) |
| `m` | Toggle surface mesh grid lines |
| `n` | Next simulated tabletop scene (mock mode only) |
| `s` | Save high-resolution dashboard snapshot |
| `q` / `ESC` | Exit demo |
