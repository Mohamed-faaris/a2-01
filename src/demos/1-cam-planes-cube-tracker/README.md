# Single-Camera Multi-Feature Realtime Tracker Demo

Real-time **one-camera** tracker for AprilTags with physics-constrained geometry:
- **Base Reference Frame**: Tag 0 (origin and coordinate basis loaded from [`calibrations/markers.yaml`](../../../calibrations/markers.yaml)).
- **Plane A (Line / Pair Plane)**: Tags 40 & 41 (66 mm each).
- **Plane B (Quad Platform Plane)**: Tags 42, 43, 44, 45 (66 mm each).
- **6-Faced AprilTag Cube**: Tags 20, 21, 22, 23, 24, 25 (41 mm each).

---

## Physics-Constrained Estimation (what makes this demo smarter)

Real scenes obey geometry, so the solver enforces it instead of fitting blindly:

1. **Platform plane → ONE shared normal.**
   Tags 42..45 are markers placed flat on the same platform, so all four
   observed normals are *the same direction* up to PnP noise. The plane normal
   is estimated by averaging the tag orientations (hemisphere-aligned) — far
   less noisy than fitting a plane through 4 center points — then the offset is
   the median-gated mean of the center distances along that normal. Every
   center is projected **onto** the plane along the normal, so the reported
   centers are exactly coplanar (`centers_projected_mm`), and
   `coplanarity_rmse_mm` measures flatness *along the shared normal*.

2. **Cube on the platform → face normals parallel or perpendicular.**
   A cube resting on the plane has face normals either **parallel** to the
   plane normal (top/bottom) or **perpendicular** to it (four sides). Each
   observed face normal is snapped to that allowed set first
   (`snap_face_normal_to_axes`); faces disagreeing by more than `25°` are
   inconsistent with the model and rejected as outliers. The centroid is the
   medoid-gated mean of the remaining face-center candidates (pulled inward by
   half the cube side along the snapped normal).

3. **On-plane gap metric.**
   `on_plane_gap_mm` = distance between the cube bottom and the platform
   surface along the plane normal (**0 = resting exactly**). A
   contact-corrected centroid (`xyz_contact_mm`) is exported for reference.
   The HUD turns green when |gap| ≤ 5 mm, and the overlay labels the gap
   directly under the CUBE readout.

Measured results (offline check, noise-injected synthetic scene):
shared-normal error `0.23°`, flatness RMSE `0.59 mm`, cube centroid error
`0.36 mm`, gap error `0.21 mm`, wrong-pose face rejected.

---

## Architecture & Layout

The UI is a fixed **1920 × 720** composite:
- **Left Column ($640 \times 720$)**:
  - Top: Camera 0 annotated feed (tag polygons, IDs, re-projected 3D overlay).
  - Bottom: live **Tag Detections panel** — base lock status, tags seen this
    frame with entity class and decision margin.
- **Middle Column ($640 \times 720$, toggle with `V`)**: software-rendered 3D
  perspective view of the Base frame (ground grid, Base axes, Plane A segment,
  Plane B quad + shared normal, cube quads + centroid, `H` frame, camera
  frustum).
- **Right Column ($640 \times 720$)**: telemetry HUD — Base lock, cube centroid
  with **on-plane gap / snap / yaw**, Plane A baseline, Plane B shared normal +
  flatness RMSE, and the custom `H` frame status.

### 3D Overlay (array verification)

The camera view re-projects the *fused metric results* (the exact `[X, Y, Z]`
arrays shown in the HUD and exported JSON) back into the image. Any
misalignment against the raw 2D tag detections exposes a bad coordinate array.

---

## Code Layout

| File | Contents |
|:---|:---|
| `main.py` | CLI entrypoint: argument parsing, `--list-cams` / `--list-calibs`, session setup, `outputs/<name>/<time>/` wiring. |
| `tracker.py` | `CameraCalibration` (NPZ/JSON/YAML/profile + live V4L2 node matching), `VideoFeedSource`, tag detection & PnP (best-of `solvePnPGeneric`), **constrained geometry** (`fit_platform_plane_constrained`, `snap_face_normal_to_axes`, `estimate_cube_centroid_constrained`), `SingleCameraPlanesCubePipeline` (`process_frame` + `run`, `save_snapshot_bundle`). |
| `viz.py` | `annotate_camera_view`, `draw_base_frame_overlay`, `render_scene3d_panel`, `render_detections_panel`, `render_hud_dashboard`, `SessionLogger`. |
| `checks/check_overlays.py` | Offline verification: projection round-trip (< 1e-6 px), **constrained geometry units** (shared normal / snapping / cube gap + outlier rejection), overlay pixels at projected positions, scene panel, composite UI (1920×720), session layout, detector smoke. |
| `README.md` | This document. |

```bash
PYTHONPATH=src uv run python src/demos/1-cam-planes-cube-tracker/checks/check_overlays.py
# → ALL CHECKS PASSED
```

---

## Custom Calibrations & Device Node Matching

Same rules as the 2-cam demo ([`custom-calibrator-npz.md`](/home/mfk01/Projects/econ-systems/tools/calibrator/docs/custom-calibrator-npz.md)):
- **Universal Loading**: `.npz`, `.json`, `.yaml`, or named profile (`--calib see3cam`).
- **Dynamic V4L2 Node Matching**: re-binds to the live RGB streaming node if a
  camera was replugged into another port (e.g. RealSense depth on
  `/dev/video0`, RGB on `/dev/video6`).
- **Stream-Type Verification**: warns when a color calibration lands on a
  depth/metadata node.

---

## Usage

### 1. Live USB / V4L2 Webcam (auto device node resolution)
```bash
PYTHONPATH=src uv run python src/demos/1-cam-planes-cube-tracker/main.py \
  --calib see3cam
```

### 2. Inspect cameras & calibrations
```bash
PYTHONPATH=src uv run python src/demos/1-cam-planes-cube-tracker/main.py --list-cams
PYTHONPATH=src uv run python src/demos/1-cam-planes-cube-tracker/main.py --list-calibs
```

### 3. Static image / video file
```bash
PYTHONPATH=src uv run python src/demos/1-cam-planes-cube-tracker/main.py \
  --cam path/to/frame.png \
  --active-ratio 0.8
```

### 4. Headless (JSON export + snapshot bundle)
```bash
PYTHONPATH=src uv run python src/demos/1-cam-planes-cube-tracker/main.py \
  --cam experiment/20_cube_centroid_and_target_tag/001/inputs/cam0_render.png \
  --exp my_run \
  --no-display
```

### 5. Session / snapshot layout (as requested)

Everything lands under **`outputs/<name>/<time>/`**:

```
outputs/<name>/<time>/
├── cam0.png                   raw Camera 0 frame
├── cam0_annotated.png         annotated view
├── combined_ui.png            full dashboard composite
├── live_tracking_results.json telemetry snapshot
├── history.csv                time series (one row per frame)
└── frames/*.png               UI photos ('S' key + --photo-interval)
```

- `<name>` = `--exp <name>` (default folder `outputs/default/`).
- `<time>` = `YYYYMMDD_HHMMSS` folder created per run (skip with `--no-timestamp`).
- `outputs/<name>/live_tracking_results.json` and the root
  `outputs/live_tracking_results.json` are refreshed on every snapshot.
- `--session-dir` moves the root; `--no-session` disables CSV/photos.

---

## Interactive Keyboard Shortcuts

| Key | Action |
|:---|:---|
| **`H`** | Toggle Custom Baseline Reference Frame (same spec as 2-cam: baseline → +X, camera ray → +Z, rotY 90° → Z=UP, offset +2 cm X / +5 cm Z; cube centroid re-expressed in it) |
| **`V`** | Toggle the 3D scene panel |
| **`S`** | Save snapshot bundle: `outputs/<name>/<time>/{cam0, cam0_annotated, combined_ui}.png` + `live_tracking_results.json` |
| **`Q`** / **`ESC`** | Quit |

---

## Files handled in this demo

- [`main.py`](./main.py) — CLI entrypoint
- [`tracker.py`](./tracker.py) — calibration, feed, PnP geometry, constrained plane/cube estimation, pipeline
- [`viz.py`](./viz.py) — overlays, detections panel, 3D scene, HUD, session logger
- [`checks/check_overlays.py`](./checks/check_overlays.py) — offline verification incl. constrained-geometry units
- [`README.md`](./README.md) — this document
