# RealSense Pringles 3D Cylinder & BoxerNet Construction Demo

Real-time demo for tracking, constructing 3D cylinders, and fitting 3D Oriented Bounding Boxes (OBB) for **Pringles Original** and **Pringles Spicy (Extra Hot)** cans using Intel RealSense (RGB + aligned Depth) and an AprilTag table reference frame.

---

## 🌟 Key Features

1. **Dual Flavor Recognition & Wireframe Color Coding:**
   - **Pringles Original (Red):** Bright iconic crimson wireframe `(0, 0, 255)`.
   - **Pringles Spicy (Extra Hot):** Dark charred flame texture detection, rendered in flame orange-red `(0, 69, 255)`.
   - **Other Flavors:** Yellow wireframe `(0, 255, 255)`.

2. **Accurate 3D Geometry Reconstruction:**
   - **16-Rib Cylindrical Wireframe:** Circular top/bottom endcaps ($R = 3.18\text{ cm}$, $H = 25.0\text{ cm}$), central axial cylinder spine, and center marker.
   - **3D Oriented Bounding Box (OBB):** 12-edge BoxerNet style 3D bounding box wrapping the cylinder in arbitrary 3D orientations.
   - **Physical Depth Shift Compensation:** Automatically adds radius ray compensation ($+3.18\text{ cm}$) to convert RealSense front surface reflections into true cylinder centroid 3D coordinates.

3. **AprilTag Global Table Frame Anchoring:**
   - Uses `tag36h11` to calibrate the tabletop plane.
   - Projects all telemetry relative to the AprilTag origin: $X_{\text{table}}, Y_{\text{table}}, Z_{\text{table}}$ in meters.
   - Automatically determines whether cans are **Standing** (axis aligned with table normal) or **Rolling** (axis parallel to table plane).

4. **4-Quadrant 1080p Telemetry Dashboard:**
   - **Top-Left:** Main Camera View with 3D Cylinder & Bounding Box wireframe overlays.
   - **Top-Right:** Telemetry HUD Table with real-time target status, 3D coordinates, and $(\sigma_x, \sigma_y, \sigma_z)$ aleatoric uncertainties.
   - **Bottom-Left:** RealSense Metric Depth Colormap (`cv2.COLORMAP_TURBO`).
   - **Bottom-Right:** Top-Down Bird's-Eye View (BEV) Tabletop Radar Map.

5. **Physical Hardware + Simulated Mock Dual Mode:**
   - Automatically connects to an attached Intel RealSense camera (D415/D435).
   - If hardware is disconnected or `--mock` is specified, it gracefully plays simulated photorealistic tabletop sequences from Experiment 15.

---

## 🚀 How to Run

Run all commands using `uv`:

### 1. Live Camera Stream (RealSense Hardware Connected)
```bash
uv run python -m demos.realsense_pringles_spicy.main
```
*(or via the CLI shortcut)*
```bash
uv run realsense-pringles-spicy
```

> **Default Weights:** Uses `yolo26s.pt` (`/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt`).
> You can also specify it explicitly:
> ```bash
> uv run python -m demos.realsense_pringles_spicy.main --yolo-model /home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt
> ```

### 2. Simulated Mock Playback (Offline / No Camera Required)
```bash
uv run python -m demos.realsense_pringles_spicy.main --mock
```

### 3. Headless Snapshot Generation
```bash
uv run python -m demos.realsense_pringles_spicy.main --mock --headless --snapshot dashboard_snapshot.png
```

---

## 🎮 Interactive Controls

| Key | Action |
|:---:|:---|
| `b` | Toggle 3D Bounding Box (OBB) wireframes |
| `c` | Toggle 3D Cylinder wireframes |
| `n` | Step to the next simulated tabletop scene (mock mode only) |
| `s` | Save a high-resolution snapshot (`pringles_spicy_demo_snapshot.png`) |
| `q` / `ESC` | Exit demo |

---

## 📊 Pipeline Architecture

```
[RealSense RGB-D / Mock Scene]
              │
      ┌───────┴───────────────────────┐
      ▼                               ▼
[AprilTag Detector]           [YOLOv8 Detection]
(Extracts Table Plane &       (Extracts 2D RoIs: bottles,
 Coordinate Frame R, t)        cans, elongated cylinders)
      │                               │
      │                               ▼
      │                     [Spectral Flavor Filter]
      │                     (Classifies Original vs. Spicy)
      │                               │
      └──────────────┬────────────────┘
                     ▼
           [3D Depth Backprojection]
           (Samples depth cloud inside box)
                     │
                     ▼
           [Physical Depth Radius Compensation (+3.18cm)]
                     │
                     ▼
           [PCA Axial Fitting & Pose State Classifier]
           (Standing vs. Rolling on Table)
                     │
                     ▼
           [4-Quadrant Telemetry HUD Composition]
```
