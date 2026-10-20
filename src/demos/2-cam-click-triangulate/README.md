# Two-Camera Interactive Multi-Point Click Triangulator with Realtime Feeds & Custom Calibrations

Interactive 3D manual point localization and triangulation tool for stereo cameras and dual-view tabletop scenes with automatic AprilTag reference coordinate localization.

Adheres strictly to the calibration specifications in [`custom-calibrator-npz.md`](/home/mfk01/Projects/econ-systems/tools/calibrator/docs/custom-calibrator-npz.md).

---

## Key Features

1. **Custom Calibration Support (`--calib0`, `--calib1`)**:
   - Loads calibration parameters directly from:
     - `.npz` archive (`camera_matrix`, `dist_coeffs`, `image_size`, `rms_error`, `card_name`, `device_node`, `stream_type`).
     - `.json` OpenCV calibration export.
     - `.yaml` ROS `camera_info.yaml`.
     - Named profile from `tools/calibrator/output/` or `calibrations/` (e.g. `--calib0 econ-lap-0 --calib1 rl720`).
   - Automatically checks stream type compatibility (`RGB / Color`, `Color (MJPEG)`, `Depth`, etc.).
   - Applies live frame undistortion via `cv2.undistort()` using loaded camera matrix and distortion coefficients.

2. **Realtime Live Feeds (`--realtime` / `--live`)**:
   - Zero-latency background grabber threads (`NonQueuingCameraReader`) draining V4L2 buffers continuously.
   - **Automatic Camera Node Resolution**: Automatically maps the camera device nodes (`/dev/video*`) matching the calibration's `card_name` and `stream_type`. If a camera's device node changed, logs `[CAMERA MOVED / RE-MAPPED]` and connects to the active node.

3. **Inspection Commands**:
   - `--list-cams`: Scans all connected V4L2 camera devices, cards, and video capture capabilities.
   - `--list-calibs`: Scans and lists all available calibration profiles with resolutions, RMS reprojection errors, and cards.

4. **Multi-Point Interactive Workflow**:
   - Starts with **Point 1** ready immediately.
   - Left-click on Camera 1 (Top-Left) to set Point P1 $(x, y)$.
   - Left-click on Camera 2 (Bottom-Left) to set Point P2 $(x, y)$.
   - Real-time 3D coordinates $(X, Y, Z)$ relative to the reference AprilTag corner/origin.

5. **Keyboard Shortcuts for Point Management**:
   - `n`, `a`, `Space`, `+`: **Create new point** (automatically selects it for clicking).
   - `p`, `[`, `Left Arrow`: **Select PREVIOUS point**.
   - `]`, `Tab`, `Right Arrow`: **Select NEXT point**.
   - `1` – `9`: **Direct jump** to Point 1..9.
   - `c`: **Clear clicks** for the current active point.
   - `d`, `x`, `Delete`, `Backspace`: **Delete** the current active point.
   - `s`: **Save all points** to JSON immediately.
   - `q`, `ESC`: **Quit** and auto-save all points to JSON.

---

## Keyboard Shortcuts Summary

| Shortcut Key | Action |
|---|---|
| `n` / `a` / `Space` / `+` | **Create New Point** (appends next point and activates it) |
| `p` / `[` / `Left Arrow` | **Select Previous Point** |
| `]` / `Tab` / `Right Arrow` | **Select Next Point** |
| `1` – `9` | **Jump directly** to Point 1 – 9 |
| `c` | **Clear** current active point clicks |
| `d` / `x` / `Delete` | **Delete** current active point |
| `s` | **Save all points** to JSON immediately |
| `q` / `ESC` | **Quit** and auto-save JSON |

---

## Usage Examples

### 1. List Available Cameras and Calibrations
```bash
# Scan connected V4L2 cameras & cards
python3 src/demos/2-cam-click-triangulate/main.py --list-cams

# Scan available calibration profiles
python3 src/demos/2-cam-click-triangulate/main.py --list-calibs
```

### 2. Live Realtime Streaming with Custom Calibrations
```bash
# Auto-resolve nodes for econ-lap-0 and rl720, streaming live at 10 FPS
python3 src/demos/2-cam-click-triangulate/main.py \
  --realtime \
  --calib0 econ-lap-0 \
  --calib1 rl720 \
  --fps 10 \
  -o live_measurement.json
```

### 3. Live Realtime Streaming with Explicit Camera Nodes
```bash
python3 src/demos/2-cam-click-triangulate/main.py \
  --realtime \
  --cam0 /dev/video0 \
  --cam1 /dev/video3 \
  --calib0 /path/to/cam0_calibration.npz \
  --calib1 /path/to/cam1_calibration.npz \
  -o realtime_points.json
```

### 4. Offline Scene / Dataset Triangulation
```bash
python3 src/demos/2-cam-click-triangulate/main.py \
  --scene experiment/two_camera_rel_pose/001 \
  -o tmp/scene001_points.json
```

---

## JSON Output Format

```json
{
  "scene": "scene_a",
  "saved_at": "2026-10-16T12:00:00Z",
  "total_points": 2,
  "completed_points": 2,
  "reference_tag": {
    "id": 0,
    "origin": "corner",
    "size_m": 0.134
  },
  "cameras": {
    "camera0": {
      "name": "econ-lap-0",
      "card_name": "ASUS HD webcam: ASUS HD webcam (usb-0000:00:14.0-5)",
      "device_node": "/dev/video0",
      "stream_type": "Color (MJPEG)",
      "rms_error_px": 0.239,
      "resolution": [1280, 720],
      "distortion_model": "standard"
    },
    "camera1": {
      "name": "rl720",
      "card_name": "Intel(R) RealSense(TM) Depth Ca (usb-0000:00:14.0-2)",
      "device_node": "/dev/video6",
      "stream_type": "RGB / Color",
      "rms_error_px": 0.273,
      "resolution": [1280, 720],
      "distortion_model": "standard"
    }
  },
  "points": [
    {
      "id": 1,
      "name": "Point 1",
      "cam0_px": [320.02, 239.97],
      "cam1_px": [319.97, 239.95],
      "status": "complete",
      "triangulated": {
        "x_cm": 6.71,
        "y_cm": 6.69,
        "z_cm": -0.08,
        "euclidean_distance_cm": 9.47,
        "reproj_err_px": 0.02,
        "ray_skew_cm": 0.01,
        "ray_angle_deg": 24.5,
        "est_depth_err_cm": 1.0,
        "reference_tag_id": 0
      }
    }
  ]
}
```
