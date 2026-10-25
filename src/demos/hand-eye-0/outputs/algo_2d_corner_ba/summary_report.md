# Hand-Eye Calibration Report - Method: `2d_corner_ba`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `2d_corner_ba`
- **Runtime**: `1.226 s`
- **2D Reprojection Error**: `0.70 px RMS` (Mean: `0.64 px`, Max: `1.13 px`)
- **3D Position Residual**: `8.21 mm RMS` (Mean: `5.67 mm`, Max: `24.96 mm`)
- **3D Rotation Residual**: `3.73° RMS` (Mean: `2.60°`, Max: `13.18°`)
- **Workpiece Box Coplanarity RMSE**: `2.11 mm`
- **Total Tool Tag Sightings**: `20` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.933, -0.313, +0.793] m`
- **USB Position $(X, Y, Z)$**: `[+1.984, -0.197, +0.306] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 20** | `tcp` | `[ +48.1,  +15.2,   -5.7]` | `50.7 mm` |
| **Tag 21** | `tcp` | `[ +14.1,  +14.6,  -27.1]` | `33.8 mm` |
| **Tag 40** | `tcp` | `[ -86.2,   +3.3, -137.9]` | `162.7 mm` |

---

## 4. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_ready` | Tag 20 | `  4.78 mm` | ` 1.84°` | ` 0.60 px` |
| `pose01_ready` | Tag 40 | `  0.69 mm` | ` 1.25°` | ` 0.46 px` |
| `pose02_yaw_m40` | Tag 40 | `  3.72 mm` | ` 1.56°` | ` 1.07 px` |
| `pose04_yaw_m20` | Tag 20 | ` 16.20 mm` | ` 1.42°` | ` 0.42 px` |
| `pose04_yaw_m20` | Tag 40 | `  4.39 mm` | ` 1.18°` | ` 0.60 px` |
| `pose05_yaw_p20` | Tag 40 | `  0.97 mm` | ` 1.58°` | ` 1.02 px` |
| `pose06_pitch_up15` | Tag 20 | `  7.79 mm` | ` 1.79°` | ` 0.97 px` |
| `pose06_pitch_up15` | Tag 40 | `  4.21 mm` | ` 2.04°` | ` 0.26 px` |
| `pose07_pitch_dn15` | Tag 20 | ` 10.72 mm` | `13.18°` | ` 0.80 px` |
| `pose07_pitch_dn15` | Tag 40 | `  0.87 mm` | ` 4.62°` | ` 0.36 px` |
| `pose08_wrist_dn25` | Tag 20 | `  8.87 mm` | ` 3.72°` | ` 0.25 px` |
| `pose08_wrist_dn25` | Tag 40 | `  1.25 mm` | ` 4.44°` | ` 0.54 px` |
| `pose09_wrist_up25` | Tag 40 | `  5.44 mm` | ` 2.10°` | ` 0.50 px` |
| `pose10_roll_m45` | Tag 40 | `  3.03 mm` | ` 1.80°` | ` 0.72 px` |
| `pose11_roll_p45` | Tag 40 | `  1.17 mm` | ` 1.30°` | ` 1.13 px` |
| `pose16_comb_5` | Tag 20 | ` 24.96 mm` | ` 0.76°` | ` 1.07 px` |
| `pose16_comb_5` | Tag 21 | `  0.09 mm` | ` 0.04°` | ` 0.22 px` |
| `pose16_comb_5` | Tag 40 | `  6.71 mm` | ` 2.21°` | ` 0.70 px` |
| `pose18_ready_return` | Tag 20 | `  6.75 mm` | ` 2.84°` | ` 0.54 px` |
| `pose18_ready_return` | Tag 40 | `  0.72 mm` | ` 2.39°` | ` 0.54 px` |
