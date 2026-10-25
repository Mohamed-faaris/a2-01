# Hand-Eye Calibration Report - Method: `rigid_cube_ba`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `rigid_cube_ba`
- **Runtime**: `1.125 s`
- **2D Reprojection Error**: `15.07 px RMS` (Mean: `4.00 px`, Max: `67.34 px`)
- **3D Position Residual**: `21.15 mm RMS` (Mean: `9.98 mm`, Max: `87.28 mm`)
- **3D Rotation Residual**: `6.80° RMS` (Mean: `3.94°`, Max: `24.98°`)
- **Workpiece Box Coplanarity RMSE**: `2.11 mm`
- **Total Tool Tag Sightings**: `20` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.932, -0.314, +0.793] m`
- **USB Position $(X, Y, Z)$**: `[+1.984, -0.197, +0.307] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 20** | `tcp` | `[ +47.9,  +15.2,   -5.8]` | `50.6 mm` |
| **Tag 21** | `tcp` | `[ +36.5,  +37.1,  +54.2]` | `75.2 mm` |
| **Tag 40** | `tcp` | `[ -86.3,   +3.3, -138.0]` | `162.7 mm` |

---

## 4. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_ready` | Tag 20 | `  4.63 mm` | ` 1.88°` | ` 0.61 px` |
| `pose01_ready` | Tag 40 | `  0.68 mm` | ` 1.23°` | ` 0.45 px` |
| `pose02_yaw_m40` | Tag 40 | `  3.83 mm` | ` 1.58°` | ` 1.11 px` |
| `pose04_yaw_m20` | Tag 20 | ` 16.13 mm` | ` 2.17°` | ` 0.50 px` |
| `pose04_yaw_m20` | Tag 40 | `  4.45 mm` | ` 1.19°` | ` 0.60 px` |
| `pose05_yaw_p20` | Tag 40 | `  0.94 mm` | ` 1.56°` | ` 1.01 px` |
| `pose06_pitch_up15` | Tag 20 | `  7.63 mm` | ` 1.78°` | ` 0.99 px` |
| `pose06_pitch_up15` | Tag 40 | `  4.23 mm` | ` 2.03°` | ` 0.26 px` |
| `pose07_pitch_dn15` | Tag 20 | ` 10.58 mm` | `13.92°` | ` 0.80 px` |
| `pose07_pitch_dn15` | Tag 40 | `  0.88 mm` | ` 4.59°` | ` 0.36 px` |
| `pose08_wrist_dn25` | Tag 20 | `  8.74 mm` | ` 3.81°` | ` 0.25 px` |
| `pose08_wrist_dn25` | Tag 40 | `  1.25 mm` | ` 4.42°` | ` 0.55 px` |
| `pose09_wrist_up25` | Tag 40 | `  5.43 mm` | ` 2.10°` | ` 0.50 px` |
| `pose10_roll_m45` | Tag 40 | `  3.00 mm` | ` 1.80°` | ` 0.71 px` |
| `pose11_roll_p45` | Tag 40 | `  1.18 mm` | ` 1.31°` | ` 1.14 px` |
| `pose16_comb_5` | Tag 20 | ` 24.79 mm` | ` 1.13°` | ` 0.99 px` |
| `pose16_comb_5` | Tag 21 | ` 87.28 mm` | `24.98°` | `67.34 px` |
| `pose16_comb_5` | Tag 40 | `  6.69 mm` | ` 2.22°` | ` 0.72 px` |
| `pose18_ready_return` | Tag 20 | `  6.60 mm` | ` 2.83°` | ` 0.56 px` |
| `pose18_ready_return` | Tag 40 | `  0.71 mm` | ` 2.37°` | ` 0.54 px` |
