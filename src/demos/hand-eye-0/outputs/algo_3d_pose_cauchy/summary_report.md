# Hand-Eye Calibration Report - Method: `3d_pose_cauchy`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `3d_pose_cauchy`
- **Runtime**: `6.795 s`
- **2D Reprojection Error**: `68.50 px RMS` (Mean: `52.36 px`, Max: `181.00 px`)
- **3D Position Residual**: `112.94 mm RMS` (Mean: `85.77 mm`, Max: `286.33 mm`)
- **3D Rotation Residual**: `18.10° RMS` (Mean: `13.78°`, Max: `36.05°`)
- **Workpiece Box Coplanarity RMSE**: `2.11 mm`
- **Total Tool Tag Sightings**: `20` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.272, +0.451, +1.204] m`
- **USB Position $(X, Y, Z)$**: `[+1.623, +0.429, +0.844] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 20** | `tcp` | `[ +95.0,  -84.0, -194.5]` | `232.2 mm` |
| **Tag 21** | `tcp` | `[+168.3,  -53.9, -147.0]` | `229.9 mm` |
| **Tag 40** | `tcp` | `[ -91.7,  -13.1, -174.1]` | `197.2 mm` |

---

## 4. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_ready` | Tag 20 | ` 55.52 mm` | ` 0.37°` | `42.09 px` |
| `pose01_ready` | Tag 40 | `  2.40 mm` | ` 0.98°` | ` 1.92 px` |
| `pose02_yaw_m40` | Tag 40 | `286.33 mm` | `36.05°` | `181.00 px` |
| `pose04_yaw_m20` | Tag 20 | `131.75 mm` | `18.61°` | `91.34 px` |
| `pose04_yaw_m20` | Tag 40 | `155.77 mm` | `19.58°` | `93.62 px` |
| `pose05_yaw_p20` | Tag 40 | `206.63 mm` | `16.48°` | `20.29 px` |
| `pose06_pitch_up15` | Tag 20 | ` 65.69 mm` | ` 0.45°` | `57.99 px` |
| `pose06_pitch_up15` | Tag 40 | ` 88.14 mm` | ` 1.41°` | `51.87 px` |
| `pose07_pitch_dn15` | Tag 20 | `132.33 mm` | `12.67°` | `79.88 px` |
| `pose07_pitch_dn15` | Tag 40 | ` 81.65 mm` | ` 2.52°` | `36.93 px` |
| `pose08_wrist_dn25` | Tag 20 | ` 63.28 mm` | `22.45°` | `58.08 px` |
| `pose08_wrist_dn25` | Tag 40 | ` 39.96 mm` | `21.34°` | `22.69 px` |
| `pose09_wrist_up25` | Tag 40 | ` 55.41 mm` | `21.70°` | `31.17 px` |
| `pose10_roll_m45` | Tag 40 | ` 14.20 mm` | `30.34°` | `16.58 px` |
| `pose11_roll_p45` | Tag 40 | `  4.59 mm` | `32.04°` | `13.44 px` |
| `pose16_comb_5` | Tag 20 | `127.65 mm` | `19.90°` | `98.02 px` |
| `pose16_comb_5` | Tag 21 | `  0.01 mm` | ` 0.24°` | ` 0.23 px` |
| `pose16_comb_5` | Tag 40 | `145.16 mm` | `17.48°` | `106.27 px` |
| `pose18_ready_return` | Tag 20 | ` 56.62 mm` | ` 0.65°` | `41.99 px` |
| `pose18_ready_return` | Tag 40 | `  2.43 mm` | ` 0.33°` | ` 1.87 px` |
