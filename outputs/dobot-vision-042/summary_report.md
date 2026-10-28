# Hand-Eye Calibration Report - Method: `coplanar_plate_ba`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `coplanar_plate_ba`
- **Runtime**: `88.419 s`
- **2D Reprojection Error**: `1.78 px RMS` (Mean: `1.65 px`, Max: `4.93 px`)
- **3D Position Residual**: `38.01 mm RMS` (Mean: `31.67 mm`, Max: `115.53 mm`)
- **3D Rotation Residual**: `46.28° RMS` (Mean: `25.85°`, Max: `154.61°`)
- **Workpiece Box Coplanarity RMSE**: `0.84 mm`
- **Workpiece [box] Coplanarity RMSE**: `0.84 mm` (Tags: `42, 43, 44, 45`)
- **Workpiece [box2] Coplanarity RMSE**: `5.07 mm` (Tags: `50, 51, 52, 53`)
- **Total Tool Tag Sightings**: `140` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.899, -0.366, +0.786] m`
- **USB Position $(X, Y, Z)$**: `[+1.671, -0.998, +0.813] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 40** | `tcp` | `[ -90.7,   +6.5, -136.7]` | `164.2 mm` |
| **Tag 62** | `tcp` | `[ -34.4,  -70.1,  -19.7]` | `80.5 mm` |
| **Tag 63** | `tcp` | `[ +39.2,  -66.5,  -19.2]` | `79.5 mm` |
| **Tag 64** | `tcp` | `[ -38.9,   +2.5,  -22.0]` | `44.8 mm` |
| **Tag 65** | `tcp` | `[ +35.0,   +5.5,  -21.5]` | `41.4 mm` |
| **Tag 66** | `tcp` | `[ -42.8,  +76.8,  -24.4]` | `91.2 mm` |
| **Tag 67** | `tcp` | `[ +30.7,  +80.5,  -23.9]` | `89.4 mm` |

---

## 4. Parameter Observability & Covariance Uncertainty
- **Jacobian Condition Number $\kappa$**: `522343962.52`
- **Minimum Singular Value $\sigma_{min}$**: `0.0001`
- **Top Singular Values**: `[55668.824, 52402.001, 39382.442, 32697.633, 18333.419, 15575.82, 14342.381, 13308.589]`
- **D415 Estimated Position 1-$\sigma$ Uncertainty $(X, Y, Z)$**: `±[0.98, 1.6, 1.26] mm`
- **D415 Estimated Rotation 1-$\sigma$ Uncertainty $(Roll, Pitch, Yaw)$**: `±[0.05, 0.06, 0.06]°`
- **USB Estimated Position 1-$\sigma$ Uncertainty $(X, Y, Z)$**: `±[1.05, 0.9, 0.87] mm`
- **USB Estimated Rotation 1-$\sigma$ Uncertainty $(Roll, Pitch, Yaw)$**: `±[0.03, 0.04, 0.03]°`

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 62 | ` 39.48 mm` | `58.76°` | ` 1.85 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 63 | ` 39.23 mm` | ` 5.63°` | ` 2.13 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 64 | `  7.71 mm` | `60.40°` | ` 2.13 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 65 | ` 16.37 mm` | ` 3.36°` | ` 1.73 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 66 | ` 15.18 mm` | `62.12°` | ` 1.73 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 67 | ` 62.31 mm` | `12.90°` | ` 1.37 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 62 | ` 40.62 mm` | `98.02°` | ` 1.09 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 63 | ` 54.99 mm` | `96.41°` | ` 1.44 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 64 | ` 20.16 mm` | ` 2.92°` | ` 1.04 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 65 | ` 68.19 mm` | ` 5.11°` | ` 1.47 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 66 | ` 45.61 mm` | ` 6.10°` | ` 1.58 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 67 | ` 38.89 mm` | ` 2.88°` | ` 0.98 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 62 | ` 43.00 mm` | `103.08°` | ` 1.14 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 63 | ` 92.29 mm` | `100.55°` | ` 1.54 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 64 | ` 44.30 mm` | ` 4.14°` | ` 1.63 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 65 | `115.53 mm` | ` 7.56°` | ` 2.08 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 66 | ` 35.36 mm` | ` 4.54°` | ` 1.31 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 67 | ` 38.29 mm` | ` 2.60°` | ` 1.37 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 62 | ` 46.34 mm` | ` 0.74°` | ` 1.12 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 63 | ` 33.58 mm` | ` 1.76°` | ` 1.94 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 64 | ` 32.77 mm` | `124.83°` | ` 1.46 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 65 | ` 16.30 mm` | ` 1.82°` | ` 0.95 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 66 | ` 17.92 mm` | ` 0.38°` | ` 0.88 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 67 | ` 16.76 mm` | `116.52°` | ` 1.18 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 40 | `  9.07 mm` | ` 0.74°` | ` 1.42 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 62 | ` 42.40 mm` | ` 2.18°` | ` 1.38 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 63 | ` 31.26 mm` | ` 3.78°` | ` 1.35 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 64 | ` 36.28 mm` | ` 1.64°` | ` 1.06 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 65 | ` 16.94 mm` | ` 3.90°` | ` 2.04 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 66 | ` 38.00 mm` | ` 3.06°` | ` 1.97 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 67 | ` 33.73 mm` | ` 3.40°` | ` 2.10 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 62 | `  1.81 mm` | ` 1.01°` | ` 0.99 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 63 | ` 29.51 mm` | ` 4.62°` | ` 1.83 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 64 | `  6.56 mm` | `116.39°` | ` 2.63 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 65 | ` 17.89 mm` | ` 2.74°` | ` 2.91 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 66 | ` 34.20 mm` | `118.01°` | ` 1.78 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 67 | ` 15.94 mm` | ` 2.93°` | ` 2.78 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 62 | ` 21.14 mm` | `74.23°` | ` 1.22 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 63 | ` 22.34 mm` | `69.29°` | ` 1.37 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 64 | ` 25.37 mm` | `77.92°` | ` 2.09 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 65 | ` 25.67 mm` | `73.36°` | ` 1.12 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 66 | ` 30.92 mm` | `82.13°` | ` 2.00 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 67 | ` 30.30 mm` | `78.70°` | ` 1.70 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 62 | ` 48.63 mm` | `46.51°` | ` 1.99 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 63 | `  8.90 mm` | ` 7.51°` | ` 3.16 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 64 | ` 35.03 mm` | `51.19°` | ` 3.00 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 65 | ` 50.93 mm` | `44.93°` | ` 2.74 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 66 | ` 55.79 mm` | `56.27°` | ` 3.54 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 67 | ` 15.60 mm` | ` 5.77°` | ` 2.32 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 62 | `  1.78 mm` | `61.20°` | ` 1.40 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 63 | `  5.76 mm` | ` 5.42°` | ` 1.88 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 64 | ` 14.28 mm` | `58.62°` | ` 1.64 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 65 | ` 15.82 mm` | ` 6.63°` | ` 3.05 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 66 | ` 18.29 mm` | ` 1.59°` | ` 0.84 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 67 | ` 49.37 mm` | ` 5.62°` | ` 1.52 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 65 | ` 24.36 mm` | ` 6.18°` | ` 4.93 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 66 | ` 47.31 mm` | ` 0.80°` | ` 3.22 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 67 | ` 46.20 mm` | ` 3.09°` | ` 3.34 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 63 | ` 35.09 mm` | ` 5.28°` | ` 1.76 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 65 | ` 36.74 mm` | ` 6.41°` | ` 1.74 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 67 | ` 47.63 mm` | ` 3.88°` | ` 1.16 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 62 | ` 17.26 mm` | `84.14°` | ` 1.25 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 63 | ` 32.85 mm` | ` 4.92°` | ` 1.53 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 64 | `  0.26 mm` | ` 6.95°` | ` 1.68 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 65 | ` 31.02 mm` | ` 3.74°` | ` 1.56 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 66 | `  3.90 mm` | ` 6.07°` | ` 1.45 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 67 | ` 55.46 mm` | ` 2.06°` | ` 1.24 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 62 | `  4.16 mm` | `122.39°` | ` 1.00 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 63 | `  3.59 mm` | `123.25°` | ` 1.14 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 64 | ` 50.48 mm` | ` 2.47°` | ` 0.81 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 65 | ` 61.13 mm` | ` 3.63°` | ` 1.14 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 66 | `  4.08 mm` | ` 2.43°` | ` 1.16 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 67 | ` 17.17 mm` | ` 2.55°` | ` 1.18 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 62 | ` 48.55 mm` | ` 1.52°` | ` 0.72 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 63 | ` 44.85 mm` | ` 1.76°` | ` 0.96 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 64 | ` 50.41 mm` | ` 1.82°` | ` 0.93 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 65 | ` 56.77 mm` | ` 2.40°` | ` 0.99 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 66 | ` 14.73 mm` | ` 1.93°` | ` 1.41 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 67 | ` 18.59 mm` | ` 2.09°` | ` 0.95 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 62 | ` 50.13 mm` | ` 0.47°` | ` 1.06 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 63 | ` 51.61 mm` | ` 2.29°` | ` 1.64 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 64 | `  8.64 mm` | `131.45°` | ` 1.46 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 65 | ` 36.81 mm` | ` 1.77°` | ` 0.94 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 66 | ` 40.94 mm` | ` 4.64°` | ` 1.76 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 67 | ` 17.55 mm` | ` 2.16°` | ` 1.22 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 62 | `  9.37 mm` | `124.83°` | ` 1.46 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 63 | ` 25.20 mm` | ` 2.50°` | ` 1.52 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 64 | ` 20.31 mm` | ` 0.99°` | ` 0.88 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 65 | ` 16.49 mm` | `117.35°` | ` 0.85 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 66 | `  8.08 mm` | ` 3.68°` | ` 1.63 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 67 | `  7.64 mm` | ` 1.14°` | ` 0.84 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 40 | `  4.02 mm` | ` 1.75°` | ` 0.91 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 62 | ` 38.22 mm` | ` 2.61°` | ` 2.00 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 63 | ` 41.18 mm` | ` 3.71°` | ` 1.79 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 64 | ` 29.01 mm` | ` 1.66°` | ` 1.00 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 65 | ` 33.04 mm` | ` 3.18°` | ` 1.43 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 66 | ` 40.12 mm` | ` 3.34°` | ` 1.81 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 67 | ` 46.79 mm` | ` 3.37°` | ` 1.67 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 62 | ` 45.48 mm` | `66.45°` | ` 1.66 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 63 | ` 35.02 mm` | ` 6.81°` | ` 1.63 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 64 | ` 30.76 mm` | `67.06°` | ` 1.03 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 65 | ` 37.75 mm` | ` 8.09°` | ` 1.64 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 66 | ` 59.37 mm` | `63.66°` | ` 1.56 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 67 | ` 58.52 mm` | ` 7.76°` | ` 1.53 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 62 | ` 12.65 mm` | `38.26°` | ` 1.73 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 63 | ` 13.61 mm` | ` 5.27°` | ` 1.84 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 64 | `  3.54 mm` | `40.11°` | ` 1.79 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 65 | `  2.41 mm` | ` 4.98°` | ` 0.80 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 66 | ` 40.07 mm` | `27.52°` | ` 1.85 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 67 | ` 13.22 mm` | `17.48°` | ` 1.38 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 62 | ` 12.38 mm` | ` 2.79°` | ` 1.41 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 63 | ` 24.34 mm` | `76.39°` | ` 1.07 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 64 | ` 15.60 mm` | ` 9.60°` | ` 2.34 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 65 | ` 14.68 mm` | ` 1.72°` | ` 1.93 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 66 | `  9.09 mm` | ` 6.78°` | ` 1.62 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 67 | ` 39.20 mm` | ` 2.82°` | ` 1.79 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 63 | `  6.18 mm` | `48.67°` | ` 1.39 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 65 | ` 36.80 mm` | `33.64°` | ` 3.00 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 66 | ` 21.67 mm` | `13.26°` | ` 1.79 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 67 | `  7.56 mm` | ` 7.01°` | ` 2.10 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_15x_usb` | Tag 40 | `  7.34 mm` | ` 3.44°` | ` 0.71 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 40 | `  1.13 mm` | ` 0.86°` | ` 1.16 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 62 | `104.10 mm` | ` 0.76°` | ` 1.92 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 63 | ` 54.08 mm` | ` 2.26°` | ` 1.91 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 64 | ` 83.17 mm` | `154.61°` | ` 1.91 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 65 | ` 62.98 mm` | ` 1.57°` | ` 2.64 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 66 | ` 58.33 mm` | ` 3.09°` | ` 3.57 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 67 | ` 45.55 mm` | ` 1.92°` | ` 1.75 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 63 | ` 41.01 mm` | `64.69°` | ` 2.36 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 65 | ` 53.81 mm` | `17.21°` | ` 2.64 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 66 | ` 52.54 mm` | ` 8.02°` | ` 1.15 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 67 | ` 64.51 mm` | ` 7.60°` | ` 1.45 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 62 | ` 39.17 mm` | ` 2.79°` | ` 1.77 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 63 | ` 28.47 mm` | ` 2.95°` | ` 1.44 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 64 | ` 15.53 mm` | ` 3.55°` | ` 1.39 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 65 | ` 62.76 mm` | ` 2.08°` | ` 1.69 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 66 | ` 13.06 mm` | ` 4.18°` | ` 1.51 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 67 | ` 12.45 mm` | ` 2.37°` | ` 1.25 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_30x_usb` | Tag 66 | `  9.53 mm` | ` 7.74°` | ` 1.59 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_30x_usb` | Tag 67 | ` 36.36 mm` | ` 2.68°` | ` 0.87 px` |
