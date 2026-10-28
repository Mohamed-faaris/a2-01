# Hand-Eye Calibration Report - Method: `2d_corner_ba`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `2d_corner_ba`
- **Runtime**: `16.654 s`
- **2D Reprojection Error**: `8.05 px RMS` (Mean: `6.35 px`, Max: `18.94 px`)
- **3D Position Residual**: `44.61 mm RMS` (Mean: `37.97 mm`, Max: `125.08 mm`)
- **3D Rotation Residual**: `46.58° RMS` (Mean: `25.89°`, Max: `156.87°`)
- **Workpiece Box Coplanarity RMSE**: `0.84 mm`
- **Workpiece [box] Coplanarity RMSE**: `0.84 mm` (Tags: `42, 43, 44, 45`)
- **Workpiece [box2] Coplanarity RMSE**: `5.07 mm` (Tags: `50, 51, 52, 53`)
- **Total Tool Tag Sightings**: `140` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.904, -0.354, +0.783] m`
- **USB Position $(X, Y, Z)$**: `[+1.690, -0.960, +0.799] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 40** | `tcp` | `[-101.9,   +1.0, -126.7]` | `162.6 mm` |
| **Tag 62** | `tcp` | `[ -31.5,  -74.3,  -16.0]` | `82.3 mm` |
| **Tag 63** | `tcp` | `[ +41.0,  -71.6,  -15.9]` | `84.0 mm` |
| **Tag 64** | `tcp` | `[ -34.1,   -0.5,  -18.4]` | `38.7 mm` |
| **Tag 65** | `tcp` | `[ +38.4,   +1.4,  -18.5]` | `42.6 mm` |
| **Tag 66** | `tcp` | `[ -39.4,  +72.3,  -19.9]` | `84.6 mm` |
| **Tag 67** | `tcp` | `[ +35.4,  +76.4,  -21.0]` | `86.8 mm` |

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 62 | ` 37.21 mm` | `59.10°` | ` 2.06 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 63 | ` 36.63 mm` | ` 3.77°` | ` 1.56 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 64 | `  5.96 mm` | `62.67°` | ` 1.86 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 65 | ` 18.27 mm` | ` 4.16°` | ` 1.99 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 66 | ` 18.77 mm` | `61.18°` | ` 1.60 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 67 | ` 60.71 mm` | `12.54°` | ` 1.43 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 62 | ` 39.03 mm` | `98.65°` | ` 1.02 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 63 | ` 53.18 mm` | `97.52°` | ` 1.55 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 64 | ` 19.63 mm` | ` 2.21°` | ` 1.50 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 65 | ` 67.32 mm` | ` 6.05°` | ` 1.45 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 66 | ` 42.85 mm` | ` 4.86°` | ` 1.47 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 67 | ` 38.09 mm` | ` 2.18°` | ` 0.87 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 62 | ` 40.51 mm` | `103.64°` | ` 0.98 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 63 | ` 89.59 mm` | `101.64°` | ` 1.63 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 64 | ` 42.81 mm` | ` 3.54°` | ` 1.81 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 65 | `113.70 mm` | ` 8.85°` | ` 2.27 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 66 | ` 31.66 mm` | ` 3.69°` | ` 1.80 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 67 | ` 36.44 mm` | ` 1.95°` | ` 1.27 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 62 | ` 46.52 mm` | ` 0.68°` | ` 0.81 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 63 | ` 33.05 mm` | ` 1.44°` | ` 2.07 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 64 | ` 30.97 mm` | `127.07°` | ` 1.03 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 65 | ` 17.14 mm` | ` 0.45°` | ` 0.58 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 66 | ` 18.71 mm` | ` 1.48°` | ` 0.74 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 67 | ` 18.09 mm` | `115.60°` | ` 0.98 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 40 | `  7.47 mm` | ` 3.93°` | ` 0.72 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 62 | ` 42.07 mm` | ` 1.78°` | ` 1.84 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 63 | ` 30.11 mm` | ` 2.45°` | ` 1.65 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 64 | ` 37.16 mm` | ` 3.73°` | ` 1.57 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 65 | ` 16.78 mm` | ` 2.46°` | ` 1.16 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 66 | ` 36.54 mm` | ` 1.46°` | ` 1.34 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 67 | ` 34.22 mm` | ` 2.49°` | ` 1.23 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 62 | `  3.09 mm` | ` 1.93°` | ` 0.70 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 63 | ` 27.18 mm` | ` 4.05°` | ` 1.87 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 64 | `  7.03 mm` | `117.68°` | ` 1.06 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 65 | ` 15.73 mm` | ` 3.15°` | ` 1.44 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 66 | ` 31.60 mm` | `118.29°` | ` 1.15 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 67 | ` 14.29 mm` | ` 2.28°` | ` 1.42 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 62 | ` 18.04 mm` | `73.92°` | ` 0.86 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 63 | ` 19.17 mm` | `67.25°` | ` 1.40 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 64 | ` 22.30 mm` | `79.15°` | ` 1.63 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 65 | ` 22.69 mm` | `72.17°` | ` 1.49 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 66 | ` 26.85 mm` | `82.75°` | ` 1.68 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 67 | ` 27.61 mm` | `78.89°` | ` 1.71 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 62 | ` 44.12 mm` | `46.12°` | ` 1.95 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 63 | `  3.86 mm` | ` 7.75°` | ` 1.79 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 64 | ` 30.24 mm` | `52.48°` | ` 2.72 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 65 | ` 46.58 mm` | `43.97°` | ` 1.99 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 66 | ` 50.02 mm` | `57.24°` | ` 2.08 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 67 | ` 11.14 mm` | ` 4.65°` | ` 1.19 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 62 | `  1.81 mm` | `61.45°` | ` 1.02 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 63 | `  2.30 mm` | ` 6.48°` | ` 1.34 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 64 | ` 11.75 mm` | `60.89°` | ` 1.71 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 65 | ` 18.43 mm` | ` 6.90°` | ` 2.78 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 66 | ` 13.98 mm` | ` 1.58°` | ` 1.44 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 67 | ` 47.00 mm` | ` 5.06°` | ` 1.73 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 65 | ` 25.85 mm` | ` 6.42°` | ` 7.69 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 66 | ` 46.32 mm` | ` 1.78°` | ` 6.24 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 67 | ` 47.24 mm` | ` 4.00°` | ` 5.97 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 63 | ` 31.03 mm` | ` 3.85°` | ` 1.62 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 65 | ` 33.79 mm` | ` 4.88°` | ` 1.89 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 67 | ` 45.16 mm` | ` 3.61°` | ` 1.83 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 62 | `  4.71 mm` | `84.30°` | ` 3.09 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 63 | ` 17.49 mm` | ` 4.22°` | ` 2.58 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 64 | ` 15.74 mm` | ` 4.56°` | ` 3.12 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 65 | ` 15.65 mm` | ` 4.62°` | ` 2.61 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 66 | ` 16.42 mm` | ` 4.79°` | ` 3.98 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 67 | ` 40.89 mm` | ` 3.20°` | ` 3.23 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 62 | ` 18.54 mm` | `122.54°` | ` 6.38 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 63 | ` 25.82 mm` | `124.30°` | ` 7.45 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 64 | ` 70.41 mm` | ` 4.13°` | ` 4.86 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 65 | ` 82.07 mm` | ` 3.89°` | ` 6.46 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 66 | ` 15.11 mm` | ` 0.75°` | ` 4.41 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 67 | ` 38.12 mm` | ` 1.65°` | ` 5.58 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 62 | ` 66.12 mm` | ` 1.59°` | ` 6.09 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 63 | ` 63.71 mm` | ` 2.13°` | ` 7.74 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 64 | ` 68.41 mm` | ` 3.88°` | ` 5.37 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 65 | ` 75.78 mm` | ` 2.63°` | ` 6.81 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 66 | ` 10.33 mm` | ` 0.52°` | ` 7.14 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 67 | ` 37.43 mm` | ` 1.19°` | ` 5.58 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 62 | ` 71.63 mm` | ` 1.31°` | ` 7.56 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 63 | ` 73.95 mm` | ` 1.19°` | ` 9.22 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 64 | ` 31.84 mm` | `132.91°` | ` 6.75 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 65 | ` 59.95 mm` | ` 1.95°` | ` 8.00 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 66 | ` 60.85 mm` | ` 2.80°` | ` 7.32 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 67 | ` 40.95 mm` | ` 0.89°` | ` 8.10 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 62 | ` 29.22 mm` | `125.30°` | ` 8.40 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 63 | ` 12.06 mm` | ` 1.74°` | ` 8.65 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 64 | ` 39.99 mm` | ` 2.70°` | ` 8.15 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 65 | ` 36.53 mm` | `117.11°` | ` 8.59 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 66 | ` 24.77 mm` | ` 2.24°` | ` 8.64 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 67 | ` 13.92 mm` | ` 2.08°` | ` 8.74 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 40 | ` 14.99 mm` | ` 4.93°` | `13.52 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 62 | ` 50.87 mm` | ` 2.95°` | `13.33 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 63 | ` 53.18 mm` | ` 2.77°` | `13.81 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 64 | ` 41.53 mm` | ` 4.09°` | `13.12 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 65 | ` 44.44 mm` | ` 2.50°` | `12.98 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 66 | ` 47.95 mm` | ` 1.51°` | `11.74 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 67 | ` 56.80 mm` | ` 2.62°` | `14.04 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 62 | ` 55.14 mm` | `66.73°` | ` 9.81 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 63 | ` 44.84 mm` | ` 5.85°` | ` 9.69 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 64 | ` 39.93 mm` | `69.43°` | ` 9.45 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 65 | ` 46.48 mm` | ` 7.53°` | ` 8.87 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 66 | ` 64.24 mm` | `63.22°` | ` 8.54 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 67 | ` 65.84 mm` | ` 7.30°` | ` 9.98 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 62 | ` 21.23 mm` | `38.63°` | `10.03 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 63 | ` 21.83 mm` | ` 4.24°` | ` 9.08 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 64 | ` 12.70 mm` | `42.42°` | `10.11 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 65 | ` 12.27 mm` | ` 6.10°` | ` 9.81 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 66 | ` 42.23 mm` | `26.69°` | ` 9.96 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 67 | ` 20.35 mm` | `18.69°` | `10.21 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 62 | ` 27.07 mm` | ` 1.73°` | ` 9.27 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 63 | ` 38.74 mm` | `76.63°` | ` 9.36 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 64 | ` 29.46 mm` | ` 7.18°` | ` 9.02 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 65 | ` 28.51 mm` | ` 1.96°` | ` 7.72 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 66 | ` 21.21 mm` | ` 5.40°` | ` 9.99 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 67 | ` 29.55 mm` | ` 4.16°` | ` 8.27 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 63 | ` 23.25 mm` | `49.48°` | `12.82 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 65 | ` 57.32 mm` | `33.71°` | `11.45 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 66 | ` 38.44 mm` | `11.79°` | ` 8.75 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 67 | ` 27.50 mm` | ` 6.07°` | ` 8.17 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_15x_usb` | Tag 40 | ` 19.45 mm` | ` 7.30°` | `18.56 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 40 | ` 21.85 mm` | ` 3.67°` | `18.94 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 62 | `125.08 mm` | ` 1.75°` | `12.73 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 63 | ` 76.06 mm` | ` 0.59°` | `14.75 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 64 | `105.33 mm` | `156.87°` | `12.96 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 65 | ` 85.05 mm` | ` 1.96°` | `12.83 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 66 | ` 77.03 mm` | ` 0.92°` | `11.73 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 67 | ` 67.80 mm` | ` 0.87°` | `13.90 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 63 | ` 65.77 mm` | `66.54°` | `16.32 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 65 | ` 78.12 mm` | `17.98°` | `16.22 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 66 | ` 74.56 mm` | ` 7.01°` | `14.40 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 67 | ` 88.76 mm` | ` 7.40°` | `15.98 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 62 | ` 51.96 mm` | ` 1.80°` | `13.07 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 63 | ` 24.10 mm` | ` 2.43°` | `13.07 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 64 | ` 16.51 mm` | ` 1.43°` | `12.96 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 65 | ` 54.29 mm` | ` 2.29°` | `13.00 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 66 | ` 25.97 mm` | ` 2.60°` | `14.34 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 67 | ` 27.70 mm` | ` 3.45°` | `14.50 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_30x_usb` | Tag 66 | ` 36.21 mm` | ` 6.93°` | `12.97 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_30x_usb` | Tag 67 | ` 30.79 mm` | ` 3.28°` | `14.36 px` |
