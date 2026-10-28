# Hand-Eye Calibration Report - Method: `3d_pose_cauchy`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `3d_pose_cauchy`
- **Runtime**: `72.412 s`
- **2D Reprojection Error**: `15.12 px RMS` (Mean: `13.30 px`, Max: `38.33 px`)
- **3D Position Residual**: `38.77 mm RMS` (Mean: `31.94 mm`, Max: `112.58 mm`)
- **3D Rotation Residual**: `42.42° RMS` (Mean: `28.76°`, Max: `134.31°`)
- **Workpiece Box Coplanarity RMSE**: `0.84 mm`
- **Workpiece [box] Coplanarity RMSE**: `0.84 mm` (Tags: `42, 43, 44, 45`)
- **Workpiece [box2] Coplanarity RMSE**: `5.07 mm` (Tags: `50, 51, 52, 53`)
- **Total Tool Tag Sightings**: `140` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.908, -0.335, +0.847] m`
- **USB Position $(X, Y, Z)$**: `[+1.698, -0.943, +0.856] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 40** | `tcp` | `[ -63.3,   -5.4, -142.6]` | `156.1 mm` |
| **Tag 62** | `tcp` | `[  -9.5,  -86.2,   -5.7]` | `86.9 mm` |
| **Tag 63** | `tcp` | `[ +61.9,  -86.9,  -36.3]` | `112.7 mm` |
| **Tag 64** | `tcp` | `[  -9.8,  -12.6,  -20.3]` | `25.8 mm` |
| **Tag 65** | `tcp` | `[ +62.5,  -10.8,  -25.3]` | `68.3 mm` |
| **Tag 66** | `tcp` | `[ -17.1,  +64.3,  -22.0]` | `70.1 mm` |
| **Tag 67** | `tcp` | `[ +60.1,  +67.6,  -18.8]` | `92.3 mm` |

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 62 | `  9.82 mm` | `28.68°` | ` 4.36 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 63 | ` 23.74 mm` | ` 5.28°` | `11.54 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 64 | ` 28.96 mm` | `28.12°` | ` 6.17 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 65 | ` 50.75 mm` | ` 4.59°` | ` 6.04 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 66 | ` 53.61 mm` | `48.12°` | ` 6.95 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x` | Tag 67 | ` 21.34 mm` | `11.01°` | ` 2.99 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 62 | ` 11.25 mm` | `80.40°` | ` 8.84 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 63 | ` 38.35 mm` | `91.59°` | `17.72 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 64 | ` 22.04 mm` | `44.07°` | `13.51 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 65 | ` 34.05 mm` | ` 5.34°` | ` 9.48 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 66 | ` 11.88 mm` | `26.26°` | ` 9.31 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x` | Tag 67 | `  9.62 mm` | ` 3.23°` | ` 5.54 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 62 | `  2.59 mm` | `86.02°` | ` 7.98 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 63 | ` 68.54 mm` | `95.77°` | `14.52 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 64 | ` 10.34 mm` | `45.48°` | `14.19 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 65 | ` 81.70 mm` | ` 8.56°` | ` 4.97 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 66 | `  0.27 mm` | `25.33°` | ` 6.21 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x` | Tag 67 | `  3.53 mm` | ` 3.57°` | ` 2.45 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 62 | ` 18.75 mm` | `30.04°` | `10.23 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 63 | ` 23.99 mm` | ` 6.24°` | `14.50 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 64 | ` 53.90 mm` | `101.97°` | `16.16 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 65 | `  6.42 mm` | ` 0.02°` | ` 3.67 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 66 | ` 41.17 mm` | `21.85°` | ` 7.27 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x` | Tag 67 | `  9.20 mm` | `112.76°` | ` 5.04 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 40 | `  0.00 mm` | ` 0.00°` | ` 0.70 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 62 | ` 18.40 mm` | `29.12°` | ` 9.85 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 63 | ` 21.41 mm` | ` 5.11°` | `10.13 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 64 | ` 18.85 mm` | `41.22°` | `11.72 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 65 | `  3.79 mm` | ` 1.88°` | ` 2.88 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 66 | ` 16.29 mm` | `22.67°` | ` 6.43 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x` | Tag 67 | ` 14.26 mm` | ` 0.04°` | ` 7.09 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 62 | ` 29.95 mm` | `28.62°` | ` 9.80 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 63 | ` 22.84 mm` | ` 0.03°` | `10.49 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 64 | ` 23.40 mm` | `77.17°` | `10.99 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 65 | `  2.00 mm` | ` 4.37°` | ` 1.94 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 66 | ` 13.14 mm` | `94.85°` | ` 7.34 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x` | Tag 67 | `  8.25 mm` | ` 3.44°` | ` 2.95 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 62 | ` 16.29 mm` | `60.16°` | ` 5.62 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 63 | ` 14.22 mm` | `67.75°` | ` 7.16 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 64 | `  0.43 mm` | `42.80°` | ` 5.60 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 65 | `  2.94 mm` | `70.98°` | ` 1.69 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 66 | `  4.80 mm` | `59.27°` | ` 5.15 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x` | Tag 67 | `  2.62 mm` | `77.78°` | ` 2.44 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 62 | ` 16.49 mm` | `47.64°` | ` 6.44 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 63 | `  1.55 mm` | ` 3.42°` | ` 1.84 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 64 | ` 13.80 mm` | `23.75°` | ` 4.67 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 65 | ` 30.11 mm` | `42.96°` | ` 3.47 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 66 | ` 30.79 mm` | `34.75°` | ` 3.13 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x` | Tag 67 | ` 16.36 mm` | ` 3.45°` | ` 4.51 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 62 | ` 32.53 mm` | `31.27°` | ` 8.47 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 63 | `  7.95 mm` | ` 9.24°` | ` 5.58 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 64 | ` 11.10 mm` | `24.80°` | ` 7.86 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 65 | ` 36.30 mm` | ` 7.26°` | ` 7.71 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 66 | `  9.59 mm` | `23.63°` | ` 6.41 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x` | Tag 67 | ` 25.68 mm` | ` 3.54°` | ` 9.49 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 65 | ` 20.34 mm` | ` 6.53°` | `10.42 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 66 | ` 34.07 mm` | `20.87°` | ` 7.80 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x` | Tag 67 | ` 29.48 mm` | ` 6.48°` | ` 8.54 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 63 | ` 32.15 mm` | ` 7.63°` | ` 8.96 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 65 | ` 28.86 mm` | ` 4.26°` | `10.64 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x` | Tag 67 | ` 36.00 mm` | ` 3.16°` | `14.65 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 62 | ` 48.10 mm` | `62.09°` | `17.02 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 63 | ` 43.65 mm` | ` 8.78°` | `15.99 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 64 | ` 23.50 mm` | `48.22°` | `17.19 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 65 | ` 48.35 mm` | ` 5.12°` | `13.71 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 66 | ` 26.39 mm` | `27.29°` | `18.43 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_30x_usb` | Tag 67 | ` 79.70 mm` | ` 5.58°` | `16.85 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 62 | ` 35.33 mm` | `113.20°` | `16.25 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 63 | ` 38.62 mm` | `118.99°` | `22.65 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 64 | ` 39.10 mm` | `41.34°` | `18.96 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 65 | ` 52.73 mm` | ` 3.29°` | `14.99 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 66 | ` 30.40 mm` | `21.80°` | `16.58 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_15x_usb` | Tag 67 | ` 19.06 mm` | ` 2.05°` | `12.08 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 62 | ` 35.29 mm` | `30.19°` | `15.78 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 63 | ` 45.73 mm` | ` 4.22°` | `18.31 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 64 | ` 38.18 mm` | `41.36°` | `17.86 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 65 | ` 46.57 mm` | ` 2.53°` | `11.26 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 66 | ` 34.42 mm` | `21.54°` | `15.56 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_15x_usb` | Tag 67 | ` 17.65 mm` | ` 2.40°` | `12.16 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 62 | ` 44.82 mm` | `29.76°` | `14.03 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 63 | ` 62.49 mm` | ` 4.76°` | `21.55 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 64 | ` 14.09 mm` | `126.54°` | `19.44 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 65 | ` 40.47 mm` | ` 1.62°` | `10.93 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 66 | ` 42.33 mm` | `24.04°` | `15.26 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_15x_usb` | Tag 67 | ` 20.18 mm` | ` 2.17°` | ` 9.74 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 62 | ` 18.51 mm` | `97.58°` | `19.15 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 63 | ` 29.73 mm` | ` 6.34°` | `19.14 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 64 | ` 25.35 mm` | `42.12°` | `20.85 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 65 | ` 23.73 mm` | `117.19°` | `13.90 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 66 | ` 19.44 mm` | `25.11°` | `20.47 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_15x_usb` | Tag 67 | ` 21.26 mm` | ` 2.98°` | `17.88 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 40 | ` 28.33 mm` | ` 1.36°` | `25.38 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 62 | ` 37.93 mm` | `28.82°` | `26.34 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 63 | ` 50.95 mm` | ` 4.81°` | `24.61 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 64 | ` 34.81 mm` | `40.86°` | `25.30 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 65 | ` 37.65 mm` | ` 1.90°` | `22.39 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 66 | ` 40.16 mm` | `22.64°` | `25.43 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_15x_usb` | Tag 67 | ` 46.13 mm` | ` 0.82°` | `26.14 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 62 | ` 33.24 mm` | `36.84°` | `20.26 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 63 | ` 40.74 mm` | ` 7.60°` | `18.01 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 64 | ` 28.02 mm` | `30.76°` | `18.04 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 65 | ` 33.92 mm` | ` 6.61°` | `17.68 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 66 | ` 48.58 mm` | `48.08°` | `19.89 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_15x_usb` | Tag 67 | ` 46.05 mm` | ` 5.32°` | `20.97 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 62 | ` 23.78 mm` | ` 9.30°` | `15.87 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 63 | ` 20.45 mm` | ` 7.23°` | `10.84 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 64 | ` 25.02 mm` | `19.90°` | `14.60 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 65 | ` 23.18 mm` | ` 6.17°` | `14.44 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 66 | ` 30.95 mm` | `21.10°` | `16.31 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_15x_usb` | Tag 67 | ` 24.51 mm` | `18.17°` | `17.64 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 62 | ` 19.03 mm` | `32.11°` | `14.38 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 63 | ` 31.67 mm` | `71.53°` | ` 9.31 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 64 | ` 16.07 mm` | `50.27°` | `13.25 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 65 | ` 14.51 mm` | ` 2.43°` | ` 8.58 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 66 | ` 17.58 mm` | `27.75°` | `14.99 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_15x_usb` | Tag 67 | ` 55.19 mm` | ` 6.54°` | `15.51 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 63 | ` 27.66 mm` | `44.18°` | `15.27 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 65 | ` 46.48 mm` | `33.13°` | ` 9.53 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 66 | ` 25.36 mm` | `33.19°` | ` 8.73 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_15x_usb` | Tag 67 | `  8.44 mm` | ` 5.26°` | ` 4.66 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_15x_usb` | Tag 40 | ` 34.16 mm` | ` 3.61°` | `25.06 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 40 | ` 25.89 mm` | ` 0.87°` | `16.97 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 62 | `112.58 mm` | `28.98°` | `16.00 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 63 | ` 72.33 mm` | ` 5.62°` | `25.08 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 64 | ` 96.72 mm` | `134.31°` | `23.17 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 65 | ` 77.48 mm` | ` 2.31°` | `11.11 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 66 | ` 69.72 mm` | `23.59°` | `16.62 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_15x_usb` | Tag 67 | ` 59.54 mm` | ` 1.71°` | `10.80 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 63 | ` 97.60 mm` | `61.20°` | `38.33 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 65 | ` 92.04 mm` | `19.02°` | `33.35 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 66 | ` 81.31 mm` | `24.37°` | `28.73 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_15x_usb` | Tag 67 | ` 89.86 mm` | ` 4.17°` | `29.06 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 62 | ` 36.57 mm` | `31.67°` | `17.39 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 63 | ` 27.32 mm` | ` 7.43°` | ` 9.77 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 64 | ` 21.27 mm` | `45.27°` | `17.23 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 65 | ` 64.27 mm` | ` 2.58°` | ` 9.73 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 66 | ` 20.48 mm` | `25.13°` | `16.59 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_15x_usb` | Tag 67 | ` 22.76 mm` | ` 3.99°` | `18.55 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_30x_usb` | Tag 66 | ` 48.35 mm` | `25.80°` | `25.15 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_30x_usb` | Tag 67 | ` 57.39 mm` | ` 4.04°` | `25.24 px` |
