# Hand-Eye Calibration Report - Method: `3d_pose_cauchy`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `3d_pose_cauchy`
- **Runtime**: `157.849 s`
- **2D Reprojection Error**: `13.34 px RMS` (Mean: `11.33 px`, Max: `27.22 px`)
- **3D Position Residual**: `35.48 mm RMS` (Mean: `25.58 mm`, Max: `159.89 mm`)
- **3D Rotation Residual**: `33.92° RMS` (Mean: `14.15°`, Max: `134.55°`)
- **Workpiece Box Coplanarity RMSE**: `0.84 mm`
- **Workpiece [box] Coplanarity RMSE**: `0.84 mm` (Tags: `42, 43, 44, 45`)
- **Workpiece [box2] Coplanarity RMSE**: `9.80 mm` (Tags: `50, 51, 52, 53`)
- **Total Tool Tag Sightings**: `184` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.943, -0.383, +0.777] m`
- **USB Position $(X, Y, Z)$**: `[+1.705, -1.002, +0.794] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 6** | `tcp` | `[ +56.7, +126.7,   -6.2]` | `138.9 mm` |
| **Tag 7** | `tcp` | `[ -38.5, +118.8,   -6.5]` | `125.0 mm` |
| **Tag 8** | `tcp` | `[ +62.1,  +59.0,   -8.2]` | `86.1 mm` |
| **Tag 9** | `tcp` | `[ -36.7,  +51.8,   -6.1]` | `63.8 mm` |
| **Tag 10** | `tcp` | `[ +66.2,   -8.6,   -1.7]` | `66.8 mm` |
| **Tag 11** | `tcp` | `[ -35.0,  -17.1,   +4.3]` | `39.2 mm` |
| **Tag 12** | `tcp` | `[ +65.2,  -85.1,   +7.0]` | `107.4 mm` |
| **Tag 13** | `tcp` | `[ -29.0,  -87.6,  +11.0]` | `92.9 mm` |
| **Tag 40** | `tcp` | `[ -60.8,  +17.7, -121.7]` | `137.2 mm` |

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 6 | `  4.38 mm` | ` 1.11°` | ` 2.07 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 7 | `  3.92 mm` | ` 0.13°` | ` 2.23 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 8 | `  7.26 mm` | ` 0.50°` | ` 4.14 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 9 | ` 15.49 mm` | ` 8.35°` | ` 5.54 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 10 | `  7.09 mm` | ` 0.26°` | ` 4.56 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 11 | ` 14.19 mm` | `25.02°` | ` 5.81 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 12 | ` 15.33 mm` | ` 0.26°` | ` 8.33 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 13 | ` 10.60 mm` | ` 0.86°` | ` 4.81 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 6 | `  5.97 mm` | ` 0.25°` | ` 3.76 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 7 | ` 11.14 mm` | `97.32°` | ` 4.14 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 8 | ` 14.38 mm` | ` 0.82°` | ` 6.04 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 9 | ` 12.11 mm` | `88.34°` | ` 7.28 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 10 | ` 12.18 mm` | `98.45°` | ` 5.91 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 11 | ` 15.53 mm` | `75.76°` | ` 8.88 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 12 | ` 32.00 mm` | ` 1.71°` | ` 9.26 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 13 | ` 25.68 mm` | ` 0.64°` | ` 6.51 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 6 | ` 11.03 mm` | ` 0.37°` | ` 5.06 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 7 | ` 23.74 mm` | ` 1.49°` | ` 5.17 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 8 | `  6.29 mm` | ` 0.41°` | ` 4.21 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 9 | ` 20.63 mm` | `94.09°` | ` 7.45 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 10 | ` 16.02 mm` | ` 0.64°` | ` 5.52 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 11 | ` 22.60 mm` | `81.43°` | `11.00 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 12 | ` 18.55 mm` | ` 1.13°` | ` 8.42 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 13 | ` 17.20 mm` | ` 0.38°` | ` 8.84 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 6 | `  7.99 mm` | ` 0.22°` | ` 3.27 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 7 | ` 24.44 mm` | ` 0.70°` | ` 3.23 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 8 | ` 11.38 mm` | ` 0.86°` | ` 7.35 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 9 | ` 12.19 mm` | ` 8.11°` | ` 7.20 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 10 | ` 14.83 mm` | ` 0.80°` | ` 6.62 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 11 | ` 10.85 mm` | `105.23°` | ` 8.57 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 12 | ` 13.45 mm` | `125.25°` | ` 8.35 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 13 | ` 14.56 mm` | ` 0.35°` | ` 3.21 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 6 | `  0.37 mm` | ` 0.37°` | ` 0.49 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 7 | `  0.47 mm` | ` 0.49°` | ` 0.37 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 8 | `  9.11 mm` | ` 0.05°` | ` 4.34 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 9 | ` 10.84 mm` | ` 8.31°` | ` 4.63 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 10 | `  8.40 mm` | ` 0.27°` | ` 4.04 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 11 | ` 12.08 mm` | `25.43°` | ` 6.22 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 12 | ` 11.86 mm` | ` 0.56°` | ` 6.18 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 13 | `  1.96 mm` | ` 0.58°` | ` 0.47 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 40 | `  0.00 mm` | ` 0.00°` | ` 0.61 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 6 | ` 13.02 mm` | `109.72°` | ` 2.92 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 7 | ` 23.41 mm` | ` 0.84°` | ` 2.45 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 8 | ` 14.51 mm` | ` 1.17°` | ` 2.61 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 9 | ` 12.59 mm` | ` 7.16°` | ` 2.89 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 10 | `  8.29 mm` | ` 1.32°` | ` 1.46 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 11 | `  9.58 mm` | `24.64°` | ` 6.57 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 12 | ` 10.79 mm` | ` 0.90°` | ` 7.77 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 13 | ` 17.28 mm` | ` 1.88°` | ` 4.52 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 6 | `  6.56 mm` | `73.04°` | ` 2.93 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 7 | `  5.48 mm` | ` 1.15°` | ` 2.37 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 8 | ` 12.69 mm` | ` 1.35°` | ` 1.06 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 9 | `  2.22 mm` | ` 7.88°` | ` 1.74 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 10 | `  3.64 mm` | ` 2.04°` | ` 0.41 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 11 | `  3.60 mm` | `24.50°` | ` 5.10 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 12 | ` 11.08 mm` | ` 1.73°` | ` 7.43 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 13 | `  7.62 mm` | ` 2.16°` | ` 4.37 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 6 | `  8.26 mm` | ` 2.47°` | ` 4.81 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 7 | `  8.90 mm` | ` 0.93°` | ` 4.42 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 8 | ` 12.10 mm` | ` 1.37°` | ` 4.35 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 9 | `  5.70 mm` | ` 8.85°` | ` 2.84 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 10 | `  9.52 mm` | ` 0.85°` | ` 3.60 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 11 | `  0.32 mm` | `26.05°` | ` 3.54 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 12 | `  6.83 mm` | ` 0.95°` | ` 3.98 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 13 | `  5.04 mm` | ` 2.48°` | ` 2.04 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 6 | `  9.53 mm` | ` 0.58°` | ` 6.05 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 7 | ` 11.96 mm` | ` 0.99°` | ` 5.60 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 8 | ` 14.29 mm` | ` 1.80°` | ` 4.97 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 9 | ` 14.77 mm` | ` 8.91°` | ` 4.74 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 10 | ` 16.49 mm` | ` 2.36°` | ` 5.10 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 11 | ` 18.27 mm` | `26.28°` | ` 5.32 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 12 | ` 13.96 mm` | ` 0.95°` | ` 1.71 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 13 | ` 10.67 mm` | ` 0.95°` | ` 3.39 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 6 | `  1.10 mm` | ` 5.61°` | ` 0.61 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 7 | `  0.31 mm` | ` 2.59°` | ` 0.37 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 8 | `  8.22 mm` | ` 5.15°` | ` 1.59 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 10 | `  8.44 mm` | ` 4.78°` | ` 3.39 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 6 | ` 15.92 mm` | ` 0.72°` | `11.14 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 8 | ` 11.54 mm` | ` 0.56°` | ` 8.72 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 10 | ` 12.41 mm` | ` 0.75°` | ` 9.32 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 12 | `  9.49 mm` | ` 0.15°` | ` 4.73 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 6 | ` 10.82 mm` | ` 1.08°` | ` 7.68 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 7 | ` 12.96 mm` | ` 1.06°` | ` 8.68 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 8 | `  9.10 mm` | ` 0.84°` | ` 6.99 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 9 | ` 14.58 mm` | ` 8.46°` | `10.94 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 10 | ` 16.15 mm` | ` 0.58°` | ` 7.79 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 11 | ` 17.51 mm` | `24.81°` | `13.15 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 12 | ` 15.79 mm` | ` 1.59°` | `10.64 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 13 | ` 15.17 mm` | ` 0.60°` | `10.71 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 6 | ` 26.98 mm` | ` 3.53°` | ` 7.96 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 7 | ` 86.23 mm` | ` 4.74°` | ` 8.33 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 8 | `105.24 mm` | ` 5.19°` | `10.75 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 9 | `159.89 mm` | ` 3.59°` | `11.72 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 10 | `137.55 mm` | ` 9.10°` | `11.02 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 11 | `119.46 mm` | `18.49°` | `13.39 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 12 | ` 54.57 mm` | `123.90°` | `14.61 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 13 | ` 41.96 mm` | `122.81°` | `12.38 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 6 | ` 28.78 mm` | ` 3.27°` | `11.76 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 7 | ` 45.00 mm` | ` 2.75°` | `13.73 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 8 | ` 82.85 mm` | ` 3.86°` | `11.26 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 9 | ` 68.98 mm` | ` 2.59°` | `15.21 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 10 | `101.59 mm` | ` 7.07°` | `12.06 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 11 | ` 78.46 mm` | `19.45°` | `18.88 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 12 | ` 61.05 mm` | `134.55°` | `16.51 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 13 | ` 57.13 mm` | `133.15°` | `17.71 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 6 | ` 21.82 mm` | ` 0.99°` | `10.49 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 7 | ` 35.17 mm` | ` 1.07°` | `10.78 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 8 | ` 27.75 mm` | ` 1.16°` | `14.43 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 9 | ` 23.81 mm` | ` 8.07°` | `14.82 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 10 | ` 93.84 mm` | ` 4.68°` | `13.61 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 11 | ` 20.16 mm` | `24.96°` | `15.20 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 12 | ` 31.85 mm` | ` 2.39°` | `15.68 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 13 | ` 23.42 mm` | ` 1.01°` | `12.02 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 6 | ` 20.36 mm` | ` 0.77°` | `11.44 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 7 | ` 28.48 mm` | ` 0.99°` | `12.18 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 8 | ` 25.72 mm` | ` 1.11°` | `12.47 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 9 | ` 25.95 mm` | ` 8.51°` | `15.58 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 10 | ` 33.54 mm` | ` 1.26°` | `12.41 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 11 | ` 23.20 mm` | `102.95°` | `16.40 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 12 | ` 23.42 mm` | ` 1.91°` | `12.52 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 13 | ` 20.87 mm` | ` 0.54°` | `11.44 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 6 | ` 28.84 mm` | ` 1.06°` | `21.21 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 7 | ` 23.05 mm` | ` 1.08°` | `18.85 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 8 | ` 26.15 mm` | ` 1.31°` | `20.26 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 9 | ` 21.44 mm` | ` 9.09°` | `19.34 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 10 | ` 24.41 mm` | ` 0.82°` | `18.89 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 11 | ` 18.56 mm` | `25.71°` | `18.71 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 12 | ` 14.52 mm` | ` 1.49°` | `10.32 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 13 | ` 21.90 mm` | ` 0.58°` | `13.55 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 40 | ` 24.54 mm` | ` 1.59°` | `19.18 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 6 | ` 22.94 mm` | ` 1.70°` | `17.61 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 7 | ` 20.34 mm` | ` 1.03°` | `16.60 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 8 | ` 19.76 mm` | ` 1.35°` | `16.76 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 9 | ` 20.49 mm` | ` 8.82°` | `16.86 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 10 | ` 18.85 mm` | ` 1.26°` | `16.25 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 11 | ` 21.80 mm` | `25.51°` | `16.25 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 12 | ` 10.92 mm` | ` 1.38°` | ` 9.30 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 13 | ` 16.98 mm` | ` 1.06°` | `13.00 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 6 | ` 28.72 mm` | ` 5.65°` | `21.56 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 7 | ` 31.57 mm` | ` 1.40°` | `21.17 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 8 | ` 31.67 mm` | ` 2.52°` | `21.16 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 9 | ` 35.81 mm` | ` 8.91°` | `21.39 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 10 | ` 32.10 mm` | ` 1.68°` | `20.92 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 11 | ` 40.22 mm` | `26.45°` | `20.75 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 12 | ` 24.42 mm` | ` 1.01°` | `14.79 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 13 | ` 31.46 mm` | ` 2.19°` | `18.73 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 6 | ` 26.75 mm` | ` 1.95°` | `19.66 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 7 | ` 28.33 mm` | ` 1.29°` | `20.07 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 8 | ` 21.89 mm` | ` 0.89°` | `17.65 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 9 | ` 27.47 mm` | ` 7.74°` | `20.70 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 10 | ` 23.75 mm` | ` 1.11°` | `18.28 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 11 | ` 37.91 mm` | `25.79°` | `22.82 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 12 | ` 23.32 mm` | ` 0.92°` | `17.16 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 13 | ` 30.41 mm` | ` 0.78°` | `20.85 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 6 | ` 24.70 mm` | ` 2.76°` | `13.90 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 7 | ` 28.01 mm` | ` 1.52°` | `15.35 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 8 | ` 25.64 mm` | ` 2.62°` | `14.77 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 10 | ` 27.52 mm` | ` 3.34°` | `16.99 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 12 | ` 37.33 mm` | ` 1.58°` | `21.58 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_12x_usb` | Tag 10 | ` 34.10 mm` | ` 1.50°` | `22.11 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_12x_usb` | Tag 40 | ` 24.04 mm` | ` 3.55°` | `21.02 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 6 | ` 31.27 mm` | ` 0.72°` | `13.76 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 7 | `133.31 mm` | ` 0.79°` | `13.59 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 8 | ` 20.38 mm` | ` 1.20°` | `19.09 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 10 | ` 42.05 mm` | ` 1.49°` | `17.78 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 12 | ` 21.37 mm` | ` 2.06°` | `16.48 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 40 | ` 22.39 mm` | ` 1.31°` | `16.64 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 6 | ` 45.52 mm` | ` 3.50°` | `22.39 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 7 | ` 38.44 mm` | ` 2.04°` | `21.13 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 8 | ` 50.57 mm` | ` 3.14°` | `23.23 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 10 | ` 53.02 mm` | ` 2.32°` | `22.16 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 12 | ` 62.88 mm` | `67.07°` | `25.54 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 6 | ` 29.08 mm` | ` 0.52°` | `25.52 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 7 | ` 30.11 mm` | `119.90°` | `25.22 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 8 | ` 26.44 mm` | ` 0.70°` | `22.13 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 9 | ` 30.71 mm` | ` 8.28°` | `24.90 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 10 | ` 26.98 mm` | ` 0.50°` | `22.12 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 11 | ` 33.33 mm` | `96.36°` | `27.22 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 12 | ` 27.27 mm` | ` 1.14°` | `20.41 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 13 | ` 33.33 mm` | ` 0.68°` | `24.38 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 6 | ` 38.26 mm` | ` 3.02°` | `18.88 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 7 | ` 35.67 mm` | ` 2.63°` | `18.09 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 8 | ` 40.00 mm` | ` 3.03°` | `19.85 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 10 | ` 50.61 mm` | `64.55°` | `20.34 px` |
