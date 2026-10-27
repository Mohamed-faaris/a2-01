# Hand-Eye Calibration Report - Method: `rigid_cube_ba`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `rigid_cube_ba`
- **Runtime**: `28.950 s`
- **2D Reprojection Error**: `5.93 px RMS` (Mean: `4.50 px`, Max: `14.13 px`)
- **3D Position Residual**: `51.28 mm RMS` (Mean: `43.46 mm`, Max: `184.10 mm`)
- **3D Rotation Residual**: `35.52° RMS` (Mean: `13.14°`, Max: `134.58°`)
- **Workpiece Box Coplanarity RMSE**: `0.84 mm`
- **Workpiece [box] Coplanarity RMSE**: `0.84 mm` (Tags: `42, 43, 44, 45`)
- **Workpiece [box2] Coplanarity RMSE**: `9.80 mm` (Tags: `50, 51, 52, 53`)
- **Total Tool Tag Sightings**: `184` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.904, -0.357, +0.784] m`
- **USB Position $(X, Y, Z)$**: `[+1.675, -0.979, +0.794] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 6** | `tcp` | `[ +36.2, +108.5,  -24.6]` | `117.0 mm` |
| **Tag 7** | `tcp` | `[ -56.6, +102.5,  -24.2]` | `119.6 mm` |
| **Tag 8** | `tcp` | `[ +40.7,  +42.1,  -20.8]` | `62.1 mm` |
| **Tag 9** | `tcp` | `[ -51.9,  +35.9,  -20.5]` | `66.3 mm` |
| **Tag 10** | `tcp` | `[ +45.6,  -24.5,  -17.1]` | `54.5 mm` |
| **Tag 11** | `tcp` | `[ -47.3,  -30.6,  -17.3]` | `58.9 mm` |
| **Tag 12** | `tcp` | `[ +49.9,  -91.1,  -12.5]` | `104.6 mm` |
| **Tag 13** | `tcp` | `[ -43.0,  -96.4,  -13.6]` | `106.4 mm` |
| **Tag 40** | `tcp` | `[ -95.3,   +4.4, -130.7]` | `161.8 mm` |

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 6 | ` 33.29 mm` | ` 3.15°` | ` 0.57 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 7 | ` 32.90 mm` | ` 0.95°` | ` 0.44 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 8 | ` 28.45 mm` | ` 0.67°` | ` 0.69 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 9 | ` 17.57 mm` | ` 0.45°` | ` 0.33 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 10 | ` 27.69 mm` | ` 0.59°` | ` 0.77 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 11 | ` 27.81 mm` | ` 1.46°` | ` 0.60 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 12 | ` 28.08 mm` | ` 0.75°` | ` 0.73 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 13 | ` 48.15 mm` | ` 2.17°` | ` 1.11 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 6 | ` 27.01 mm` | ` 1.92°` | ` 0.72 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 7 | ` 20.71 mm` | `97.46°` | ` 0.63 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 8 | ` 35.64 mm` | ` 0.65°` | ` 0.68 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 9 | ` 29.29 mm` | `96.43°` | ` 0.61 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 10 | ` 35.05 mm` | `98.70°` | ` 0.79 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 11 | ` 26.81 mm` | `100.68°` | ` 0.69 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 12 | ` 66.98 mm` | ` 1.74°` | ` 1.09 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 13 | ` 63.88 mm` | ` 1.60°` | ` 0.87 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 6 | ` 23.42 mm` | ` 1.39°` | ` 0.70 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 7 | ` 10.44 mm` | ` 1.89°` | ` 0.82 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 8 | ` 28.93 mm` | ` 0.44°` | ` 0.82 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 9 | ` 14.24 mm` | `101.97°` | ` 0.69 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 10 | ` 17.87 mm` | ` 0.59°` | ` 0.50 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 11 | ` 22.04 mm` | `105.92°` | ` 0.80 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 12 | ` 55.28 mm` | ` 1.10°` | ` 0.87 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 13 | ` 53.91 mm` | ` 1.02°` | ` 0.95 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 6 | ` 27.06 mm` | ` 2.30°` | ` 1.01 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 7 | ` 46.28 mm` | ` 1.27°` | ` 0.75 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 8 | ` 24.36 mm` | ` 0.95°` | ` 0.59 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 9 | ` 30.69 mm` | ` 1.03°` | ` 0.52 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 10 | ` 32.88 mm` | ` 1.49°` | ` 0.66 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 11 | ` 38.05 mm` | `129.84°` | ` 1.03 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 12 | ` 39.83 mm` | `125.91°` | ` 0.66 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 13 | ` 47.53 mm` | ` 1.41°` | ` 0.67 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 6 | ` 21.42 mm` | ` 2.50°` | ` 1.47 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 7 | ` 23.05 mm` | ` 0.55°` | ` 0.89 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 8 | ` 10.26 mm` | ` 0.47°` | ` 0.50 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 9 | ` 15.18 mm` | ` 0.63°` | ` 0.52 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 10 | ` 13.52 mm` | ` 1.14°` | ` 0.42 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 11 | ` 19.38 mm` | ` 0.86°` | ` 0.40 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 12 | ` 20.07 mm` | ` 0.92°` | ` 0.43 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 13 | ` 34.14 mm` | ` 1.15°` | ` 0.76 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 40 | `  5.76 mm` | ` 2.85°` | ` 0.69 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 6 | ` 40.82 mm` | `110.30°` | ` 1.59 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 7 | ` 52.23 mm` | ` 1.74°` | ` 0.93 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 8 | ` 38.25 mm` | ` 1.67°` | ` 1.48 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 9 | ` 41.75 mm` | ` 2.03°` | ` 1.05 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 10 | ` 33.88 mm` | ` 2.29°` | ` 1.69 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 11 | ` 43.90 mm` | ` 2.08°` | ` 1.05 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 12 | ` 31.82 mm` | ` 1.88°` | ` 2.00 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 13 | ` 49.50 mm` | ` 2.72°` | ` 1.09 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 6 | ` 42.47 mm` | `73.48°` | ` 0.98 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 7 | ` 41.63 mm` | ` 1.43°` | ` 0.71 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 8 | ` 44.69 mm` | ` 1.61°` | ` 0.85 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 9 | ` 38.11 mm` | ` 1.02°` | ` 0.63 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 10 | ` 38.08 mm` | ` 2.80°` | ` 0.80 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 11 | ` 43.25 mm` | ` 1.83°` | ` 0.72 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 12 | ` 37.58 mm` | ` 2.50°` | ` 1.13 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 13 | ` 46.98 mm` | ` 2.72°` | ` 0.88 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 6 | ` 46.39 mm` | ` 4.07°` | ` 1.13 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 7 | ` 54.56 mm` | ` 1.04°` | ` 1.28 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 8 | ` 54.02 mm` | ` 1.93°` | ` 0.91 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 9 | ` 51.08 mm` | ` 1.26°` | ` 0.97 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 10 | ` 54.45 mm` | ` 1.59°` | ` 0.90 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 11 | ` 54.41 mm` | ` 2.34°` | ` 0.83 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 12 | ` 46.62 mm` | ` 0.96°` | ` 0.76 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 13 | ` 51.60 mm` | ` 3.06°` | ` 1.00 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 6 | ` 37.65 mm` | ` 1.84°` | ` 1.25 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 7 | ` 32.94 mm` | ` 0.62°` | ` 0.87 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 8 | ` 23.74 mm` | ` 1.87°` | ` 1.11 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 9 | ` 27.51 mm` | ` 0.77°` | ` 0.69 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 10 | ` 24.56 mm` | ` 1.63°` | ` 0.97 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 11 | ` 31.84 mm` | ` 1.61°` | ` 0.67 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 12 | ` 32.92 mm` | ` 0.45°` | ` 0.81 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 13 | ` 41.91 mm` | ` 0.90°` | ` 0.76 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 6 | ` 53.67 mm` | ` 6.74°` | ` 4.95 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 7 | ` 52.19 mm` | ` 3.37°` | ` 4.33 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 8 | ` 39.49 mm` | ` 5.59°` | ` 5.54 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 10 | ` 43.79 mm` | ` 4.86°` | ` 6.73 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 6 | ` 25.75 mm` | ` 1.49°` | ` 0.86 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 8 | ` 27.12 mm` | ` 0.47°` | ` 1.03 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 10 | ` 30.94 mm` | ` 0.43°` | ` 0.95 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 12 | ` 33.60 mm` | ` 0.38°` | ` 0.89 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 6 | ` 32.63 mm` | ` 2.53°` | ` 3.99 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 7 | ` 35.08 mm` | ` 1.71°` | ` 3.07 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 8 | ` 23.21 mm` | ` 0.90°` | ` 4.51 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 9 | ` 29.18 mm` | ` 2.10°` | ` 3.80 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 10 | ` 37.89 mm` | ` 1.21°` | ` 5.35 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 11 | ` 38.46 mm` | ` 2.35°` | ` 4.46 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 12 | ` 39.79 mm` | ` 1.87°` | ` 5.80 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 13 | ` 38.24 mm` | ` 1.78°` | ` 4.83 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 6 | ` 51.38 mm` | ` 5.07°` | ` 8.37 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 7 | `112.45 mm` | ` 4.73°` | ` 6.93 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 8 | `128.16 mm` | ` 5.02°` | ` 8.14 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 9 | `184.10 mm` | ` 9.52°` | ` 7.83 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 10 | `162.90 mm` | ` 9.37°` | ` 9.10 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 11 | `148.68 mm` | ` 8.23°` | ` 7.83 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 12 | ` 85.32 mm` | `124.12°` | ` 9.04 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 13 | ` 73.39 mm` | `122.69°` | ` 8.36 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 6 | ` 50.63 mm` | ` 4.70°` | ` 6.65 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 7 | ` 68.33 mm` | ` 2.68°` | ` 5.81 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 8 | `106.33 mm` | ` 3.51°` | ` 6.62 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 9 | ` 91.45 mm` | ` 5.61°` | ` 6.84 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 10 | `127.08 mm` | ` 7.20°` | ` 7.56 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 11 | `104.56 mm` | ` 6.10°` | ` 7.14 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 12 | ` 92.18 mm` | `134.58°` | ` 7.95 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 13 | ` 86.52 mm` | `132.84°` | ` 7.43 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 6 | ` 37.44 mm` | ` 2.82°` | `10.07 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 7 | ` 53.18 mm` | ` 1.75°` | ` 8.49 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 8 | ` 38.52 mm` | ` 1.34°` | ` 9.28 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 9 | ` 34.92 mm` | ` 1.78°` | ` 8.75 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 10 | `111.13 mm` | ` 4.95°` | ` 9.50 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 11 | ` 34.64 mm` | ` 1.93°` | ` 8.90 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 12 | ` 54.57 mm` | ` 2.30°` | ` 9.87 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 13 | ` 47.38 mm` | ` 1.95°` | ` 8.94 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 6 | ` 33.98 mm` | ` 2.32°` | ` 5.62 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 7 | ` 44.59 mm` | ` 1.70°` | ` 4.93 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 8 | ` 36.08 mm` | ` 1.31°` | ` 6.06 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 9 | ` 38.97 mm` | ` 2.18°` | ` 5.72 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 10 | ` 47.18 mm` | ` 2.08°` | ` 7.03 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 11 | ` 42.35 mm` | `127.66°` | ` 6.33 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 12 | ` 46.26 mm` | ` 2.11°` | ` 7.63 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 13 | ` 44.99 mm` | ` 1.65°` | ` 6.72 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 6 | ` 42.08 mm` | ` 2.66°` | ` 5.52 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 7 | ` 34.64 mm` | ` 1.17°` | ` 4.65 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 8 | ` 34.95 mm` | ` 1.35°` | ` 4.64 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 9 | ` 30.12 mm` | ` 1.63°` | ` 4.14 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 10 | ` 35.65 mm` | ` 1.40°` | ` 3.76 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 11 | ` 30.25 mm` | ` 1.82°` | ` 3.62 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 12 | ` 37.77 mm` | ` 1.44°` | ` 3.33 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 13 | ` 47.22 mm` | ` 1.54°` | ` 3.65 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 40 | ` 19.29 mm` | ` 3.19°` | ` 6.28 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 6 | ` 43.34 mm` | ` 3.63°` | ` 3.12 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 7 | ` 35.66 mm` | ` 1.85°` | ` 2.98 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 8 | ` 31.64 mm` | ` 1.61°` | ` 2.47 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 9 | ` 29.31 mm` | ` 2.12°` | ` 2.51 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 10 | ` 31.96 mm` | ` 1.56°` | ` 1.89 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 11 | ` 29.39 mm` | ` 2.17°` | ` 2.31 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 12 | ` 35.70 mm` | ` 1.67°` | ` 1.78 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 13 | ` 47.53 mm` | ` 2.26°` | ` 2.56 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 6 | ` 38.21 mm` | ` 6.85°` | ` 4.85 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 7 | ` 29.94 mm` | ` 2.23°` | ` 5.11 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 8 | ` 22.87 mm` | ` 3.00°` | ` 4.58 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 9 | ` 20.33 mm` | ` 1.07°` | ` 4.91 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 10 | ` 24.11 mm` | ` 1.02°` | ` 4.28 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 11 | ` 20.72 mm` | ` 1.48°` | ` 4.77 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 12 | ` 32.40 mm` | ` 1.30°` | ` 4.13 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 13 | ` 33.03 mm` | ` 3.38°` | ` 5.09 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 6 | ` 23.04 mm` | ` 3.07°` | ` 4.51 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 7 | ` 22.63 mm` | ` 2.09°` | ` 5.03 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 8 | ` 28.95 mm` | ` 0.99°` | ` 4.99 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 9 | ` 26.22 mm` | ` 1.13°` | ` 5.51 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 10 | ` 28.15 mm` | ` 1.39°` | ` 5.59 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 11 | ` 18.54 mm` | ` 2.05°` | ` 6.13 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 12 | ` 37.09 mm` | ` 1.46°` | ` 6.27 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 13 | ` 33.60 mm` | ` 1.92°` | ` 6.49 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 6 | ` 35.38 mm` | ` 2.35°` | ` 9.34 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 7 | ` 31.96 mm` | ` 1.44°` | ` 8.20 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 8 | ` 31.52 mm` | ` 2.08°` | `10.14 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 10 | ` 53.23 mm` | ` 3.05°` | `11.10 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 12 | ` 38.63 mm` | ` 2.12°` | `12.09 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_12x_usb` | Tag 10 | ` 40.44 mm` | ` 1.22°` | `10.70 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_12x_usb` | Tag 40 | ` 13.45 mm` | ` 1.21°` | ` 8.98 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 6 | ` 41.52 mm` | ` 1.81°` | `12.86 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 7 | `147.20 mm` | ` 1.59°` | `10.94 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 8 | ` 19.48 mm` | ` 1.51°` | `12.00 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 10 | ` 51.45 mm` | ` 2.21°` | `12.50 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 12 | ` 19.24 mm` | ` 2.51°` | `12.58 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 40 | ` 18.81 mm` | ` 3.37°` | ` 9.12 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 6 | ` 72.77 mm` | ` 3.62°` | `14.13 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 7 | ` 49.43 mm` | ` 1.80°` | `12.57 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 8 | ` 73.98 mm` | ` 3.41°` | `13.09 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 10 | ` 81.93 mm` | ` 2.85°` | `12.61 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 12 | ` 93.87 mm` | `67.40°` | `12.19 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 6 | ` 27.36 mm` | ` 2.20°` | ` 8.44 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 7 | ` 32.38 mm` | `120.30°` | ` 8.55 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 8 | ` 27.49 mm` | ` 0.89°` | ` 8.14 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 9 | ` 22.41 mm` | ` 1.76°` | ` 8.12 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 10 | ` 27.64 mm` | ` 1.05°` | ` 7.60 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 11 | ` 35.60 mm` | `121.60°` | ` 7.49 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 12 | ` 44.21 mm` | ` 1.42°` | ` 7.60 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 13 | ` 46.18 mm` | ` 1.82°` | ` 7.62 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 6 | ` 68.98 mm` | ` 4.55°` | `13.00 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 7 | ` 64.39 mm` | ` 1.99°` | `11.50 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 8 | ` 63.43 mm` | ` 2.92°` | `12.53 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 10 | ` 29.98 mm` | `64.06°` | `12.82 px` |
