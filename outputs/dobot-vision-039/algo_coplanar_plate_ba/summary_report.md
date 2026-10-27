# Hand-Eye Calibration Report - Method: `coplanar_plate_ba`

## 1. Executive Summary & Verification Metrics
- **Methodology**: `coplanar_plate_ba`
- **Runtime**: `62.823 s`
- **2D Reprojection Error**: `1.33 px RMS` (Mean: `1.15 px`, Max: `4.83 px`)
- **3D Position Residual**: `42.87 mm RMS` (Mean: `34.22 mm`, Max: `165.80 mm`)
- **3D Rotation Residual**: `35.37° RMS` (Mean: `13.46°`, Max: `134.15°`)
- **Workpiece Box Coplanarity RMSE**: `0.84 mm`
- **Workpiece [box] Coplanarity RMSE**: `0.84 mm` (Tags: `42, 43, 44, 45`)
- **Workpiece [box2] Coplanarity RMSE**: `9.80 mm` (Tags: `50, 51, 52, 53`)
- **Total Tool Tag Sightings**: `184` across 18 arm poses

---

## 2. Solved Camera Extrinsics in `base_link` Frame
- **D415 Position $(X, Y, Z)$**: `[+1.902, -0.364, +0.784] m`
- **USB Position $(X, Y, Z)$**: `[+1.674, -0.997, +0.813] m`

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 6** | `tcp` | `[ +33.5, +111.2,  -26.6]` | `119.2 mm` |
| **Tag 7** | `tcp` | `[ -58.8, +105.1,  -26.6]` | `123.4 mm` |
| **Tag 8** | `tcp` | `[ +38.4,  +44.6,  -22.7]` | `63.1 mm` |
| **Tag 9** | `tcp` | `[ -54.0,  +38.5,  -22.7]` | `70.1 mm` |
| **Tag 10** | `tcp` | `[ +43.6,  -21.7,  -18.7]` | `52.2 mm` |
| **Tag 11** | `tcp` | `[ -49.2,  -28.1,  -18.7]` | `59.7 mm` |
| **Tag 12** | `tcp` | `[ +48.5,  -88.2,  -14.7]` | `101.8 mm` |
| **Tag 13** | `tcp` | `[ -45.1,  -94.6,  -14.7]` | `105.9 mm` |
| **Tag 40** | `tcp` | `[ -90.2,   +7.4, -135.9]` | `163.2 mm` |

---

## 4. Parameter Observability & Covariance Uncertainty
- **Jacobian Condition Number $\kappa$**: `412617048.94`
- **Minimum Singular Value $\sigma_{min}$**: `0.0002`
- **Top Singular Values**: `[74272.506, 64841.822, 49924.707, 40494.288, 20636.539, 17149.313, 15819.501, 15137.726]`
- **D415 Estimated Position 1-$\sigma$ Uncertainty $(X, Y, Z)$**: `±[0.7, 1.13, 0.89] mm`
- **D415 Estimated Rotation 1-$\sigma$ Uncertainty $(Roll, Pitch, Yaw)$**: `±[0.03, 0.04, 0.04]°`
- **USB Estimated Position 1-$\sigma$ Uncertainty $(X, Y, Z)$**: `±[0.73, 0.62, 0.71] mm`
- **USB Estimated Rotation 1-$\sigma$ Uncertainty $(Roll, Pitch, Yaw)$**: `±[0.02, 0.03, 0.02]°`

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 6 | ` 34.86 mm` | ` 5.39°` | ` 0.71 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 7 | ` 34.92 mm` | ` 2.59°` | ` 0.94 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 8 | ` 29.83 mm` | ` 2.43°` | ` 0.59 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 9 | ` 19.18 mm` | ` 2.40°` | ` 0.72 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 10 | ` 28.75 mm` | ` 1.27°` | ` 0.73 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 11 | ` 28.53 mm` | ` 0.28°` | ` 0.74 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 12 | ` 29.73 mm` | ` 0.46°` | ` 0.66 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x` | Tag 13 | ` 48.29 mm` | ` 1.73°` | ` 0.82 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 6 | ` 28.21 mm` | ` 4.07°` | ` 0.89 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 7 | ` 22.35 mm` | `95.61°` | ` 0.72 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 8 | ` 36.47 mm` | ` 1.87°` | ` 0.78 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 9 | ` 30.56 mm` | `96.24°` | ` 0.60 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 10 | ` 35.76 mm` | `98.13°` | ` 0.93 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 11 | ` 27.28 mm` | `99.94°` | ` 0.57 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 12 | ` 68.11 mm` | ` 1.88°` | ` 1.30 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x` | Tag 13 | ` 63.57 mm` | ` 1.19°` | ` 0.83 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 6 | ` 25.00 mm` | ` 3.58°` | ` 0.92 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 7 | ` 12.40 mm` | ` 3.58°` | ` 1.09 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 8 | ` 30.11 mm` | ` 1.80°` | ` 0.76 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 9 | ` 15.85 mm` | `101.92°` | ` 0.89 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 10 | ` 18.93 mm` | ` 1.17°` | ` 0.53 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 11 | ` 22.83 mm` | `105.28°` | ` 0.78 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 12 | ` 56.71 mm` | ` 1.53°` | ` 1.00 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x` | Tag 13 | ` 53.89 mm` | ` 1.01°` | ` 0.89 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 6 | ` 26.76 mm` | ` 4.18°` | ` 1.65 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 7 | ` 46.53 mm` | ` 2.31°` | ` 1.43 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 8 | ` 23.89 mm` | ` 2.80°` | ` 1.04 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 9 | ` 30.64 mm` | ` 1.92°` | ` 0.95 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 10 | ` 32.40 mm` | ` 1.76°` | ` 0.79 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 11 | ` 37.42 mm` | `128.41°` | ` 1.24 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 12 | ` 39.86 mm` | `125.37°` | ` 0.95 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x` | Tag 13 | ` 46.11 mm` | ` 1.05°` | ` 0.71 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 6 | ` 21.46 mm` | ` 4.59°` | ` 2.27 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 7 | ` 23.57 mm` | ` 3.16°` | ` 1.70 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 8 | ` 10.30 mm` | ` 2.18°` | ` 1.33 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 9 | ` 15.36 mm` | ` 2.33°` | ` 1.07 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 10 | ` 13.35 mm` | ` 1.32°` | ` 0.63 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 11 | ` 18.97 mm` | ` 0.91°` | ` 0.70 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 12 | ` 20.46 mm` | ` 0.90°` | ` 0.59 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 13 | ` 33.15 mm` | ` 0.57°` | ` 0.66 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x` | Tag 40 | ` 13.43 mm` | ` 0.41°` | ` 0.88 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 6 | ` 41.60 mm` | `112.25°` | ` 2.50 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 7 | ` 53.27 mm` | ` 2.57°` | ` 1.68 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 8 | ` 39.18 mm` | ` 2.93°` | ` 2.42 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 9 | ` 42.53 mm` | ` 3.25°` | ` 1.79 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 10 | ` 34.53 mm` | ` 2.32°` | ` 2.40 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 11 | ` 44.23 mm` | ` 1.02°` | ` 1.73 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 12 | ` 33.00 mm` | ` 1.53°` | ` 2.56 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x` | Tag 13 | ` 49.74 mm` | ` 2.14°` | ` 2.28 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 6 | ` 44.11 mm` | `75.13°` | ` 1.07 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 7 | ` 43.56 mm` | ` 2.89°` | ` 1.33 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 8 | ` 46.33 mm` | ` 3.11°` | ` 0.95 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 9 | ` 39.71 mm` | ` 2.17°` | ` 1.01 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 10 | ` 39.33 mm` | ` 2.84°` | ` 0.84 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 11 | ` 44.16 mm` | ` 0.66°` | ` 0.95 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 12 | ` 39.40 mm` | ` 2.17°` | ` 1.03 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x` | Tag 13 | ` 47.75 mm` | ` 2.15°` | ` 1.35 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 6 | ` 48.75 mm` | ` 5.83°` | ` 1.97 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 7 | ` 57.21 mm` | ` 2.63°` | ` 2.31 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 8 | ` 56.23 mm` | ` 3.68°` | ` 1.40 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 9 | ` 53.33 mm` | ` 2.22°` | ` 1.77 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 10 | ` 56.18 mm` | ` 2.09°` | ` 1.29 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 11 | ` 55.81 mm` | ` 1.78°` | ` 1.16 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 12 | ` 48.87 mm` | ` 1.51°` | ` 1.14 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x` | Tag 13 | ` 52.79 mm` | ` 2.50°` | ` 0.94 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 6 | ` 39.18 mm` | ` 4.14°` | ` 0.99 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 7 | ` 34.93 mm` | ` 3.60°` | ` 1.22 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 8 | ` 25.08 mm` | ` 2.17°` | ` 0.99 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 9 | ` 29.07 mm` | ` 3.05°` | ` 1.01 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 10 | ` 25.55 mm` | ` 1.79°` | ` 1.05 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 11 | ` 32.51 mm` | ` 1.73°` | ` 0.80 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 12 | ` 34.50 mm` | ` 0.94°` | ` 0.91 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x` | Tag 13 | ` 42.04 mm` | ` 0.91°` | ` 0.82 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 6 | ` 53.21 mm` | ` 8.60°` | ` 3.21 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 7 | ` 52.35 mm` | ` 4.09°` | ` 2.77 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 8 | ` 38.74 mm` | ` 6.06°` | ` 3.93 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x` | Tag 10 | ` 42.60 mm` | ` 5.52°` | ` 4.83 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 6 | ` 27.14 mm` | ` 3.55°` | ` 0.85 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 8 | ` 28.39 mm` | ` 1.58°` | ` 0.66 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 10 | ` 32.03 mm` | ` 0.52°` | ` 0.88 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x` | Tag 12 | ` 35.29 mm` | ` 0.43°` | ` 0.79 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 6 | ` 14.15 mm` | ` 4.28°` | ` 0.88 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 7 | ` 17.78 mm` | ` 2.70°` | ` 0.92 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 8 | `  3.97 mm` | ` 1.65°` | ` 0.48 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 9 | ` 11.11 mm` | ` 1.53°` | ` 0.50 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 10 | ` 18.37 mm` | ` 1.06°` | ` 0.44 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 11 | ` 19.36 mm` | ` 0.12°` | ` 0.33 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 12 | ` 20.58 mm` | ` 1.29°` | ` 0.48 px` |
| `pose01_pose_20261027_204520_255_l01_shot01_fused_24x_usb` | Tag 13 | ` 18.17 mm` | ` 0.79°` | ` 0.46 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 6 | ` 31.49 mm` | ` 5.77°` | ` 1.38 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 7 | ` 94.21 mm` | ` 5.53°` | ` 1.58 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 8 | `108.84 mm` | ` 5.25°` | ` 1.41 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 9 | `165.80 mm` | `10.88°` | ` 2.46 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 10 | `143.65 mm` | ` 8.70°` | ` 2.02 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 11 | `129.75 mm` | ` 8.33°` | ` 1.92 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 12 | ` 65.78 mm` | `123.69°` | ` 1.12 px` |
| `pose02_pose_20261027_204527_840_l01_shot01_fused_12x_usb` | Tag 13 | ` 53.02 mm` | `122.52°` | ` 1.07 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 6 | ` 32.87 mm` | ` 5.49°` | ` 1.78 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 7 | ` 51.71 mm` | ` 3.70°` | ` 1.43 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 8 | ` 88.67 mm` | ` 3.87°` | ` 1.69 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 9 | ` 74.63 mm` | ` 6.49°` | ` 1.77 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 10 | `109.46 mm` | ` 6.59°` | ` 1.89 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 11 | ` 87.28 mm` | ` 5.71°` | ` 1.58 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 12 | ` 74.49 mm` | `134.15°` | ` 0.87 px` |
| `pose03_pose_20261027_204538_615_l01_shot01_fused_12x_usb` | Tag 13 | ` 68.29 mm` | `132.66°` | ` 1.29 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 6 | ` 16.73 mm` | ` 3.86°` | ` 1.38 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 7 | ` 34.19 mm` | ` 2.61°` | ` 1.09 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 8 | ` 17.89 mm` | ` 1.99°` | ` 0.96 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 9 | ` 15.01 mm` | ` 2.08°` | ` 0.67 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 10 | ` 91.79 mm` | ` 4.48°` | ` 2.37 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 11 | ` 14.10 mm` | ` 1.24°` | ` 0.72 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 12 | ` 34.73 mm` | ` 1.83°` | ` 0.95 px` |
| `pose04_pose_20261027_204542_935_l01_shot01_fused_12x_usb` | Tag 13 | ` 26.55 mm` | ` 0.80°` | ` 0.72 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 6 | ` 14.61 mm` | ` 3.95°` | ` 1.49 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 7 | ` 26.56 mm` | ` 2.50°` | ` 1.29 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 8 | ` 16.34 mm` | ` 2.48°` | ` 0.87 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 9 | ` 20.26 mm` | ` 1.55°` | ` 0.74 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 10 | ` 27.31 mm` | ` 1.79°` | ` 0.71 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 11 | ` 22.75 mm` | `126.11°` | ` 0.61 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 12 | ` 26.67 mm` | ` 1.47°` | ` 0.60 px` |
| `pose05_pose_20261027_204546_326_l01_shot01_fused_12x_usb` | Tag 13 | ` 24.55 mm` | ` 0.60°` | ` 0.44 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 6 | ` 25.77 mm` | ` 4.69°` | ` 1.67 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 7 | ` 19.19 mm` | ` 2.95°` | ` 1.52 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 8 | ` 18.08 mm` | ` 2.56°` | ` 0.91 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 9 | ` 13.81 mm` | ` 1.82°` | ` 0.93 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 10 | ` 18.15 mm` | ` 1.56°` | ` 0.61 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 11 | ` 12.88 mm` | ` 0.33°` | ` 0.44 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 12 | ` 20.41 mm` | ` 1.07°` | ` 0.64 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 13 | ` 28.94 mm` | ` 0.79°` | ` 0.64 px` |
| `pose06_pose_20261027_204550_465_l01_shot01_fused_12x_usb` | Tag 40 | ` 10.45 mm` | ` 1.54°` | ` 0.78 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 6 | ` 28.36 mm` | ` 5.53°` | ` 1.12 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 7 | ` 21.62 mm` | ` 2.31°` | ` 1.01 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 8 | ` 16.01 mm` | ` 2.60°` | ` 0.50 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 9 | ` 14.38 mm` | ` 1.34°` | ` 0.66 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 10 | ` 15.58 mm` | ` 1.72°` | ` 0.49 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 11 | ` 13.17 mm` | ` 0.40°` | ` 0.36 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 12 | ` 19.50 mm` | ` 1.18°` | ` 0.48 px` |
| `pose07_pose_20261027_204555_040_l01_shot01_fused_12x_usb` | Tag 13 | ` 30.38 mm` | ` 1.42°` | ` 0.60 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 6 | ` 26.17 mm` | ` 8.41°` | ` 0.87 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 7 | ` 18.78 mm` | ` 3.20°` | ` 0.78 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 8 | `  9.84 mm` | ` 3.23°` | ` 0.58 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 9 | `  7.97 mm` | ` 3.21°` | ` 0.58 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 10 | ` 10.34 mm` | ` 0.99°` | ` 0.65 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 11 | `  7.00 mm` | ` 2.20°` | ` 0.39 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 12 | ` 18.90 mm` | ` 0.25°` | ` 0.58 px` |
| `pose08_pose_20261027_204602_450_l01_shot01_fused_12x_usb` | Tag 13 | ` 18.62 mm` | ` 2.73°` | ` 0.46 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 6 | `  7.56 mm` | ` 4.62°` | ` 1.01 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 7 | `  8.26 mm` | ` 3.01°` | ` 0.84 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 8 | ` 12.95 mm` | ` 1.55°` | ` 0.64 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 9 | ` 11.17 mm` | ` 2.76°` | ` 0.77 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 10 | ` 11.57 mm` | ` 1.14°` | ` 0.83 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 11 | `  1.69 mm` | ` 1.27°` | ` 0.51 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 12 | ` 20.90 mm` | ` 0.56°` | ` 0.82 px` |
| `pose09_pose_20261027_204606_814_l01_shot01_fused_12x_usb` | Tag 13 | ` 16.44 mm` | ` 0.86°` | ` 0.46 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 6 | ` 16.99 mm` | ` 2.47°` | ` 2.05 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 7 | ` 15.07 mm` | ` 3.43°` | ` 1.62 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 8 | ` 11.90 mm` | ` 0.73°` | ` 1.54 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 10 | ` 34.43 mm` | ` 2.08°` | ` 1.23 px` |
| `pose10_pose_20261027_204611_178_l01_shot01_fused_12x_usb` | Tag 12 | ` 18.65 mm` | ` 1.57°` | ` 1.31 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_12x_usb` | Tag 10 | ` 21.85 mm` | ` 1.11°` | ` 1.63 px` |
| `pose11_pose_20261027_204620_151_l01_shot01_fused_12x_usb` | Tag 40 | `  1.82 mm` | ` 2.85°` | ` 0.67 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 6 | ` 21.60 mm` | ` 3.33°` | ` 2.47 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 7 | `129.57 mm` | ` 3.43°` | ` 3.50 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 8 | `  3.10 mm` | ` 2.37°` | ` 1.56 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 10 | ` 31.90 mm` | ` 1.87°` | ` 1.63 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 12 | `  4.23 mm` | ` 1.91°` | ` 1.71 px` |
| `pose12_pose_20261027_204624_005_l01_shot01_fused_12x_usb` | Tag 40 | `  5.54 mm` | ` 1.63°` | ` 1.18 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 6 | ` 54.48 mm` | ` 2.72°` | ` 0.91 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 7 | ` 30.51 mm` | ` 4.40°` | ` 0.86 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 8 | ` 56.49 mm` | ` 2.92°` | ` 0.74 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 10 | ` 65.35 mm` | ` 1.49°` | ` 1.04 px` |
| `pose13_pose_20261027_204630_185_l01_shot01_fused_12x_usb` | Tag 12 | ` 78.81 mm` | `66.84°` | ` 2.23 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 6 | ` 12.34 mm` | ` 3.70°` | ` 1.60 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 7 | ` 18.77 mm` | `117.34°` | ` 1.31 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 8 | ` 11.97 mm` | ` 2.20°` | ` 1.17 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 9 | `  7.55 mm` | ` 1.93°` | ` 0.90 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 10 | ` 11.95 mm` | ` 0.97°` | ` 0.95 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 11 | ` 20.74 mm` | `120.25°` | ` 0.92 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 12 | ` 29.28 mm` | ` 0.87°` | ` 0.75 px` |
| `pose14_pose_20261027_204651_305_l01_shot01_fused_12x_usb` | Tag 13 | ` 30.52 mm` | ` 0.74°` | ` 0.83 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 6 | ` 49.54 mm` | ` 4.88°` | ` 1.18 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 7 | ` 46.13 mm` | ` 4.79°` | ` 0.80 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 8 | ` 44.00 mm` | ` 2.15°` | ` 1.19 px` |
| `pose15_pose_20261027_204658_169_l01_shot01_fused_24x_usb` | Tag 10 | `  3.56 mm` | `63.39°` | ` 1.42 px` |
