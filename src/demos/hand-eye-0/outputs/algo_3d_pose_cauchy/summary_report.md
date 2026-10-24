# Hand-Eye Multi-Pose Calibration Report (Demo `hand-eye-0`)

## 1. Executive Summary
- **2D Reprojection Error (Tool Tags)**: `1.41 px RMS` (Mean: `1.06 px`, P90: `1.98 px`, Max: `4.64 px` on 1280x720)
- **2D Reprojection Error (Static Tags)**: `0.21 px RMS` (Mean: `0.18 px`, Max: `0.65 px` — sub-pixel accuracy)
- **3D Position Residual (Tool Tags)**: `5.19 mm RMS` (Mean: `2.84 mm`, Max: `17.87 mm`)
- **3D Rotation Residual (Tool Tags)**: `3.33° RMS` (Max: `12.54°`)
- **Total Tool Tag Sightings**: `20` across 18 arm poses
- **Static Workpiece Coplanarity RMSE**: `2.11 mm` (plane normal: `[+0.016, +0.059, +0.998]`)

---

## 2. Solved Camera Extrinsics in `base_link` Frame

### D415 Camera (`T_base_link -> d415`):
- **Position $(X, Y, Z)$**: `[ +1.936,  -0.309,  +0.792] m`
- **Quaternion $(qx, qy, qz, qw)$**: `[+0.6979, +0.5332, -0.3659, -0.3077]`

```yaml
T_base_d415:
  xyz: [1.935955, -0.308975, 0.792143]
  quat: [0.697927, 0.533217, -0.365939, -0.307679]
```

### USB Camera (`T_base_link -> usb`):
- **Position $(X, Y, Z)$**: `[ +1.987,  -0.193,  +0.306] m`
- **Quaternion $(qx, qy, qz, qw)$**: `[+0.0130, +0.7613, +0.1075, -0.6393]`

```yaml
T_base_usb:
  xyz: [1.987179, -0.193492, 0.305516]
  quat: [0.012999, 0.761275, 0.107507, -0.639323]
```

---

## 3. Tool Mountings in `tcp` Frame (`T_tcp -> tag`)

| Tag ID | Link | Translation $(X,Y,Z)$ [mm] | Distance to TCP Origin [mm] |
|:---:|:---:|:---|:---:|
| **Tag 20** | `tcp` | `[ +43.1,  +15.0,   -2.1]` | `45.7 mm` |
| **Tag 21** | `tcp` | `[ +16.6,  +15.1,  -26.0]` | `34.4 mm` |
| **Tag 40** | `tcp` | `[ -84.8,   +3.3, -136.1]` | `160.4 mm` |

---

## 4. Static Tags in `base_link` Reference Frame

| Tag ID | Role / Link | Position $(X,Y,Z)$ [mm] | Height $Z$ [mm] |
|:---:|:---|:---|:---:|
| **Tag  4** | Base Cluster (`base_link`) | `[  -60.7,  +238.0,  -699.9]` | ` -699.9` |
| **Tag  5** | Base Cluster (`base_link`) | `[  -34.6,  +138.3,  -685.6]` | ` -685.6` |
| **Tag 42** | Workpiece Box (`box`) | `[ +613.6,  -239.0,  -478.9]` | ` -478.9` |
| **Tag 43** | Workpiece Box (`box`) | `[ +615.0,   +27.6,  -490.4]` | ` -490.4` |
| **Tag 44** | Workpiece Box (`box`) | `[ +896.8,    +5.7,  -497.7]` | ` -497.7` |
| **Tag 45** | Workpiece Box (`box`) | `[ +888.7,  -255.6,  -478.0]` | ` -478.0` |

### Workpiece Platform Plane (Tags 42, 43, 44, 45):
- **Consensus Surface Normal**: `[+0.016, +0.059, +0.998]`
- **Center Position**: `[ +753.5,  -115.3,  -486.3] mm`
- **Coplanarity RMSE**: `2.11 mm`

---

## 5. Per-Pose Sighting Residuals & Reprojection Errors

| Pose Name | Tag ID | 3D Position Error [mm] | 3D Rotation Error [deg] | 2D Reprojection Error [px] |
|:---|:---:|:---:|:---:|:---:|
| `pose01_ready` | Tag 40 | `0.97 mm` | `0.61°` | `0.55 px` |
| `pose01_ready` | Tag 20 | `3.06 mm` | `0.26°` | `0.42 px` |
| `pose02_yaw_m40` | Tag 40 | `0.47 mm` | `2.71°` | `1.06 px` |
| `pose04_yaw_m20` | Tag 40 | `2.13 mm` | `2.07°` | `0.78 px` |
| `pose04_yaw_m20` | Tag 20 | `9.51 mm` | `1.77°` | `4.18 px` |
| `pose05_yaw_p20` | Tag 40 | `1.26 mm` | `0.78°` | `1.23 px` |
| `pose06_pitch_up15` | Tag 40 | `5.36 mm` | `1.00°` | `0.27 px` |
| `pose06_pitch_up15` | Tag 20 | `1.40 mm` | `0.36°` | `1.26 px` |
| `pose07_pitch_dn15` | Tag 40 | `0.50 mm` | `3.63°` | `0.41 px` |
| `pose07_pitch_dn15` | Tag 20 | `3.74 mm` | `12.54°` | `1.97 px` |
| `pose08_wrist_dn25` | Tag 40 | `2.81 mm` | `3.20°` | `0.90 px` |
| `pose08_wrist_dn25` | Tag 20 | `1.11 mm` | `1.85°` | `0.91 px` |
| `pose09_wrist_up25` | Tag 40 | `4.63 mm` | `1.20°` | `0.93 px` |
| `pose10_roll_m45` | Tag 40 | `2.01 mm` | `2.13°` | `1.22 px` |
| `pose11_roll_p45` | Tag 40 | `1.77 mm` | `2.03°` | `0.94 px` |
| `pose16_comb_5` | Tag 40 | `5.30 mm` | `2.32°` | `0.52 px` |
| `pose16_comb_5` | Tag 20 | `17.87 mm` | `1.42°` | `2.28 px` |
| `pose16_comb_5` | Tag 21 | `0.00 mm` | `0.00°` | `0.22 px` |
| `pose18_ready_return` | Tag 40 | `1.07 mm` | `1.39°` | `0.62 px` |
| `pose18_ready_return` | Tag 20 | `1.19 mm` | `0.97°` | `0.54 px` |