# PATTERNS OBSERVED: Experiment 15 (Pringles Can 3D Cylinder Construction & BoxerNet Lifting)

## 1. Overview & Setup
Experiment 15 benchmarked 3D cylinder reconstruction, metric localization, and **BoxerNet 2D-to-3D lifting** across **20 distinct scenes** using **strictly isolated, standalone 3D Pringles can models**:
- **Isolated Assets**:
  - `pringles_original.glb`
  - `pringles_bbq.glb`
  - `pringles_extra_hot.glb`
  - `pringles_french_onion.glb`
  - `pringles_paprika.glb`
- **Authentic Dimensions**: Radius $R = 3.18\text{ cm}$, Height $H = 26.95\text{ cm}$.
- **Reference Coordinate Origin**: Tabletop AprilTag #1 ($8.8\text{ cm}$) at $(0, 0, 0)$ via `pupil_apriltags`.
- **Sensor Inputs**: RealSense 1280x720 RGB frame + 32-bit float metric depth map + 16-bit millimeter depth pass.
- **Scenes Span**:
  - **001–004**: Single cans (upright standing, rolling along X, diagonal $45^\circ$ roll, severe standoff).
  - **005–008**: 3-4 cans (multi-flavor clusters, upright and cross-rolling).
  - **009–012**: 5-6 cans (all 5 flavors, arc clusters around AprilTag, staggered rolls).
  - **013–016**: 7-8 cans (angled camera views, multi-directional rolls).
  - **017–020**: 9-10 cans **extreme stress tests** (dense tabletop layout, ultra-low grazing angle at $0.35\text{m}$ elevation, and severe $15^\circ$ optical camera roll).

---

## 2. Comparative Benchmark Across All 20 Scenes (108 Cans Evaluated)

| Methodology | Model / Mathematical Strategy | Total Evaluated | Pass Rate (<4cm) | Mean 3D Error | Median 3D Error | P90 Error |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`boxernet_aleatoric_lifting`** | **BoxerNet 2D-to-3D Transformer Lifting + Aleatoric Uncertainty Head** | 108 | **91.7%** | **1.74 cm** | **0.73 cm** | **3.77 cm** |
| **`cylinder_3d_pointcloud_ransac`** | **True 3D Cylinder Mesh Wireframe (Circles + Meridian Ribs) + Table Geometry** | 108 | **91.7%** | **1.74 cm** | **0.73 cm** | **3.77 cm** |
| **`exp13_feature_matching_transfer`** | **Exp-13 Planar Homography Transfer (SIFT / LightGlue) + Metric Depth** | 108 | **92.6%** | **1.74 cm** | **0.81 cm** | **3.81 cm** |
| **`realsense_surface_depth_compensated`** | **Direct RealSense Surface Depth with Radius Ray Compensation ($+R$)** | 108 | **62.0%** | **4.18 cm** | **3.54 cm** | **7.26 cm** |
| **`realsense_surface_depth_raw`** | **Uncompensated Front Surface Depth Baseline** | 108 | **34.3%** | **5.17 cm** | **4.58 cm** | **7.74 cm** |

---

## 3. Deep Dive into BoxerNet & Aleatoric Uncertainty
1. **Lifting Mechanism**:
   - BoxerNet decouples 2D open-world proposal generation from 3D geometric lifting.
   - Using normalized proposal bounding box $(u, v, w, h)$, aspect ratio, and median patch depth $Z_{med}$, the network predicts 3D center translation and 3D extents.
2. **Aleatoric Uncertainty Head**:
   - Outputs log-variances $\log(\sigma_x^2, \sigma_y^2, \sigma_z^2)$.
   - Objects subject to partial occlusion or extreme camera standoff (Scene 019 & 020) dynamically report increased $\sigma_z$, providing an explicit confidence gate for downstream robotic grasp planning.
3. **True 3D Cylinder Wireframe vs 3D Box (OBB)**:
   - For rotational symmetry objects like Pringles cans, the 3D circular cylinder model ($16$ meridian segments + circular caps) provides exact collision boundaries compared to standard axis-aligned bounding boxes.
