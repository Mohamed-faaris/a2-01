# Analysis & Patterns Observed: Experiment 10 (Mixed FOVs + In-Plane Optical Z Roll)

## 1. Experiment Overview
- **Identifier**: `10_can_mixed_fov_optical_roll`
- **Total Scenes**: 15 distinct scene geometries
- **Variables Tested**:
  1. **Heterogeneous Camera FOVs**: Camera 0 Wide ($75^\circ - 90^\circ$ H-FOV) vs Camera 1 Narrow/Standard ($45^\circ - 60^\circ$ H-FOV).
  2. **In-Plane Camera Rotations around Optical Z-Axis (Roll)**: Ranging from $-40^\circ$ to $+40^\circ$.
  3. **3D Free-Floating Targets**: 3 cylinder cans with arbitrary heights ($Z \in [0.08, 0.35]\text{ m}$) and 3D tilts.
  4. **Marker Reference**: Ground AprilTag (id 3, size 0.088 m).

---

## 2. Benchmark Summary Metrics (15 Scenes, 60 Evaluations)

| Methodology | Pass Rate ($\ge 3$ Cans & $<5\text{ cm}$) | Valid Mean Error | Avg Cans Triangulated |
|---|---|---|---|
| **`contour_dlt`** | **86.7% (13 / 15)** | **0.44 cm (4.4 mm)** | **2.9 / 3.0** |
| **`contour_midpoint`** | **86.7% (13 / 15)** | **0.44 cm (4.4 mm)** | **2.9 / 3.0** |
| **`yolo_dlt`** | 6.7% (1 / 15) | 0.28 cm (2.8 mm) | 0.3 / 3.0 |
| **`yolo_midpoint`** | 6.7% (1 / 15) | 0.27 cm (2.7 mm) | 0.3 / 3.0 |

---

## 3. Key Algorithmic Patterns & Insights

1. **In-Plane Optical Roll Impact on Axis-Aligned Bounding Box Detectors (YOLO)**:
   - Standard axis-aligned YOLO models struggle when camera roll exceeds $\pm 15^\circ$ because upright cylinders become diagonal in pixel coordinates.
   - When YOLO *does* detect pairs (e.g. at small roll angles like Scene `014` with $+14.5^\circ$ and $-5.5^\circ$), its accuracy is high (**0.24 cm**).

2. **HSV Spatial Contour Centroid Invariance**:
   - `contour_dlt` and `contour_midpoint` are inherently invariant to optical roll around the camera axis.
   - Across roll angles up to $\pm 39.8^\circ$, contour segmentation maintains an **86.7% pass rate** and **sub-5 mm triangulation accuracy** (overall average **0.44 cm**).

3. **DLT vs. Multi-K Ray Midpoint Equivalence**:
   - Both Direct Linear Transform (`contour_dlt`) and the heterogeneous intrinsic Ray Midpoint solver (`contour_midpoint`) perform identically down to the sub-millimeter level across all mixed FOV pairs.
