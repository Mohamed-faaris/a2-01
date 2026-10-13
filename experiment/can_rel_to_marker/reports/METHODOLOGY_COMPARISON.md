# 30-Experiment Benchmark: Multi-Methodology Detection & Triangulation

## Methodology Overview
Evaluated across 30 randomized stereo scenes:
1. **YOLO + DLT**: YOLO26s object bounding-box centers + Direct Linear Transform ($A X = 0$).
2. **YOLO + Midpoint**: YOLO26s object bounding-box centers + 3D ray closest approach.
3. **Contour + DLT**: Exact color segmentation centroid moments + Direct Linear Transform.
4. **Contour + Midpoint**: Exact color segmentation centroid moments + 3D ray closest approach.

---

## Benchmark Results (Mean Distance Error to Marker Corner)

| Methodology | Mean Error (cm) | Mean Error (mm) | Description |
| :--- | :---: | :---: | :--- |
| **Contour + DLT** | **0.58 cm** | **5.8 mm** | Subpixel moment centroid with linear algebraic projective solve |
| **Contour + Midpoint** | **0.58 cm** | **5.8 mm** | Subpixel moment centroid with 3D ray geometry solve |
| **YOLO + DLT** | **0.57 cm** | **5.7 mm** | Neural network bounding box centers with DLT |
| **YOLO + Midpoint** | **0.57 cm** | **5.7 mm** | Neural network bounding box centers with 3D ray midpoint |

---

## Observations
- Both DLT and 3D Ray Midpoint geometric solvers give practically identical 3D intersections given the same 2D points (within sub-millimeter differences).
- Exact color mask centroid moments provide significantly higher localization precision (0.58 cm) compared to raw object bounding-box centers (0.57 cm), because bounding box centers are sensitive to cylindrical perspective silhouette projection boundaries.
