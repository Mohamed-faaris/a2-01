# Analysis & Patterns Observed: Experiment 11 (Textured Floor & Background Distractors)

## 1. Experiment Overview
- **Identifier**: `11_can_textured_background`
- **Objective**: Introduce procedural high-frequency texture, Voronoi noise, and colorful background distractors (red/green checker patterns and textured walls) into the scene to evaluate the resilience of color contour segmentation vs. deep-learning object detection (YOLO).
- **Setup**:
  - Two calibrated cameras ($1080 \times 720$, 60° H-FOV) observing the workspace.
  - Multi-tone textured floor plane with Voronoi color noise.
  - Standing background poster plane with high-contrast red and green distractor shapes.
  - 3 target cans (Red, Green, Blue) with known ground truth metric poses.
  - Ground AprilTag reference marker.

---

## 2. Benchmark Comparison (5 Scenes, 20 Evaluations)

| Methodology | Pass Rate ($\ge 3$ Targets, $< 5\text{ cm}$) | Mean Error (Valid) | Average Candidate Blobs Detected (Cam0 / Cam1) | Triangulation Status |
|---|---|---|---|---|
| **`yolo_dlt`** | **100% (5 / 5)** | **0.73 cm (7.3 mm)** | **3.0 / 3.0** | Robust, zero false positives |
| **`yolo_midpoint`** | **100% (5 / 5)** | **0.73 cm (7.3 mm)** | **3.0 / 3.0** | Robust, zero false positives |
| **`contour_dlt`** | **0% (0 / 5)** | **41.50 cm** | **9.2 / 9.0** | Severe false-positive clutter & mismatch |
| **`contour_midpoint`** | **0% (0 / 5)** | **37.68 cm** | **9.2 / 9.0** | Severe false-positive clutter & mismatch |

---

## 3. Key Algorithmic Insights

1. **Failure Mode of Color Contour Segmentation in Textured / Cluttered Environments**:
   - While HSV color contouring performed well on flat uniform floors (Experiments 01–10), the presence of colorful background elements, wall distractors, and floor textures generates **8 to 10 candidate blobs per frame**.
   - These false-positive blobs corrupt cross-camera stereo feature matching: background tiles match with foreground cans, throwing triangulated 3D points completely off (errors exceeding 30 cm – 40 cm).

2. **Semantic Shape Robustness of Deep Learning (YOLO)**:
   - YOLO detects objects based on learned geometric structure (cylindrical aspect ratio, top/bottom rims, shadows) rather than raw pixel color thresholding.
   - Despite intense color clutter and distracting background patterns, YOLO maintained **exactly 3 detections per camera in 100% of scenes** with zero background false-positives and an average triangulated distance error of only **0.73 cm**.
