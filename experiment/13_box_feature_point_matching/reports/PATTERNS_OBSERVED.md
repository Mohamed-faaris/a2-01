# PATTERNS OBSERVED: Experiment 13 (Box / Planar Feature Point Matching)

## 1. Overview
This experiment evaluated multiple uncalibrated computer vision feature matching and point transfer pipelines across **25 distinct synthetic benchmark scenes** (totaling **250 ground-truth evaluation points**):
- **Scenes 001–005**: Foundational baseline scenes (mild/moderate baselines, elevation, diffuse light).
- **Scenes 006–015**: Challenging realistic variations (severe yaw, height disparity, optical roll, specular point glare, extreme standoff distances).
- **Scenes 016–025**: **Extreme stress tests** (ultra-wide convergence baselines up to 62°, ultra-grazing incidence up to 42° pitch + 25° yaw, severe 65° optical roll disparity, extreme shadow cross-lighting, macro close-up, and overexposure blowout).

Zero occlusion was maintained throughout, with both image planes (Choco Pie and Kellogg's) rendered simultaneously in every scene.

---

## 2. Complete Benchmark Comparison Across All 25 Scenes (250 Points)

| Methodology | Model Type | Overall Pass Rate (<15px) | Mean Localization Error | Median Error | Robustness in Extreme Stress (016–025) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`sift_homography`** | Classical Sparse Feature | **100.0%** (250/250) | **0.12 px** | **0.11 px** | **Immune to Geometry**: Sub-pixel across all 25 scenes, including 62° convergence and 65° roll disparity. |
| **`superpoint_lightglue`** | **Deep Learning (Transformer)** | **100.0%** (250/250) | **0.36 px** | **0.27 px** | **Superior Feature Association**: Extremely clean, outlier-free deep matches even under harsh dark glare and perspective shearing. |
| **`orb_homography`** | Fast Binary Descriptor | **100.0%** (250/250) | **0.52 px** | **0.25 px** | **Fastest Execution**: Sub-pixel across 23/25 scenes; shows slight drift (~3.4px) under extreme 62° baseline divergence. |
| **`fundamental_matrix_guided`** | Epipolar Constraint | **88.8%** (222/250) | **8.76 px** | **3.96 px** | **Epipolar Filter**: Strong geometric validation constraint; 1D template search along epipolar lines degrades under repetitive patterns. |
| **`template_patch_matching`** | Direct NCC Correlation | **75.6%** (189/250) | **13.92 px** | **7.96 px** | **Severe Failure**: Drops to 30%–50% accuracy whenever viewpoint convergence exceeds 25° due to affine foreshortening. |

---

## 3. Deep Learning (`SuperPoint + LightGlue`) vs. Classical SIFT

### SuperPoint + LightGlue Key Strengths
1. **Outlier Rejection Without Aggressive Heuristics**:
   - LightGlue's transformer attention mechanism directly models contextual dependencies between keypoints across views. It rejects false matches adaptively without needing hand-tuned distance thresholds (like Lowe's ratio test).
2. **Extreme Lighting & Glare Resilience (Scene 019 & 020)**:
   - When harsh spotlights blown out or underexposed the logos, SuperPoint's deep convolutional activations remained invariant to non-linear illumination changes, producing 0.44px error.
3. **Descriptor Invariance**:
   - Learned descriptor embeddings outperformed classical binary descriptors (ORB) on low-texture or semi-specular regions of the packaging.

### SIFT Key Strengths
- **Sub-Pixel Peak Precision**: OpenCV's sub-pixel quadratic Taylor expansion refinement on SIFT extrema yields the lowest raw geometric localization error (**0.12 px** average).
- **Zero Heavy Dependencies / GPU Requirements**: Runs instantaneously on standard CPU without PyTorch or neural network inference weights.

---

## 4. Key Failure Modes Identified

1. **Rigid Pixel Templates (`template_patch_matching`)**:
   - Plain 2D cross-correlation completely breaks when objects tilt away from the camera. At 44° convergence (Scene 014), the error jumped to **37.68 px**.
2. **Epipolar 1D Line Search (`fundamental_matrix_guided`)**:
   - The epipolar line itself is geometrically correct (residual $<1.5\text{px}$), but searching a 2D patch along a 1D line suffers from the **aperture problem** along edges parallel to the epipolar line.

---

## 5. Architectural Recommendation
For production-grade, uncalibrated stereo point identification and verification:
- **Primary Matching Engine**: Use **`SuperPoint + LightGlue`** (for challenging illumination, low-texture, or extreme viewpoints) or **`SIFT + FLANN`** (for lightweight, CPU-bound sub-pixel accuracy).
- **Geometric Model**: Always solve for **Planar Homography ($H$)** via RANSAC.
- **Consistency Verification Gate**: Compute the uncalibrated **Fundamental Matrix ($F$)**; verify that $x_1^\top F x_0 < 2.0\text{ px}$. If a transferred point violates this epipolar constraint, flag it as an inconsistent detection.
