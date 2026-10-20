# AprilTag Locator & Annotator (`apriltag-locator`)

Locates AprilTags across single or multiple images, annotates them with bounding boxes, corners, and Tag IDs, and exports complete 2D and 3D tag locations in structured JSON for downstream 3D reconstruction and mapping.

Marker sizes and families are read directly from [calibrations/markers.yaml](../../../calibrations/markers.yaml).

## Features

- **Batch Image Support**: Pass multiple images, glob patterns (`"images/*.png"`), or directory paths.
- **Marker Config Integration**: Automatically configures physical tag dimensions and tag family definitions per tag ID using `calibrations/markers.yaml`.
- **Optional Camera Calibration**: Provide a `.npz`, `.json`, or `.yaml` camera calibration file to compute metric 3D positions (`translation_cam_xyz_m`, `rotation_cam_matrix`, `euler_xyz_deg`, and `distance_m`). If omitted, pixel locations and corners are exported accurately.
- **Visual Annotations**: Exports annotated images drawing tag borders, top-right orientation pins, IDs, physical sizes, and 3D coordinate axes.
- **Downstream-Ready JSON**: Outputs a comprehensive JSON containing tag centers, corner coordinates, and 3D poses.

## Usage

### 1. Basic Run (No calibration, 2D coordinates)

```bash
uv run apriltag-locator path/to/images/ --out-json tag_locations.json --out-dir annotated_output/
```

### 2. With Camera Calibration (2D + 3D pose extraction)

```bash
uv run apriltag-locator path/to/images/ \
  --calibration calibrations/calibarotor0/outputs/rl720 \
  --out-json tag_locations.json \
  --out-dir annotated_output/
```

### 3. Custom Markers Configuration

```bash
uv run apriltag-locator path/to/images/ \
  --markers-yaml calibrations/markers.yaml \
  --calibration camera.npz \
  --out-json locations.json
```

## JSON Output Format

```json
{
  "metadata": {
    "markers_yaml": "/path/to/calibrations/markers.yaml",
    "default_family": "tag36h11",
    "default_size_m": 0.0885,
    "calibration_applied": "calibrations/calibarotor0/outputs/rl720",
    "total_images_processed": 5,
    "total_tags_detected": 12
  },
  "images": [
    {
      "image_path": "path/to/image.png",
      "image_size": { "width": 1280, "height": 720 },
      "tags_count": 2,
      "tags": [
        {
          "id": 0,
          "family": "tag36h11",
          "size_m": 0.134,
          "size_mm": 134.0,
          "description": "Tabletop Reference Origin (134 mm)",
          "center_px": [642.5, 360.1],
          "corners_px": [
            [680.1, 320.4],
            [604.2, 321.1],
            [605.0, 399.8],
            [680.9, 399.1]
          ],
          "pose": {
            "translation_cam_xyz_m": [-0.02, 0.05, 0.85],
            "translation_cam_xyz_mm": [-20.0, 50.0, 850.0],
            "distance_m": 0.851,
            "euler_xyz_deg": [1.2, -0.5, 90.0],
            "reprojection_error_px": 0.42
          }
        }
      ],
      "annotated_image": "annotated_output/image_annotated.png"
    }
  ]
}
```
