import sys
import json
import math
from pathlib import Path
import numpy as np
from ultralytics import YOLO

from can_extreme_angles_100_180 import (
    run_extreme_experiment, CAN_RADIUS_M, CAN_HEIGHT_M
)

def run(idx: int):
    rng = np.random.default_rng(10180 + idx)
    exp_id = f"{idx:03d}"
    exp_dir = Path("experiment/can_extreme_angles") / exp_id
    
    marker_size = round(float(rng.uniform(0.08, 0.10)), 3)
    marker_id = int(rng.choice([0, 1, 2, 3, 4]))
    marker_pos = (round(float(rng.uniform(-0.02, 0.02)), 3), round(float(rng.uniform(-0.04, 0.04)), 3), 0.001)

    cans = [
        {"name": "can_red", "color": [0.85, 0.08, 0.08],
         "position": [round(float(rng.uniform(-0.18, -0.08)), 3), round(float(rng.uniform(-0.05, 0.05)), 3), round(float(rng.uniform(0.18, 0.28)), 3)],
         "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.3, 0.3)), 3)],
         "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        {"name": "can_green", "color": [0.08, 0.85, 0.08],
         "position": [round(float(rng.uniform(-0.04, 0.04)), 3), round(float(rng.uniform(-0.04, 0.06)), 3), round(float(rng.uniform(0.26, 0.38)), 3)],
         "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.3, 0.3)), 3)],
         "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
        {"name": "can_blue", "color": [0.08, 0.20, 0.88],
         "position": [round(float(rng.uniform(0.08, 0.18)), 3), round(float(rng.uniform(-0.05, 0.05)), 3), round(float(rng.uniform(0.16, 0.26)), 3)],
         "rotation": [round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.2, 0.2)), 3), round(float(rng.uniform(-0.3, 0.3)), 3)],
         "radius": CAN_RADIUS_M, "height": CAN_HEIGHT_M},
    ]

    target_angle_deg = float(rng.uniform(110.0, 165.0))
    angle_rad = math.radians(target_angle_deg)
    phi0 = float(rng.uniform(-2.6, -2.0))
    r0 = float(rng.uniform(0.95, 1.15))
    z0 = float(rng.uniform(0.40, 0.60))
    cam0_pos = (round(r0 * math.cos(phi0), 3), round(r0 * math.sin(phi0), 3), round(z0, 3))

    phi1 = phi0 + angle_rad
    r1 = float(rng.uniform(0.95, 1.15))
    z1 = float(rng.uniform(0.35, 0.55))
    cam1_pos = (round(r1 * math.cos(phi1), 3), round(r1 * math.sin(phi1), 3), round(z1, 3))

    v0 = -np.array(cam0_pos)
    v1 = -np.array(cam1_pos)
    actual_angle_deg = math.degrees(math.acos(float(np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))))

    model = YOLO("/home/mfk01/Projects/econ-systems/demos/demo2/yolo26s.pt")
    evals = run_extreme_experiment(exp_dir, model, marker_size, marker_id, marker_pos, cans, cam0_pos, cam1_pos)
    
    result = {
        "exp_id": exp_id,
        "angle_deg": round(actual_angle_deg, 1),
        "cam0": cam0_pos,
        "cam1": cam1_pos,
        "yolo_dlt": evals["yolo_dlt"]["mean_corner_distance_error_cm"],
        "contour_dlt": evals["contour_dlt"]["mean_corner_distance_error_cm"]
    }
    print(f"DONE_{idx}: {json.dumps(result)}")

if __name__ == "__main__":
    run(int(sys.argv[1]))
