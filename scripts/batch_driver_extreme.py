import subprocess
import json
import csv
import sys
import os
from pathlib import Path

reports_dir = Path("experiment/can_extreme_angles/reports")
reports_dir.mkdir(parents=True, exist_ok=True)

results = []
env = dict(os.environ)
env["PYTHONPATH"] = "scripts"

for i in range(1, 11):
    cmd = ["/home/mfk01/.local/bin/uv", "run", "python", "scripts/run_single_extreme.py", str(i)]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    found = False
    for line in res.stdout.splitlines():
        if f"DONE_{i}:" in line:
            json_str = line.split(f"DONE_{i}:", 1)[1].strip()
            data = json.loads(json_str)
            results.append(data)
            print(f"[{data['exp_id']}/010] Angle: {data['angle_deg']}° | YOLO DLT: {data['yolo_dlt']} cm | Contour DLT: {data['contour_dlt']} cm")
            sys.stdout.flush()
            found = True
            break
    if not found:
        print(f"FAILED {i}: stdout={res.stdout} stderr={res.stderr}")

if results:
    (reports_dir / "extreme_100_180_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with open(reports_dir / "extreme_100_180_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\nSaved extreme angles summary!")
