# Real-World AprilTag Tracker & Robot Connection

This program connects the real-world multi-camera AprilTag tracking system in `demos/a2-01` (`demos/a2-04`) to the DOBOT CR5 MoveIt planning and pick-and-place execution stack.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Cameras["Real World Cameras"]
        C1["Camera 0 (See3Cam / USB)"]
        C2["Camera 1 (RealSense D415)"]
    end

    subgraph VisionApp["demos/a2-01 (Python 3.11, .venv)"]
        Tracker["MultiCameraPlanesCubePipeline\n(AprilTags 0, 4, 20..25, 42..45)"]
        Streamer["LiveStreamBroadcaster\n(TCP port 9876 + outputs/live_tracking_results.json)"]
        Stability["Stability Monitor\n(Variance < 2mm over 1s -> LOCKED)"]
        Cameras --> Tracker
        Tracker --> Streamer
        Tracker --> Stability
    end

    subgraph RobotStack["ROS 2 Humble / moveit-dev Container"]
        Bridge["cr5_live_tracker_bridge & demo_tracker.py\n(/scene/input + /scene_tasks/input)"]
        MoveIt["MoveIt 2 move_group & RViz\n(Robot TCP Frame + 13-stage Transfer Server)"]
        Streamer -->|TCP Socket Stream (20 Hz)| Bridge
        Stability -->|Pick & Place Trigger| Bridge
        Bridge --> MoveIt
    end
```

---

## What It Tracks:

1. **Ground Collision Floor**:
   - Tracked from **Tag 4** height ($Z \approx -0.7429\text{ m}$).
   - Generates a **$10\text{ m} \times 10\text{ m}$** collision floor so the robot never collides with the real ground.
2. **Source Platform (`platform_box`)**:
   - Tracked from **Plane B (Tags 42, 43, 44, 45)** at $Z \approx -0.492\text{ m}$.
   - Solid pedestal extending all the way down to the ground ($H = 0.251\text{ m}$).
3. **Movable 5 cm Cube (`box`)**:
   - Tracked from the **6-faced AprilTag Cube (Tags 20..25)**.
   - Rested at the platform center.
4. **Target Platform (`box2`)**:
   - Scaled along the target direction vector to a safe reachable radius ($600\text{ mm}$), surface set to **`base_link` height ($Z = 0.0\text{ m}$)**, extended all the way down to the ground ($H = 0.743\text{ m}$).

---

## How to Run:

### 1. Mock Mode (Test tracking and socket stream without physical cameras):
```bash
./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py --mock --port 9876
```

### 2. Live Multi-Camera Mode:
```bash
# Auto-detects connected V4L2 cameras:
./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py

# Or specify camera nodes explicitly:
./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py --cams /dev/video0 /dev/video2
```

### 3. Automatic Pick-and-Place Execution:
Add `--auto-trigger` so that as soon as the cube is placed and stable (spread < 2 mm for 1 second), it automatically commands the MoveIt transfer server:
```bash
./.venv/bin/python src/demos/multi-cam-planes-cube-tracker/realworld_connection.py --auto-trigger
```

### 4. Interactive Keyboard Controls:
- **`T`**: Manually trigger pick-and-place execution.
- **`S`**: Save snapshot bundle (raw camera frames + annotated UI + JSON).
- **`V`**: Toggle 3D software rendering view panel.
- **`Q` / `ESC`**: Clean shutdown.
