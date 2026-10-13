import cv2
import numpy as np
from pathlib import Path

def create_photorealistic_can_label(out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 1024x512 label wrapping around cylinder
    h, w = 512, 1024
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Classic red background with gradient
    for y in range(h):
        # subtle vertical shading
        factor = 0.85 + 0.15 * np.sin(np.pi * y / h)
        img[y, :, :] = (int(25 * factor), int(25 * factor), int(220 * factor))
    
    # White ribbon / swoosh
    pts = np.array([
        [0, 320], [250, 310], [500, 340], [750, 360], [1024, 330],
        [1024, 380], [750, 410], [500, 390], [250, 360], [0, 370]
    ], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    
    # Repeat "Coca-Cola" on front and back
    cv2.putText(img, "Coca-Cola", (60, 240), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.2, (255, 255, 255), 8, cv2.LINE_AA)
    cv2.putText(img, "Coca-Cola", (570, 240), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3.2, (255, 255, 255), 8, cv2.LINE_AA)
    
    # Nutrition / barcode elements for realism
    cv2.rectangle(img, (470, 150), (530, 270), (255, 255, 255), -1)
    for x in range(475, 525, 4):
        cv2.line(img, (x, 160), (x, 260), (0, 0, 0), 2)
        
    cv2.imwrite(str(out_path), img)
    print("Saved texture to:", out_path)

if __name__ == "__main__":
    create_photorealistic_can_label(Path("experiment/can_triangulation/shared/can_label.png"))
