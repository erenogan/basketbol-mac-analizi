import cv2, json, numpy as np
from ultralytics import YOLO

PERSON = 0
IMGSZ, CONF = 1280, 0.25

saha  = np.array(json.load(open("data/saha.json"))["saha"], dtype=np.int32)
model = YOLO("yolov8m.pt")
frame = cv2.imread("out/kare.jpg")
vis   = frame.copy()

r = model.predict(frame, imgsz=IMGSZ, conf=CONF, iou=0.5,
                  classes=[PERSON], device=0, verbose=False)[0]

print(f"{'id':>3} {'ayak_x':>7} {'ayak_y':>7} {'yuk':>5} {'conf':>5}  durum")
print("-" * 44)

for i, b in enumerate(r.boxes):
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    ayak = (float((x1 + x2) / 2), float(y2))
    yuk  = y2 - y1
    conf = float(b.conf[0])
    ic   = cv2.pointPolygonTest(saha, ayak, False) >= 0

    print(f"{i:>3} {ayak[0]:>7.0f} {ayak[1]:>7.0f} {yuk:>5} {conf:>5.2f}  "
          f"{'ICERDE' if ic else 'disarda'}")

    renk = (0, 255, 0) if ic else (120, 120, 120)
    cv2.rectangle(vis, (x1, y1), (x2, y2), renk, 2)
    cv2.circle(vis, (int(ayak[0]), int(ayak[1])), 4, renk, -1)
    cv2.putText(vis, str(i), (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2)

cv2.polylines(vis, [saha], True, (0, 200, 255), 2)
cv2.imwrite("out/tani.jpg", vis)
print("\nkaydedildi -> out/tani.jpg")