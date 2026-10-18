import cv2, json, numpy as np
from ultralytics import YOLO

PERSON = 0
IMGSZ, CONF = 1280, 0.25

saha = np.array(json.load(open("data/saha.json"))["saha"], dtype=np.int32)

model = YOLO("yolov8m.pt")
frame = cv2.imread("out/kare.jpg")

r = model.predict(frame, imgsz=IMGSZ, conf=CONF, classes=[PERSON],
                  device=0, verbose=False)[0]

icerde = disarda = 0
for b in r.boxes:
    x1, y1, x2, y2 = map(int, b.xyxy[0])
    ayak = (float((x1 + x2) / 2), float(y2))

    if cv2.pointPolygonTest(saha, ayak, False) >= 0:
        icerde += 1
        renk = (0, 255, 0)
    else:
        disarda += 1
        renk = (120, 120, 120)

    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
    cv2.circle(frame, (int(ayak[0]), int(ayak[1])), 4, renk, -1)

cv2.polylines(frame, [saha], True, (0, 200, 255), 2)
print(f"saha icinde : {icerde}")
print(f"saha disinda: {disarda}")
cv2.imwrite("out/filtre.jpg", frame)
print("kaydedildi -> out/filtre.jpg")