import cv2
from ultralytics import YOLO

PERSON, BALL = 0, 32

model = YOLO("yolov8m.pt")

frame = cv2.imread("out/kare.jpg")
if frame is None:
    raise SystemExit("out/kare.jpg yok, once 01_kare.py calistir")

r = model.predict(frame, conf=0.25, classes=[PERSON, BALL],
                  device=0, verbose=False)[0]

kisi = top = 0
for box in r.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cls  = int(box.cls[0])
    conf = float(box.conf[0])

    if cls == PERSON:
        kisi += 1
        renk, etiket = (0, 255, 0), f"kisi {conf:.2f}"
    else:
        top += 1
        renk, etiket = (0, 0, 255), f"TOP {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
    cv2.putText(frame, etiket, (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1)

print(f"kisi sayisi: {kisi}")
print(f"top sayisi : {top}")

cv2.imwrite("out/tespit.jpg", frame)
print("kaydedildi -> out/tespit.jpg")