import cv2, time
from ultralytics import YOLO

PERSON, BALL = 0, 32
model = YOLO("yolov8m.pt")

frame = cv2.imread("out/kare.jpg")
if frame is None:
    raise SystemExit("out/kare.jpg yok")

print(f"{'imgsz':>6} {'conf':>6} {'kisi':>5} {'top':>4} {'sure_ms':>8}")
print("-" * 34)

for imgsz in (640, 960, 1280, 1920):
    for conf in (0.25, 0.10):
        t0 = time.perf_counter()
        r = model.predict(frame, imgsz=imgsz, conf=conf,
                          classes=[PERSON, BALL], device=0, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        sinif = [int(c) for c in r.boxes.cls.tolist()]
        kisi = sinif.count(PERSON)
        top  = sinif.count(BALL)
        print(f"{imgsz:>6} {conf:>6.2f} {kisi:>5} {top:>4} {ms:>8.0f}")

        out = frame.copy()
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            renk = (0, 255, 0) if int(b.cls[0]) == PERSON else (0, 0, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), renk, 2)
        cv2.imwrite(f"out/ayar_{imgsz}_{int(conf*100)}.jpg", out)

print("\ngorseller out/ayar_*.jpg")