import cv2
from pathlib import Path

VIDEO = Path("data/smoke.mp4")
OUT = Path("out")
OUT.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    raise SystemExit(f"Video acilamadi: {VIDEO.resolve()}")

fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"cozunurluk : {w}x{h}")
print(f"fps        : {fps:.2f}")
print(f"kare sayisi: {total}")
print(f"sure       : {total/fps:.1f} sn")

cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
ok, frame = cap.read()
cap.release()

if not ok:
    raise SystemExit("Kare okunamadi")

cv2.imwrite(str(OUT / "kare.jpg"), frame)
print("kaydedildi -> out/kare.jpg")