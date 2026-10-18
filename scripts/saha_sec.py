import cv2, json, numpy as np
from pathlib import Path

FRAME = "out/kare.jpg"
OUT   = Path("data/saha.json")
SCALE = 0.6   # ekrana sigmasi icin kucultme

frame = cv2.imread(FRAME)
if frame is None:
    raise SystemExit(f"{FRAME} yok")

H, W = frame.shape[:2]
disp0 = cv2.resize(frame, (int(W * SCALE), int(H * SCALE)))
noktalar = []

def tikla(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        noktalar.append((int(x / SCALE), int(y / SCALE)))
    elif event == cv2.EVENT_RBUTTONDOWN and noktalar:
        noktalar.pop()

cv2.namedWindow("saha")
cv2.setMouseCallback("saha", tikla)
print("sol tik: nokta ekle | sag tik: geri al | s: kaydet | q: cik")

while True:
    disp = disp0.copy()
    if noktalar:
        pts = np.array([[int(x * SCALE), int(y * SCALE)] for x, y in noktalar])
        for i, p in enumerate(pts):
            cv2.circle(disp, tuple(p), 4, (0, 0, 255), -1)
            cv2.putText(disp, str(i), (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        if len(pts) > 1:
            cv2.polylines(disp, [pts], len(pts) > 2, (0, 255, 0), 2)
    cv2.imshow("saha", disp)

    k = cv2.waitKey(20) & 0xFF
    if k == ord('s'):
        if len(noktalar) < 3:
            print("en az 3 nokta gerekli")
            continue
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps({"saha": noktalar}, indent=2))
        print(f"kaydedildi -> {OUT} ({len(noktalar)} nokta)")
        break
    if k == ord('q'):
        print("iptal")
        break

cv2.destroyAllWindows()