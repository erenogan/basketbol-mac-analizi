import cv2, json, sys, numpy as np
from pathlib import Path
from ultralytics import YOLO

# Tespit (oyuncu recall) referansi icin aday paneli.
# Secilen karelerde en guclu iki ayarin (yolo11x + yolov8x, imgsz 1920, conf 0.05)
# birlesimi aday kutu; etiketleyen hangi adaylarin sahadaki oyuncu oldugunu ve
# kutusuz kalan oyuncu sayisini yazar -> data/tespit_ref_<video>.json
# kullanim: python scripts/16_tespit_referans.py dev
AD = sys.argv[1] if len(sys.argv) > 1 else "dev"
VIDEO, N = f"data/{AD}.mp4", 16
OUT = Path("out") / AD
saha = np.array(json.load(open("data/saha.json"))["saha"], dtype=np.int32)

# olu top disinda esit aralikli kareler (14_yumusat ciktisindan)
olu = {k["kare"] for k in json.load(open(OUT / "izler_saha.json")) if k["olu_top"]}
cap = cv2.VideoCapture(VIDEO)
toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
aday_kareler = [f for f in range(30, toplam - 30) if f not in olu]
kareler = [aday_kareler[int(i)] for i in np.linspace(0, len(aday_kareler) - 1, N)]

modeller = [YOLO("yolo11x.pt"), YOLO("yolov8x.pt")]
cikti = []
for j, f in enumerate(kareler):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f)
    _, im = cap.read()
    kutular, skor = [], []
    for m in modeller:
        r = m.predict(im, imgsz=1920, conf=0.05, iou=0.5, classes=[0], device=0,
                      half=True, verbose=False)[0]
        kutular += r.boxes.xyxy.cpu().numpy().tolist(); skor += r.boxes.conf.cpu().numpy().tolist()
    # iki modelin cakisan kutularini birlestir (NMS)
    keep = cv2.dnn.NMSBoxes([[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in kutular],
                            skor, 0.05, 0.5)
    adaylar = []
    for i in np.array(keep).ravel():
        x1, y1, x2, y2 = map(int, kutular[i])
        if cv2.pointPolygonTest(saha, ((x1 + x2) / 2, y2), True) > -40:   # 40 px pay
            adaylar.append([x1, y1, x2, y2])
    adaylar.sort(key=lambda b: (b[0] + b[2]) / 2)
    vis = im.copy()
    cv2.polylines(vis, [saha], True, (0, 200, 255), 1)
    for n, (x1, y1, x2, y2) in enumerate(adaylar):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(vis, str(n), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(vis, str(n), (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(vis, f"#{j} kare {f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imwrite(str(OUT / f"tespit_ref_{j:02d}.jpg"), vis[380:1080])
    cikti.append({"kare": f, "adaylar": adaylar})
json.dump(cikti, open(OUT / "tespit_adaylar.json", "w"), indent=0)
print(f"{N} kare -> {OUT}/tespit_ref_XX.jpg, tespit_adaylar.json")
print("kareler:", kareler, " aday sayilari:", [len(c["adaylar"]) for c in cikti])
