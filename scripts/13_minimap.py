import cv2, json, sys, shutil, subprocess, numpy as np
from collections import defaultdict, deque
from pathlib import Path
from saha_model import cizgiler, POTALAR, UZUN, EN

# Yan yana: solda kamera (takim kutulari + kalibre saha modeli), sagda kusbakisi tahta.
# Girdi: 14_yumusat ciktisi (izler_saha.json) — yumusatilmis metre koordinati.
# Tahtada dolu nokta = olcum, ici bos halka = kisa kopmada ara doldurma (tahmin).
# kullanim: python scripts/13_minimap.py dev
AD    = sys.argv[1] if len(sys.argv) > 1 else "smoke"
from ortak import video_yolu
VIDEO = video_yolu(AD)
OUT   = Path("out") / AD
IZ_SN = 1.0                     # tahtada noktanin arkasindaki iz (saniye)
PPM, PAY = 40, 1.0              # tahta: piksel/metre, saha disi kenar payi (m)
SOL_W, SOL_H = 1280, 720        # kamera paneli (1920x1080 -> 2/3)
RENK = {"kirmizi": (40, 40, 230), "lacivert": (230, 120, 20),
        "disarida": (150, 150, 150), "belirsiz": (230, 230, 230)}

H = np.array(json.load(open("data/kalibrasyon.json"))["H_piksel_saha"])
Hinv = np.linalg.inv(H)
kayit = json.load(open(OUT / "izler_saha.json"))
kare_index = defaultdict(list)
for k in kayit:
    kare_index[k["kare"]].append(k)

TW, TH = int((UZUN + 2 * PAY) * PPM), SOL_H
oy = (TH - int((EN + 2 * PAY) * PPM)) // 2           # tahtayi dikeyde ortala
def m2p(x, y):
    return int((x + PAY) * PPM), int((y + PAY) * PPM) + oy

tahta0 = np.full((TH, TW, 3), (60, 95, 150), np.uint8)    # parke rengi
cv2.rectangle(tahta0, m2p(0, 0), m2p(UZUN, EN), (75, 120, 185), -1)
for pts, kapali in cizgiler():
    p = np.array([m2p(x, y) for x, y in pts], np.int32)
    cv2.polylines(tahta0, [p], kapali, (255, 255, 255), 2, cv2.LINE_AA)
for px, py in POTALAR:
    cv2.circle(tahta0, m2p(px, py), 6, (0, 140, 255), 2)

# kamera paneli icin saha modeli overlay (sabit, bir kez hazirla)
overlay = []
for pts, kapali in cizgiler():
    p = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    overlay.append((np.int32(p * SOL_W / 1920), kapali))

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
ham = str(OUT / "minimap.mp4")
yazici = cv2.VideoWriter(ham, cv2.VideoWriter_fourcc(*"mp4v"), fps, (SOL_W + TW, SOL_H))
izler = defaultdict(lambda: deque(maxlen=int(IZ_SN * fps)))
son_gorulme = {}

kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    sol = cv2.resize(frame, (SOL_W, SOL_H))
    for p, kapali in overlay:
        cv2.polylines(sol, [p], kapali, (0, 255, 0), 1, cv2.LINE_AA)
    tahta = tahta0.copy()
    tahmin = set()
    for k in kare_index.get(kare, []):
        t = k["takim"]
        if k["kutu"]:
            x1, y1, x2, y2 = (int(v * SOL_W / 1920) for v in k["kutu"])
            cv2.rectangle(sol, (x1, y1), (x2, y2), RENK[t], 2 if t != "belirsiz" else 1)
        izler[k["iz"]].append((m2p(k["sx"], k["sy"]), t))
        son_gorulme[k["iz"]] = kare
        if k["ara_doldurma"]:
            tahmin.add(k["iz"])
    for iz_id, iz in izler.items():
        if son_gorulme[iz_id] != kare:
            continue                       # bu karede yok: noktasini cizme
        pts = [p for p, _ in iz]
        t = iz[-1][1]
        for i in range(1, len(pts)):
            cv2.line(tahta, pts[i - 1], pts[i], RENK[t], 1, cv2.LINE_AA)
        r = 9 if t in ("kirmizi", "lacivert") else 6
        if iz_id in tahmin:
            cv2.circle(tahta, pts[-1], r, RENK[t], 2, cv2.LINE_AA)
        else:
            cv2.circle(tahta, pts[-1], r, RENK[t], -1, cv2.LINE_AA)
            cv2.circle(tahta, pts[-1], r, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(sol, f"kare {kare}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    yazici.write(np.hstack([sol, tahta]))
    kare += 1
    if kare % 600 == 0:
        print(f"  {kare}")
cap.release()
yazici.release()
print(f"kaydedildi -> {ham}")

try:
    exe = shutil.which("ffmpeg")
    if exe is None:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    hedef = ham.replace(".mp4", "_izle.mp4")
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", ham, "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", hedef], check=True)
    print(f"izlenebilir surum -> {hedef}")
except Exception as e:
    print(f"donusturme basarisiz: {e}")
