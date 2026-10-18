# Takim siniflandirma icin KOR referans panosu: tahmin gosterilmez,
# etiketleyen sadece kirpintiyi ve sira numarasini gorur.
import cv2, json, numpy as np
from collections import defaultdict

import sys
from pathlib import Path
from ortak import video_yolu
# kullanim: python scripts/09_referans_pano.py dev 2
AD   = sys.argv[1] if len(sys.argv) > 1 else "smoke"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
N    = 80
OUT  = Path("out") / AD
kayit = json.load(open(OUT / "izler.json"))
sec = sorted(np.random.default_rng(SEED).choice(len(kayit), N, replace=False).tolist())
by = defaultdict(list)
for j, i in enumerate(sec):
    by[kayit[i]["kare"]].append((j, i))

hucre = {}
cap = cv2.VideoCapture(video_yolu(AD)); kare = 0
while True:
    ok, f = cap.read()
    if not ok: break
    for j, i in by.get(kare, []):
        x1, y1, x2, y2 = kayit[i]["kutu"]
        h = cv2.resize(f[max(y1,0):y2, max(x1,0):x2], (110, 220))
        h = cv2.copyMakeBorder(h, 0, 26, 2, 2, cv2.BORDER_CONSTANT, value=(0,0,0))
        cv2.putText(h, str(j), (6, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        hucre[j] = h
    kare += 1
for p in range(0, N, 40):
    hs = [hucre[j] for j in range(p, min(p+40, N))]
    cv2.imwrite(str(OUT / f"referans_pano_s{SEED}_{p//40}.jpg"),
                np.vstack([np.hstack(hs[r:r+10]) for r in range(0, len(hs), 10)]))
json.dump(sec, open(OUT / f"referans_indeks_s{SEED}.json", "w"))
print(f"kaydedildi -> {OUT}/referans_pano_s{SEED}_*.jpg, referans_indeks_s{SEED}.json")
