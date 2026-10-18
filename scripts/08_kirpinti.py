import cv2, json, numpy as np
from collections import defaultdict

# Her iz icin zamana yayilmis ornek kirpintilar -> tek bir etiketleme sayfasi.
# Amac: takim siniflandirma icin elle referans (ground truth) cikarmak.

VIDEO   = "data/smoke.mp4"
CIKTI   = "out/kirpinti_sayfa.jpg"
ORNEK   = 8      # iz basina kirpinti
HUCRE_H = 120    # her kirpinti bu yukseklige olceklenir
HUCRE_W = 60
MIN_KARE = 5     # daha kisa izler etiketlemeye degmez

kayit = json.load(open("out/smoke/izler.json"))
iz = defaultdict(list)
for k in kayit:
    iz[k["id"]].append(k)

secim = {}   # kare -> [(id, sutun, kutu)]
idler = sorted((t for t in iz if len(iz[t]) >= MIN_KARE),
               key=lambda t: -len(iz[t]))
for t in idler:
    kayitlar = iz[t]
    for s, i in enumerate(np.linspace(0, len(kayitlar) - 1, ORNEK).astype(int)):
        k = kayitlar[i]
        secim.setdefault(k["kare"], []).append((t, s, k["kutu"]))

ETIKET_W = 170
sayfa = np.full((HUCRE_H * len(idler), ETIKET_W + HUCRE_W * ORNEK, 3), 255, np.uint8)
satir = {t: r for r, t in enumerate(idler)}

cap = cv2.VideoCapture(VIDEO)
kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for t, s, (x1, y1, x2, y2) in secim.get(kare, []):
        kirp = frame[max(y1, 0):y2, max(x1, 0):x2]
        if kirp.size == 0:
            continue
        kirp = cv2.resize(kirp, (HUCRE_W, HUCRE_H))
        y0 = satir[t] * HUCRE_H
        x0 = ETIKET_W + s * HUCRE_W
        sayfa[y0:y0 + HUCRE_H, x0:x0 + HUCRE_W] = kirp
    kare += 1
cap.release()

for t, r in satir.items():
    y0 = r * HUCRE_H
    cv2.putText(sayfa, f"#{t}", (8, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(sayfa, f"{len(iz[t])} kare", (8, y0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.line(sayfa, (0, y0), (sayfa.shape[1], y0), (200, 200, 200), 1)

cv2.imwrite(CIKTI, sayfa)
print(f"{len(idler)} iz ({MIN_KARE}+ kare), atlanan: {len(iz) - len(idler)}")
print(f"kaydedildi -> {CIKTI}")
