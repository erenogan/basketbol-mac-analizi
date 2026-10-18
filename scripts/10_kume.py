import cv2, json, sys, numpy as np

# out/oznitelik.npz uzerinde KMeans; her kumeden rastgele kirpintilarla denetim sayfasi.
# Kullanim: python scripts/10_kume.py [k]

K      = int(sys.argv[1]) if len(sys.argv) > 1 else 3
VIDEO  = "data/smoke.mp4"
ORNEK  = 24
HW, HH = 50, 100

kayit = json.load(open("out/smoke/izler.json"))
oz = np.load("out/oznitelik.npz")["oz"]

# Hellinger: histogramlarda karekok alip Oklid -> buyuk binlerin baskinligini azaltir
X = np.sqrt(oz)
cv2.setRNGSeed(0)
kriter = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
_, etiket, merkez = cv2.kmeans(X.astype(np.float32), K, None, kriter, 10, cv2.KMEANS_PP_CENTERS)
etiket = etiket.ravel()

rng = np.random.default_rng(0)
secim = {}
for c in range(K):
    idx = np.flatnonzero(etiket == c)
    for s, i in enumerate(rng.choice(idx, min(ORNEK, len(idx)), replace=False)):
        secim.setdefault(kayit[i]["kare"], []).append((c, s, kayit[i]["kutu"]))

sayfa = np.full((HH * K, 120 + HW * ORNEK, 3), 255, np.uint8)
cap = cv2.VideoCapture(VIDEO)
kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for c, s, (x1, y1, x2, y2) in secim.get(kare, []):
        kirp = frame[max(y1, 0):y2, max(x1, 0):x2]
        if kirp.size:
            sayfa[c * HH:(c + 1) * HH, 120 + s * HW:120 + (s + 1) * HW] = cv2.resize(kirp, (HW, HH))
    kare += 1
cap.release()

for c in range(K):
    n = int((etiket == c).sum())
    cv2.putText(sayfa, f"k{c}", (8, c * HH + 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(sayfa, f"n={n}", (8, c * HH + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    print(f"kume {c}: {n} tespit")

cv2.imwrite(f"out/kume_k{K}.jpg", sayfa)
np.save(f"out/kume_k{K}.npy", etiket)
print(f"kaydedildi -> out/kume_k{K}.jpg")
