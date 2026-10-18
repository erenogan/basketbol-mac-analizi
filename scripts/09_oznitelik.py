import cv2, json, numpy as np

# Her tespit icin govde bolgesinden HSV renk histogrami cikarir, diske yazar.
# Video bir kez okunur; kumeleme denemeleri sonra out/oznitelik.npz uzerinden yapilir.

VIDEO = "data/smoke.mp4"
CIKTI = "out/oznitelik.npz"

# Govde penceresi (kutuya gore oran): bas, sort, bacak ve kenardaki zemin disarida kalsin
Y0, Y1 = 0.18, 0.50
X0, X1 = 0.25, 0.75

# Zemin maskesi: acik bej parke dusuk doygunluk + yuksek parlaklik
ZEMIN_S_MAX, ZEMIN_V_MIN = 50, 150

H_BIN, S_BIN, V_BIN = 18, 3, 3   # 18x3x3 = 162 boyut

kayit = json.load(open("out/smoke/izler.json"))
kare_index = {}
for i, k in enumerate(kayit):
    kare_index.setdefault(k["kare"], []).append(i)

oz = np.zeros((len(kayit), H_BIN * S_BIN * V_BIN), np.float32)
gecerli = np.zeros(len(kayit), np.float32)   # maske sonrasi kalan piksel orani

cap = cv2.VideoCapture(VIDEO)
kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    for i in kare_index.get(kare, []):
        x1, y1, x2, y2 = kayit[i]["kutu"]
        w, h = x2 - x1, y2 - y1
        gx1, gx2 = int(x1 + w * X0), int(x1 + w * X1)
        gy1, gy2 = int(y1 + h * Y0), int(y1 + h * Y1)
        p = hsv[max(gy1, 0):gy2, max(gx1, 0):gx2].reshape(-1, 3)
        if len(p) == 0:
            continue
        zemin = (p[:, 1] < ZEMIN_S_MAX) & (p[:, 2] > ZEMIN_V_MIN)
        p = p[~zemin]
        gecerli[i] = len(p) / max(len(zemin), 1)
        if len(p) < 10:
            continue
        hist, _ = np.histogramdd(p.astype(np.float32),
                                 bins=(H_BIN, S_BIN, V_BIN),
                                 range=((0, 180), (0, 256), (0, 256)))
        oz[i] = hist.ravel() / len(p)
    kare += 1
cap.release()

np.savez(CIKTI, oz=oz, gecerli=gecerli)
bos = int((oz.sum(1) == 0).sum())
print(f"{len(kayit)} tespit, ozniteliksiz (bos/zemin): {bos}")
print(f"gecerli piksel orani medyan: {np.median(gecerli):.2f}")
print(f"kaydedildi -> {CIKTI}")
