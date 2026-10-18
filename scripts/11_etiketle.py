import cv2, json, os, numpy as np

# Takim siniflandirma icin referans etiketleme.
# Tum tespitlerden RASTGELE (kumeden bagimsiz) ornek secer, tek tek gosterir.
#
# Tuslar:  k = kirmizi   l = lacivert   d = diger (hakem, antrenor, kenardaki)
#          b = belirsiz (iki kisi ic ice, tanimlanamiyor)
#          z = geri al    q = kaydet ve cik
#
# Kural: kutuda birden fazla kisi varsa, AYAGI kutunun alt-ortasinda olan kisiyi etiketle
# (saha koordinati o noktadan hesaplaniyor).
# Ilerleme her tusta diske yazilir; tekrar calistirinca kaldigi yerden devam eder.

VIDEO = "data/smoke.mp4"
CIKTI = "out/etiket.json"
N     = 300
TUSLAR = {ord("k"): "kirmizi", ord("l"): "lacivert", ord("d"): "diger", ord("b"): "belirsiz"}

kayit = json.load(open("out/smoke/izler.json"))
rng = np.random.default_rng(42)
ornek = sorted(rng.choice(len(kayit), N, replace=False).tolist(),
               key=lambda i: kayit[i]["kare"])

etiket = json.load(open(CIKTI)) if os.path.exists(CIKTI) else {}

# Secilen karelerden baglamli gorselleri bellege al (seek yerine tek gecis: mp4'te seek kare-dogru olmayabilir)
gerekli = {}
for i in ornek:
    gerekli.setdefault(kayit[i]["kare"], []).append(i)

gorsel = {}
cap = cv2.VideoCapture(VIDEO)
kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for i in gerekli.get(kare, []):
        x1, y1, x2, y2 = kayit[i]["kutu"]
        h = y2 - y1
        # yakin plan: kutunun kendisi, 4 kat
        yakin = frame[max(y1, 0):y2, max(x1, 0):x2]
        yakin = cv2.resize(yakin, None, fx=400 / max(h, 1), fy=400 / max(h, 1))
        # baglam: kutunun etrafi, kutu cizili
        m = h
        bx1, by1 = max(x1 - m, 0), max(y1 - m // 2, 0)
        bx2, by2 = min(x2 + m, frame.shape[1]), min(y2 + m // 2, frame.shape[0])
        bag = frame[by1:by2, bx1:bx2].copy()
        cv2.rectangle(bag, (x1 - bx1, y1 - by1), (x2 - bx1, y2 - by1), (0, 255, 0), 1)
        cv2.circle(bag, ((x1 + x2) // 2 - bx1, y2 - by1), 3, (0, 255, 0), -1)
        bag = cv2.resize(bag, None, fx=400 / bag.shape[0], fy=400 / bag.shape[0])
        gorsel[i] = np.hstack([yakin, np.zeros((400, 10, 3), np.uint8), bag])
    kare += 1
cap.release()

sira = 0
while sira < len(ornek) and str(ornek[sira]) in etiket:
    sira += 1

while sira < len(ornek):
    i = ornek[sira]
    img = gorsel[i].copy()
    cv2.putText(img, f"{sira + 1}/{N}  kare {kayit[i]['kare']}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "k kirmizi  l lacivert  d diger  b belirsiz  z geri  q cik",
                (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow("etiketle", img)
    t = cv2.waitKey(0) & 0xFF
    if t == ord("q"):
        break
    if t == ord("z") and sira > 0:
        sira -= 1
        etiket.pop(str(ornek[sira]), None)
    elif t in TUSLAR:
        etiket[str(i)] = TUSLAR[t]
        sira += 1
    else:
        continue
    json.dump(etiket, open(CIKTI, "w"), indent=0)

cv2.destroyAllWindows()
from collections import Counter
print(f"etiketlenen: {len(etiket)}/{N}")
print(dict(Counter(etiket.values())))
