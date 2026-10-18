import cv2, numpy as np

# Forma rengi piksel kurallari (08_takim'de olculup dogrulandi; bkz. CLAUDE.md).
# OpenCV HSV: H 0-180, S/V 0-255.


def kirmizi_maske(H, S, V):
    return ((H < 2) | (H > 165)) & (S >= 120) & (V >= 60)   # H 2-10 hakemin turuncusu


def lacivert_maske(H, S, V):
    return (H >= 100) & (H <= 160) & (S >= 70) & (V >= 30) & (V <= 140)


def govde(frame, kutu):
    # ust %15-%45, ortadaki %50: kafa, bacak, zemin disarida
    x1, y1, x2, y2 = (int(v) for v in kutu)
    w, h = x2 - x1, y2 - y1
    g = frame[max(y1 + int(h * .15), 0):y1 + int(h * .45),
              max(x1 + int(w * .25), 0):x2 - int(w * .25)]
    return g if g.size else None


def oranlar(g):
    """(kirmizi orani, lacivert orani, koyu orani) — govde kirpintisi icin."""
    H, S, V = cv2.cvtColor(g, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int).T
    return kirmizi_maske(H, S, V).mean(), lacivert_maske(H, S, V).mean(), (V < 35).mean()


# Takipte kullanilan "guclu" etiket: sadece cok emin olunan tespit eslesmeyi engeller.
# Esikler 08'in siniflandirma esiklerinden siki: yanlis engel = kopan iz.
GUCLU_KIRMIZI = (0.35, 0.05)    # kirmizi >= 0.35 ve lacivert < 0.05
GUCLU_LACIVERT = (0.15, 0.15)   # lacivert >= 0.15 ve kirmizi < 0.15
GUCLU_DISARIDA = (0.25, 0.05)   # koyu (V<35) >= 0.25 ve lacivert < 0.05: hakem / siyah giyen
DISARIDA, BILINMIYOR = 2, 3


def guclu_takim(frame, kutu):
    g = govde(frame, kutu)
    if g is None:
        return BILINMIYOR
    r, n, koyu = oranlar(g)
    if koyu >= GUCLU_DISARIDA[0] and n < GUCLU_DISARIDA[1]:
        return DISARIDA
    if r >= GUCLU_KIRMIZI[0] and n < GUCLU_KIRMIZI[1]:
        return 0
    if n >= GUCLU_LACIVERT[0] and r < GUCLU_LACIVERT[1]:
        return 1
    return BILINMIYOR
