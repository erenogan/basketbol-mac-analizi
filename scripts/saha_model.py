import numpy as np

# FIBA saha cizgileri, metre. Orijin sol-uzak kose (panolar tarafi),
# x boyuna 0-28, y kameraya dogru 0-15. 12_kalibrasyon ve 13_minimap ortak kullanir.
UZUN, EN = 28.0, 15.0
SERBEST_ATIS = 5.8      # dip cizgiden
POTA = 1.575            # pota merkezi dip cizgiden
UC_SAYI = 6.75


def _yay(cx, cy, r, a0, a1, n=80):
    t = np.radians(np.linspace(a0, a1, n))
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def cizgiler():
    """[(Nx2 nokta dizisi, kapali_mi)] — tum saha cizgileri metre cinsinden."""
    B, out = SERBEST_ATIS, []
    out.append((np.array([[0, 0], [UZUN, 0], [UZUN, EN], [0, EN]]), True))
    out.append((np.array([[14, 0], [14, EN]]), False))
    out.append((_yay(14, 7.5, 1.8, 0, 360), False))
    a = np.degrees(np.arcsin(6.6 / UC_SAYI))      # 3 sayi yayi kenardan 0.9 m'de duzlesir
    for x0, s in ((0, 1), (UZUN, -1)):
        out.append((np.array([[x0, 5.05], [x0 + s * B, 5.05],
                              [x0 + s * B, 9.95], [x0, 9.95]]), False))
        bx = x0 + s * POTA
        xe = bx + s * UC_SAYI * np.cos(np.radians(a))
        out.append((np.array([[x0, 0.9], [xe, 0.9]]), False))
        out.append((np.array([[x0, 14.1], [xe, 14.1]]), False))
        if s == 1:
            out.append((_yay(x0 + B, 7.5, 1.8, -90, 90), False))
            out.append((_yay(bx, 7.5, UC_SAYI, -a, a), False))
        else:
            out.append((_yay(x0 - B, 7.5, 1.8, 90, 270), False))
            out.append((_yay(bx, 7.5, UC_SAYI, 180 - a, 180 + a), False))
    return out


POTALAR = [(POTA, 7.5), (UZUN - POTA, 7.5)]
