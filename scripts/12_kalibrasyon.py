import cv2, json, sys, numpy as np
from pathlib import Path
from saha_model import cizgiler

# Saha kalibrasyonu: bilinen saha noktalarini tikla -> homography (piksel -> metre).
# Saha: FIBA 28 x 15 m. Orijin sol-uzak kose (reklam panolari tarafi),
# x saha boyunca saga (0-28), y kameraya dogru (0-15).
#
# kullanim: python scripts/12_kalibrasyon.py          -> tiklama arayuzu
#           python scripts/12_kalibrasyon.py kontrol  -> sadece olcum + cizim
VIDEO, KARE = "data/dev.mp4", 1385
OUT = Path("data/kalibrasyon.json")
SCALE = 0.75            # ekrana sigmasi icin
B = 5.8                 # serbest atis cizgisi dip cizgiden 5.8 m
NOKTALAR = [            # (ad, x_m, y_m) — gorunmeyeni 'n' ile atla
    ("sol uzak kose",                0.0,  0.0),
    ("sol yakin kose",               0.0, 15.0),
    ("sol 3 sayi duz cizgi uzak-dip", 0.0,  0.9),
    ("sol 3 sayi duz cizgi yakin-dip",0.0, 14.1),
    ("sol boyali alan uzak-dip",     0.0,  5.05),
    ("sol boyali alan yakin-dip",    0.0,  9.95),
    ("sol serbest atis uzak ucu",    B,    5.05),
    ("sol serbest atis yakin ucu",   B,    9.95),
    ("orta cizgi uzak ucu",         14.0,  0.0),
    ("orta cizgi yakin ucu",        14.0, 15.0),
    ("sag serbest atis uzak ucu",   28-B,  5.05),
    ("sag serbest atis yakin ucu",  28-B,  9.95),
    ("sag boyali alan uzak-dip",    28.0,  5.05),
    ("sag boyali alan yakin-dip",   28.0,  9.95),
    ("sag uzak kose",               28.0,  0.0),
    ("sag yakin kose",              28.0, 15.0),
]
# ipucu icin kaba homography: saha.json poligonunun 4 kosesi (dogrulandi, ~25 px hata)
KABA = {(0.0, 0.0): (240, 603), (28.0, 0.0): (1210, 521),
        (28.0, 15.0): (1743, 596), (0.0, 15.0): (253, 1035)}


def kare_oku():
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, KARE)
    ok, im = cap.read()
    if not ok:
        raise SystemExit(f"{VIDEO} kare {KARE} okunamadi")
    return im


def homografi(saha_pts, img_pts):
    # piksel -> metre. Tum noktalarla en kucuk kareler (RANSAC degil: elle
    # tiklanan az sayida nokta var, aykiri olani olcumle ayiklariz)
    H, _ = cv2.findHomography(np.float32(img_pts), np.float32(saha_pts), 0)
    return H


def donustur(H, pts):
    return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), H).reshape(-1, 2)


def olc(cift):
    ad    = [c["ad"] for c in cift]
    saha  = np.float32([c["saha"] for c in cift])
    img   = np.float32([c["piksel"] for c in cift])
    H = homografi(saha, img)
    fit = np.linalg.norm(donustur(H, img) - saha, axis=1)
    print(f"\n{len(cift)} nokta. Hata metre cinsinden:")
    print(f"{'nokta':32s} {'uyum':>6s} {'birini-disarida':>16s}")
    loo = []
    for i in range(len(cift)):
        m = np.arange(len(cift)) != i
        if m.sum() >= 4:
            Hi = homografi(saha[m], img[m])
            e = float(np.linalg.norm(donustur(Hi, img[i:i+1])[0] - saha[i]))
            loo.append(e)
        else:
            e = float("nan")
        # >3 m: o nokta olmadan kalanlar ayni cizgiye yigiliyor (geometri yetersiz)
        not_ = "  <- bu bolgede baska nokta yok" if e > 3 else ""
        print(f"{ad[i]:32s} {fit[i]:6.2f} {e:16.2f}{not_}")
    # uyum hatasi tek basina anlamsiz: 8 serbestlikli model az noktadan gecmek zorunda
    print(f"\nuyum hatasi     : ortalama {fit.mean():.2f} m, en kotu {fit.max():.2f} m")
    if loo:
        print(f"birini-disarida : medyan {np.median(loo):.2f} m, en kotu {np.max(loo):.2f} m"
              "   <- dogruluk icin bu sayi")
    return H


def saha_ciz(im, H_img2saha, renk=(0, 255, 0)):
    Hinv = np.linalg.inv(H_img2saha)
    for pts, kapali in cizgiler():
        p = donustur(Hinv, pts).astype(np.int32)
        cv2.polylines(im, [p], kapali, renk, 2, cv2.LINE_AA)


def kontrol():
    veri = json.loads(OUT.read_text())
    H = olc(veri["noktalar"])
    veri["H_piksel_saha"] = H.tolist()   # downstream: cv2.perspectiveTransform(ayak, H)
    OUT.write_text(json.dumps(veri, indent=1))
    im = kare_oku()
    saha_ciz(im, H)
    for c in veri["noktalar"]:
        cv2.circle(im, tuple(int(v) for v in c["piksel"]), 6, (255, 0, 255), 2)
    cv2.imwrite("out/kalibrasyon_kontrol.jpg", im)
    print("cizim -> out/kalibrasyon_kontrol.jpg (yesil model gercek cizgilere oturmali)")


def tikla_arayuz():
    im = kare_oku()
    Hk, _ = cv2.findHomography(np.float32(list(KABA.keys())), np.float32(list(KABA.values())))
    h, w = im.shape[:2]
    kucuk = cv2.resize(im, (int(w * SCALE), int(h * SCALE)))
    sira, secim, imlec = 0, [], [0, 0]

    def fare(ev, x, y, *_):
        imlec[:] = [x, y]
        if ev == cv2.EVENT_LBUTTONDOWN and sira < len(NOKTALAR):
            secim.append((sira, (x / SCALE, y / SCALE)))
            ilerle()
    def ilerle():
        nonlocal sira
        sira += 1

    cv2.namedWindow("kalibrasyon")
    cv2.setMouseCallback("kalibrasyon", fare)
    print("sol tik: noktayi isaretle | n: gorunmuyor, atla | u: geri al | s: kaydet | q: cik")
    while True:
        d = kucuk.copy()
        for _, (x, y) in secim:
            cv2.circle(d, (int(x * SCALE), int(y * SCALE)), 5, (255, 0, 255), -1)
        if sira < len(NOKTALAR):
            ad, xm, ym = NOKTALAR[sira]
            ip = donustur(Hk, [[xm, ym]])[0] * SCALE
            cv2.circle(d, (int(ip[0]), int(ip[1])), 14, (0, 255, 255), 2)   # tahmini yer
            yazi = f"{sira + 1}/{len(NOKTALAR)}: {ad}  ({xm:g}, {ym:g}) m"
        else:
            yazi = "bitti - s: kaydet"
        cv2.rectangle(d, (0, 0), (d.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(d, yazi, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        # buyutec: imlecin etrafi 4x
        cx, cy = int(imlec[0] / SCALE), int(imlec[1] / SCALE)
        y0, x0 = max(cy - 40, 0), max(cx - 40, 0)
        parca = im[y0:y0 + 80, x0:x0 + 80]
        if parca.shape[0] == 80 and parca.shape[1] == 80:
            b = cv2.resize(parca, (320, 320), interpolation=cv2.INTER_NEAREST)
            cv2.drawMarker(b, (160, 160), (0, 0, 255), cv2.MARKER_CROSS, 30, 1)
            d[d.shape[0] - 330:d.shape[0] - 10, d.shape[1] - 330:d.shape[1] - 10] = b
        cv2.imshow("kalibrasyon", d)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("n") and sira < len(NOKTALAR):
            ilerle()
        elif k == ord("u") and sira > 0:
            sira -= 1
            if secim and secim[-1][0] == sira:
                secim.pop()
        elif k == ord("s"):
            if len(secim) < 5:
                print("en az 5 nokta gerekli (4'u homography, fazlasi dogrulama)")
                continue
            cift = [{"ad": NOKTALAR[i][0], "saha": list(NOKTALAR[i][1:]),
                     "piksel": [round(x, 1), round(y, 1)]} for i, (x, y) in secim]
            OUT.write_text(json.dumps({"video": VIDEO, "kare": KARE,
                                       "noktalar": cift}, indent=1))
            print(f"kaydedildi -> {OUT}")
            break
        elif k == ord("q"):
            print("iptal"); cv2.destroyAllWindows(); return
    cv2.destroyAllWindows()
    kontrol()


if __name__ == "__main__":
    kontrol() if len(sys.argv) > 1 and sys.argv[1] == "kontrol" else tikla_arayuz()
