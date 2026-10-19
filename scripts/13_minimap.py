import cv2, json, sys, shutil, subprocess, numpy as np
from collections import defaultdict, deque
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from saha_model import cizgiler, POTALAR, UZUN, EN
from ortak import video_yolu

# Teslim videosu: ustte bant (sure, CANLI/OLU TOP, takim mesafesi), solda kamera
# (takim kutulari + kalibre saha modeli), sagda kusbakisi tahta.
# Girdi: 14_yumusat ciktisi (izler_saha.json) — yumusatilmis metre koordinati.
# Tahtada dolu nokta = olcum, ici bos halka = kisa kopmada ara doldurma (tahmin).
# Bant metni Pillow ile: cv2.putText Turkce karakter (o, u, i, s) cizemiyor.
# Bantta oyuncu sayisi yok: fazla kutu (kenardaki yedek / cift kutu, ~0.12/kare)
# "6 oyuncu" gosterip yaniltiyordu; sinir LIMITASYONLAR.md'de.
# kullanim: python scripts/13_minimap.py dev
AD    = sys.argv[1] if len(sys.argv) > 1 else "smoke"
VIDEO = video_yolu(AD)
OUT   = Path("out") / AD
IZ_SN = 1.0                     # tahtada noktanin arkasindaki iz (saniye)
PPM, PAY = 40, 1.0              # tahta: piksel/metre, saha disi kenar payi (m)
SOL_W, SOL_H = 1280, 720        # kamera paneli (1920x1080 -> 2/3)
BANT = 64
TAKIMLAR = ("kirmizi", "lacivert")
RENK = {"kirmizi": (40, 40, 230), "lacivert": (230, 120, 20),
        "disarida": (150, 150, 150), "belirsiz": (230, 230, 230)}      # BGR
RGB = {t: tuple(reversed(c)) for t, c in RENK.items()}
FONT = "C:/Windows/Fonts/segoeui.ttf"
FONT_B = "C:/Windows/Fonts/segoeuib.ttf"

H = np.array(json.load(open("data/kalibrasyon.json"))["H_piksel_saha"])
Hinv = np.linalg.inv(H)
kayit = json.load(open(OUT / "izler_saha.json"))
kare_index = defaultdict(list)
for k in kayit:
    kare_index[k["kare"]].append(k)
n_kare = max(k["kare"] for k in kayit) + 1
olu_kare = {k["kare"] for k in kayit if k["olu_top"]}

# canli oyunda birikmis takim mesafesi; 18_mesafe ile ayni kurallar (saha +-1 m,
# ayni iz icinde ardisik kare) -> videodaki sayi mesafe.json ile tutarli
artis = {t: np.zeros(n_kare) for t in TAKIMLAR}
by = defaultdict(list)
for k in kayit:
    if k["takim"] in TAKIMLAR and -1 <= k["sx"] <= 29 and -1 <= k["sy"] <= 16:
        by[k["iz"]].append(k)
for ks in by.values():
    ks.sort(key=lambda k: k["kare"])
    for a, b in zip(ks, ks[1:]):
        if b["kare"] - a["kare"] == 1 and b["kare"] not in olu_kare:
            artis[b["takim"]][b["kare"]] += np.hypot(b["sx"] - a["sx"], b["sy"] - a["sy"])
birikmis = {t: np.cumsum(v) for t, v in artis.items()}

TW = int((UZUN + 2 * PAY) * PPM)
oy = (SOL_H - int((EN + 2 * PAY) * PPM)) // 2       # tahtayi dikeyde ortala
def m2p(x, y):
    return int((x + PAY) * PPM), int((y + PAY) * PPM) + oy

tahta0 = np.full((SOL_H, TW, 3), (60, 95, 150), np.uint8)    # parke rengi
cv2.rectangle(tahta0, m2p(0, 0), m2p(UZUN, EN), (75, 120, 185), -1)
for pts, kapali in cizgiler():
    p = np.array([m2p(x, y) for x, y in pts], np.int32)
    cv2.polylines(tahta0, [p], kapali, (255, 255, 255), 2, cv2.LINE_AA)
for px, py in POTALAR:
    cv2.circle(tahta0, m2p(px, py), 6, (0, 140, 255), 2)
# tahta aciklamasi (sabit, bir kez)
img = Image.fromarray(cv2.cvtColor(tahta0, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(img)
f_kucuk = ImageFont.truetype(FONT, 20)
ay = SOL_H - 34
d.ellipse((24, ay + 4, 40, ay + 20), fill=(235, 235, 235), outline=(0, 0, 0))
d.text((48, ay), "ölçülen konum", font=f_kucuk, fill=(245, 245, 245))
d.ellipse((214, ay + 4, 230, ay + 20), outline=(235, 235, 235), width=3)
d.text((238, ay), "tahmin (≤0,33 sn kopukluk)", font=f_kucuk, fill=(245, 245, 245))
d.text((TW - 300, ay), "kuşbakışı · FIBA 28 × 15 m", font=f_kucuk, fill=(245, 245, 245))
tahta0 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# kamera paneli icin saha modeli overlay (sabit, bir kez hazirla)
overlay = []
for pts, kapali in cizgiler():
    p = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    overlay.append((np.int32(p * SOL_W / 1920), kapali))

f_b, f_n = ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 24)
W_TOP = SOL_W + TW


def bant(kare):
    im = Image.new("RGB", (W_TOP, BANT), (24, 28, 27)); d = ImageDraw.Draw(im)
    sn = kare / fps
    d.text((20, 13), f"{int(sn // 60):02d}:{sn % 60:04.1f}", font=f_b, fill=(240, 240, 240))
    olu = kare in olu_kare
    etiket, renk = ("ÖLÜ TOP", (214, 160, 60)) if olu else ("CANLI", (80, 170, 120))
    w = d.textlength(etiket, font=f_n) + 28
    d.rounded_rectangle((150, 15, 150 + w, 49), radius=17, fill=renk)
    d.text((164, 17), etiket, font=f_n, fill=(20, 20, 20))
    x = W_TOP - 20
    for t, ad in (("lacivert", "Lacivert"), ("kirmizi", "Kırmızı")):
        yazi = f"{ad}  {birikmis[t][kare]:,.0f} m".replace(",", ".")
        x -= d.textlength(yazi, font=f_n)
        d.text((x, 17), yazi, font=f_n, fill=(240, 240, 240))
        d.rectangle((x - 26, 22, x - 10, 38), fill=RGB[t])
        x -= 60
    d.text((x - d.textlength("canlı oyun mesafesi →", font=f_n) + 20, 17),
           "canlı oyun mesafesi →", font=f_n, fill=(150, 160, 156))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
ham = str(OUT / "minimap.mp4")
yazici = cv2.VideoWriter(ham, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W_TOP, SOL_H + BANT))
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
        r = 9 if t in TAKIMLAR else 6
        if iz_id in tahmin:
            cv2.circle(tahta, pts[-1], r, RENK[t], 2, cv2.LINE_AA)
        else:
            cv2.circle(tahta, pts[-1], r, RENK[t], -1, cv2.LINE_AA)
            cv2.circle(tahta, pts[-1], r, (0, 0, 0), 1, cv2.LINE_AA)
    yazici.write(np.vstack([bant(kare), np.hstack([sol, tahta])]))
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
