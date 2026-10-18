import cv2, json, sys, numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.signal import medfilt, savgol_filter

# Ayak noktasi -> saha metresi, iz boyunca yumusatma.
# Kutu alt kenari kareden kareye titriyor (~9 cm) ve ara ara sicriyor (kutu
# birlesme/bolunme). Offline isliyoruz -> ortalanmis (gelecege bakan) pencere serbest.
#   1) <=MAX_BOSLUK karelik kopmalari dogrusal doldur (isaretli: ara_doldurma)
#   2) medyan filtre: tek tuk sicramalari siler (durtu gurultusu)
#   3) Savitzky-Golay: titremeyi alir, donus/hizlanmada duz ortalamadan az kirpar
# Iz anahtari "iz" (10_iz_bol): takim degisiminde bolunmus iz. "id" ile yumusatsak
# kimlik rakibe gectiginde iki oyuncunun arasinda hayalet nokta cizilirdi.
# kullanim: python scripts/14_yumusat.py dev  -> out/dev/izler_saha.json
AD  = sys.argv[1] if len(sys.argv) > 1 else "smoke"
OUT = Path("out") / AD
MAX_BOSLUK = 10        # 0.33 sn; daha uzun kopma izi ayirir, uydurma yapmayiz
MEDYAN = 7
SG_PENCERE, SG_DERECE = 15, 2
FPS = 30.0

H = np.array(json.load(open("data/kalibrasyon.json"))["H_piksel_saha"])
kayit = json.load(open(OUT / "izler_bolunmus.json"))
S = cv2.perspectiveTransform(np.float32([[k["x"], k["y"]] for k in kayit])
                             .reshape(-1, 1, 2), H).reshape(-1, 2)
for k, (sx, sy) in zip(kayit, S):
    k["ham_sx"], k["ham_sy"] = float(sx), float(sy)

by = defaultdict(list)
for k in kayit:
    by[k["iz"]].append(k)


def tek_tek(n):
    return n if n % 2 else n - 1


def yumusat(xy):
    n = len(xy)
    if n < MEDYAN:
        return xy.copy()
    out = np.c_[medfilt(xy[:, 0], MEDYAN), medfilt(xy[:, 1], MEDYAN)]
    # medfilt kenarlari sifirla dolduruyor -> ilk/son yarim pencereyi ham birak
    y = MEDYAN // 2
    out[:y], out[-y:] = xy[:y], xy[-y:]
    w = min(SG_PENCERE, tek_tek(n))
    if w > SG_DERECE + 1:
        out = np.c_[savgol_filter(out[:, 0], w, SG_DERECE),
                    savgol_filter(out[:, 1], w, SG_DERECE)]
    return out


cikti = []
for iz, ks in by.items():
    ks.sort(key=lambda k: k["kare"])
    # kopmalarda parcala: >MAX_BOSLUK ise ayri parca, degilse doldur
    parcalar, cur = [], [ks[0]]
    for a, b in zip(ks, ks[1:]):
        if b["kare"] - a["kare"] - 1 > MAX_BOSLUK:
            parcalar.append(cur); cur = []
        cur.append(b)
    parcalar.append(cur)
    for p in parcalar:
        f = np.array([k["kare"] for k in p])
        tam = np.arange(f[0], f[-1] + 1)
        xy = np.c_[np.interp(tam, f, [k["ham_sx"] for k in p]),
                   np.interp(tam, f, [k["ham_sy"] for k in p])]
        ys = yumusat(xy)
        gercek = {k["kare"]: k for k in p}
        for fr, (sx, sy), (hx, hy) in zip(tam, ys, xy):
            g = gercek.get(int(fr))
            cikti.append({"kare": int(fr), "id": p[0]["id"], "iz": iz,
                          "takim": p[0]["iz_takim"],
                          "sx": round(float(sx), 3), "sy": round(float(sy), 3),
                          "ham_sx": round(float(hx), 3), "ham_sy": round(float(hy), 3),
                          "ara_doldurma": g is None,
                          "kutu": g["kutu"] if g else None})
cikti.sort(key=lambda k: (k["kare"], k["iz"]))

# Olu top (mola, serbest atis dizilisi, degisiklik): top tespit edilemiyor ama
# olu topta herkes yavaslar. Sezgisel kural: oyuncularin medyan hizi 2 sn
# ortalamada < OLU_HIZ ve bu en az OLU_SURE surerse olu top. dev'de iki kume
# arasindaki bosluktan secildi (olu ~0.6-0.8, canli >=1.25 m/s); 2/2 aralik gozle
# dogrulandi (bekleme + serbest atis), sinirdaki canli an (kare 3342) dogru kaldi.
OLU_HIZ, OLU_SURE = 1.0, int(3 * FPS)
from scipy.ndimage import uniform_filter1d
konum = defaultdict(dict)
for k in cikti:
    if k["takim"] in ("kirmizi", "lacivert"):
        konum[k["iz"]][k["kare"]] = (k["sx"], k["sy"])
n_kare = max(k["kare"] for k in cikti) + 1
hiz = defaultdict(list)
for d in konum.values():
    for f, (x, y) in d.items():
        q = d.get(f - 5)                                 # 1/6 sn yer degistirme
        if q:
            hiz[f].append(np.hypot(x - q[0], y - q[1]) * FPS / 5)
med = np.array([np.median(hiz[f]) if len(hiz[f]) >= 4 else np.nan for f in range(n_kare)])
med = uniform_filter1d(np.nan_to_num(med, nan=np.nanmedian(med)), int(2 * FPS))
olu = np.zeros(n_kare, bool)
bas = None
for f, v in enumerate(np.r_[med < OLU_HIZ, False]):
    if v and bas is None:
        bas = f
    elif not v and bas is not None:
        if f - bas >= OLU_SURE:
            olu[bas:f] = True
        bas = None
for k in cikti:
    k["olu_top"] = bool(olu[k["kare"]])
json.dump(cikti, open(OUT / "izler_saha.json", "w"))


# --- olcum: once (ham) / sonra (yumusak) ---
def metrikler(anahtar_x, anahtar_y, sadece_gercek):
    adim, sicrama, n_adim, d1s, mesafe = [], 0, 0, [], defaultdict(float)
    izler = defaultdict(list)
    for k in cikti:
        if sadece_gercek and k["ara_doldurma"]:
            continue
        izler[k["iz"]].append(k)
    for ks in izler.values():
        f = np.array([k["kare"] for k in ks])
        p = np.array([[k[anahtar_x], k[anahtar_y]] for k in ks])
        ard = np.diff(f) == 1
        st = np.linalg.norm(np.diff(p, axis=0), axis=1)[ard]
        adim += st.tolist(); sicrama += int((st > 1.0).sum()); n_adim += int(ard.sum())
        t = ks[0]["takim"]
        if t in ("kirmizi", "lacivert"):
            mesafe[t] += float(st.sum())
        # 1 sn'lik yer degistirme: gercek hareketin olcusu (titremeden etkilenmez)
        idx = {fr: i for i, fr in enumerate(f)}
        for i, fr in enumerate(f):
            j = idx.get(fr + 30)
            if j is not None:
                d1s.append(np.linalg.norm(p[j] - p[i]))
    return np.array(adim), sicrama, n_adim, np.array(d1s), mesafe

h = metrikler("ham_sx", "ham_sy", True)
y = metrikler("sx", "sy", False)
n_doldur = sum(k["ara_doldurma"] for k in cikti)
print(f"iz: {len(by)} | kayit: {len(kayit)} gercek + {n_doldur} ara doldurma "
      f"(%{n_doldur/len(cikti)*100:.1f})")
print(f"{'':28s}{'ham':>12s}{'yumusak':>12s}")
print(f"{'kare-kare adim medyan (cm)':28s}{np.median(h[0])*100:12.1f}{np.median(y[0])*100:12.1f}")
print(f"{'hiz p99 (m/s)':28s}{np.percentile(h[0],99)*FPS:12.1f}{np.percentile(y[0],99)*FPS:12.1f}"
      "   <- kosu en fazla ~8-9 m/s")
print(f"{'imkansiz sicrama >1m/kare':28s}{h[1]:12d}{y[1]:12d}")
print(f"{'1 sn yer degistirme medyan(m)':28s}{np.median(h[3]):12.2f}{np.median(y[3]):12.2f}"
      "   <- gercek hareket korunmali")
for t in ("kirmizi", "lacivert"):
    print(f"{'mesafe ' + t + ' (m)':28s}{h[4][t]:12.0f}{y[4][t]:12.0f}")
ar, bas = [], None
for f, v in enumerate(np.r_[olu, False]):
    if v and bas is None:
        bas = f
    elif not v and bas is not None:
        ar.append(f"{bas}-{f - 1}"); bas = None
print(f"\nolu top: {olu.sum() / FPS:.1f} sn / {n_kare / FPS:.0f} sn  araliklar: {ar}")
print(f"kaydedildi -> {OUT / 'izler_saha.json'}")
