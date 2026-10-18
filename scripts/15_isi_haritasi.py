import json, sys, numpy as np
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from saha_model import cizgiler, POTALAR, UZUN, EN, POTA, UC_SAYI

# Takim bazli isi haritasi + bolge dagilimi. Girdi: 14_yumusat (izler_saha.json).
# Sadece canli oyun: olu top (mola, serbest atis dizilisi) haric. Haric tutulmazsa
# dev'de 17 sn'lik bir bekleme haritanin en koyu iki lekesini tek basina uretiyordu.
# Birim: takim suresinin m^2 basina yuzdesi. Her takim kendi toplamina normalize:
# lacivert kare basina daha az tespit ediliyor (4.0 vs 4.3); mutlak sure kullansak
# fark oyundan degil tespitten gelirdi.
# kullanim: python scripts/15_isi_haritasi.py dev
AD   = sys.argv[1] if len(sys.argv) > 1 else "smoke"
OUT  = Path("out") / AD
HUCRE = 0.25           # m
SIGMA = 0.75           # m — uzak sahadaki kalibrasyon hatasi ~0.7 m; daha ince detay gurultu
TAKIMLAR = ("kirmizi", "lacivert")
# tek renkli sirali rampalar (acik -> koyu); mavi dataviz referans rampasi,
# kirmizi ayni acikliklarla takim rengine
RAMPA = {"lacivert": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
         "kirmizi":  ["#fbd5d4", "#f2a09f", "#e34948", "#a92b2b", "#6b1414"]}
YUZEY, CIZGI, METIN = "#f4f3ef", "#5c5c5c", "#2b2b2b"

kayit = json.load(open(OUT / "izler_saha.json"))
fps = 30.0
nx, ny = int(UZUN / HUCRE), int(EN / HUCRE)


def bolge(x, y):
    yari = "sol" if x < UZUN / 2 else "sag"
    dx = x if yari == "sol" else UZUN - x               # dip cizgiden uzaklik
    if dx <= 5.8 and 5.05 <= y <= 9.95:
        return yari, "boyali alan"
    if dx <= 2.99:                                       # 3 sayinin duz kismi
        ic = 0.9 <= y <= 14.1
    else:
        ic = np.hypot(dx - POTA, y - 7.5) <= UC_SAYI
    return yari, "3 sayi ici" if ic else "3 sayi disi"


yogunluk, bolge_sure, toplam, n_tahmin = {}, defaultdict(lambda: defaultdict(float)), {}, {}
for t in TAKIMLAR:
    tum = [k for k in kayit if k["takim"] == t
           and 0 <= k["sx"] <= UZUN and 0 <= k["sy"] <= EN]
    ks = [k for k in tum if not k.get("olu_top")]
    print(f"{t}: olu top haric {len(tum) - len(ks)} kayit (%{(len(tum)-len(ks))/len(tum)*100:.0f})")
    x = np.array([k["sx"] for k in ks]); y = np.array([k["sy"] for k in ks])
    h, _, _ = np.histogram2d(y, x, bins=[ny, nx], range=[[0, EN], [0, UZUN]])
    h = gaussian_filter(h, SIGMA / HUCRE, mode="constant")
    yogunluk[t] = h / h.sum() * 100 / (HUCRE * HUCRE)    # % / m^2
    toplam[t] = len(ks) / fps
    n_tahmin[t] = sum(k["ara_doldurma"] for k in ks) / max(len(ks), 1)
    for k in ks:
        yari, b = bolge(k["sx"], k["sy"])
        bolge_sure[t][(yari, b)] += 1
    for key in bolge_sure[t]:
        bolge_sure[t][key] *= 100 / len(ks)

# --- cizim: iki takim ayni olcekte ---
vmax = max(np.percentile(yogunluk[t], 99.5) for t in TAKIMLAR)
fig, eksen = plt.subplots(2, 1, figsize=(11, 12.5), facecolor="white")
for ax, t in zip(eksen, TAKIMLAR):
    ax.set_facecolor(YUZEY)
    cmap = LinearSegmentedColormap.from_list(t, RAMPA[t])
    cmap.set_under(YUZEY)                                # ~sifir: zemin gorunsun
    im = ax.imshow(yogunluk[t], extent=[0, UZUN, EN, 0], cmap=cmap,
                   vmin=vmax * 0.04, vmax=vmax, interpolation="bilinear")
    for pts, kapali in cizgiler():
        p = np.vstack([pts, pts[:1]]) if kapali else pts
        ax.plot(p[:, 0], p[:, 1], color=CIZGI, lw=1.2)
    for px, py in POTALAR:
        ax.add_patch(plt.Circle((px, py), 0.23, fill=False, color=CIZGI, lw=1.2))
    ax.set_xlim(-0.5, UZUN + 0.5); ax.set_ylim(EN + 0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ad = "Kırmızı" if t == "kirmizi" else "Lacivert"
    ax.set_title(f"{ad} — canlı oyun, {toplam[t]:.0f} oyuncu-saniye", loc="left",
                 color=METIN, fontsize=13, pad=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, extend="min")
    cb.set_label("takım süresinin %'si / m²", color=METIN)
    cb.outline.set_visible(False)
fig.text(0.02, 0.01,
         f"{AD}.mp4 · σ={SIGMA} m yumuşatma (≈ uzak saha kalibrasyon hatası) · "
         "ölü top hariç · kamera paneller tarafı yukarıda · konum kaybı kalabalık bölgelerde daha sık",
         color="#6b6b6b", fontsize=9)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(OUT / "isi_haritasi.png", dpi=110)

# --- bolge tablosu ---
satirlar = [(y, b) for y in ("sol", "sag") for b in ("boyali alan", "3 sayi ici", "3 sayi disi")]
print(f"{'bolge':24s}" + "".join(f"{t:>10s}" for t in TAKIMLAR))
for key in satirlar:
    print(f"{key[0] + ' yari, ' + key[1]:24s}" +
          "".join(f"{bolge_sure[t][key]:9.1f}%" for t in TAKIMLAR))
for yari in ("sol", "sag"):
    print(f"{yari + ' yari toplam':24s}" + "".join(
        f"{sum(v for k, v in bolge_sure[t].items() if k[0] == yari):9.1f}%" for t in TAKIMLAR))
print(f"{'ara doldurma payi':24s}" + "".join(f"{n_tahmin[t]*100:9.1f}%" for t in TAKIMLAR))
json.dump({t: {f"{k[0]}|{k[1]}": round(v, 2) for k, v in bolge_sure[t].items()}
           for t in TAKIMLAR}, open(OUT / "bolge_dagilimi.json", "w"), indent=1, ensure_ascii=False)
print(f"\nkaydedildi -> {OUT / 'isi_haritasi.png'}, {OUT / 'bolge_dagilimi.json'}")
