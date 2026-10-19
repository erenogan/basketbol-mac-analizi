import csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path

# Teslim paketi: out/<ad>/teslim/
#   tracks.json  kare kare oyuncu konumlari (metre) + kendi aciklamasi
#   tracks.csv   ayni veri duz tablo (Excel)
#   ozet.json    takim mesafesi, bolge dagilimi, olcum dogrulugu
#   isi_haritasi.png, tracked.mp4 (varsa), LIMITASYONLAR.md
# Girdi: 14_yumusat (izler_saha.json), 18_mesafe (mesafe.json), 15 (bolge_dagilimi.json)
# kullanim: python scripts/19_teslim.py dev
AD  = sys.argv[1] if len(sys.argv) > 1 else "smoke"
OUT = Path("out") / AD
HEDEF = OUT / "teslim"
HEDEF.mkdir(exist_ok=True)
FPS = 30.0

# Olculen dogruluk (bkz. CLAUDE.md). final.mp4: kurallar ona hic dokunmadan olculdu.
OLCUM = {
    "takim_dogrulugu": {"deger": 0.93, "yakin_saha": 1.00, "uzak_saha": 0.89,
                        "kaynak": "final.mp4 (2. devre), kor referans, 73 tespit"},
    "konum_hatasi_m": {"medyan": 0.16, "en_kotu": 0.73,
                       "kaynak": "15 noktali kalibrasyon, birini-disarida"},
    "oyuncu_bulunma": {"yakin_saha": 0.97, "uzak_saha": 0.81,
                       "kaynak": "final.mp4, 16 kare, 104 kesin oyuncu"},
}

kayit = json.load(open(OUT / "izler_saha.json"))
meta = json.load(open(OUT / "meta.json")) if (OUT / "meta.json").exists() else {}
n_kare = max(k["kare"] for k in kayit) + 1

# olu top araliklari
olu = sorted({k["kare"] for k in kayit if k["olu_top"]})
araliklar, bas = [], None
for f in olu + [None]:
    if bas is None:
        bas = son = f
    elif f is not None and f == son + 1:
        son = f
    else:
        araliklar.append([bas, son])
        bas = son = f

kareler = defaultdict(list)
for k in kayit:
    kareler[k["kare"]].append(k)

sema = {
    "sema_surumu": 1,
    "video": Path(meta.get("video", f"data/{AD}.mp4")).name,
    "fps": FPS,
    "kare_sayisi": n_kare,
    "saha": {
        "standart": "FIBA, 28 x 15 m",
        "orijin": "kameranin karsisindaki uzun kenar ile sol dip cizginin kesistigi kose",
        "x": "saha boyunca saga dogru, 0-28 m",
        "y": "kameraya dogru, 0-15 m",
        "konum_noktasi": "oyuncunun ayak noktasi (kutunun alt-orta noktasi)",
    },
    "alanlar": {
        "iz": "takip parcasi numarasi. OYUNCU KIMLIGI DEGILDIR: takip kapanmalarda kopar "
              "ve oyuncu yeni bir numarayla devam eder. Oyuncu bazli analizde kullanmayin.",
        "takim": "kirmizi | lacivert | disarida (hakem, kenardaki kisi) | belirsiz",
        "x, y": "saha koordinati, metre (yumusatilmis)",
        "tahmin": "true ise kare tespit edilmedi; en fazla 0.33 sn'lik kopuklukta "
                  "dogrusal ara doldurma",
        "kutu": "piksel cinsinden [x1, y1, x2, y2]; tahmin karelerinde null",
        "olu_top": "kare olu top araliginda mi (sezgisel: tum oyuncular yavas)",
    },
    "olcum": OLCUM,
    "uretim": {k: meta[k] for k in ("model", "imgsz", "tracker", "takim_takip") if k in meta},
    "olu_top_araliklari": araliklar,
    "kareler": [
        {"kare": f, "zaman_sn": round(f / FPS, 3), "olu_top": f in set(olu),
         "oyuncular": [{"iz": k["iz"], "takim": k["takim"], "x": k["sx"], "y": k["sy"],
                        "tahmin": k["ara_doldurma"], "kutu": k["kutu"]}
                       for k in sorted(kareler.get(f, []), key=lambda k: k["iz"])]}
        for f in range(n_kare)
    ],
}
json.dump(sema, open(HEDEF / "tracks.json", "w", encoding="utf-8"), ensure_ascii=False)

with open(HEDEF / "tracks.csv", "w", newline="", encoding="utf-8-sig") as f:   # BOM: Excel Turkce
    w = csv.writer(f, delimiter=";")                                           # ; Turkce Excel
    w.writerow(["kare", "zaman_sn", "iz", "takim", "x_m", "y_m", "tahmin", "olu_top",
                "kutu_x1", "kutu_y1", "kutu_x2", "kutu_y2"])
    for k in sorted(kayit, key=lambda k: (k["kare"], k["iz"])):
        b = k["kutu"] or ["", "", "", ""]
        w.writerow([k["kare"], f"{k['kare'] / FPS:.3f}".replace(".", ","), k["iz"], k["takim"],
                    f"{k['sx']:.2f}".replace(".", ","), f"{k['sy']:.2f}".replace(".", ","),
                    int(k["ara_doldurma"]), int(k["olu_top"]), *b])

ozet = {"video": sema["video"], "sure_dk": round(n_kare / FPS / 60, 2), "olcum": OLCUM}
for ad, dosya in (("mesafe", "mesafe.json"), ("bolge_dagilimi", "bolge_dagilimi.json")):
    if (OUT / dosya).exists():
        ozet[ad] = json.load(open(OUT / dosya, encoding="utf-8"))
json.dump(ozet, open(HEDEF / "ozet.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

for kaynak, hedef in (("isi_haritasi.png", "isi_haritasi.png"), ("minimap_izle.mp4", "tracked.mp4")):
    if (OUT / kaynak).exists():
        shutil.copy2(OUT / kaynak, HEDEF / hedef)
shutil.copy2("LIMITASYONLAR.md", HEDEF / "LIMITASYONLAR.md")   # sinirlar paketle birlikte gitsin

for p in sorted(HEDEF.iterdir()):
    print(f"  {p.name:20s} {p.stat().st_size / 1024 / 1024:7.2f} MB")
print(f"kaydedildi -> {HEDEF}")
