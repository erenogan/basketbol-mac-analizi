import json, sys, numpy as np
from collections import defaultdict
from pathlib import Path

# Takim bazli mesafe. Girdi: 14_yumusat (izler_saha.json, yumusatilmis metre koordinati).
# - canli oyun / olu top ayri (olu_top alani, 14'te)
# - toplam metre eksik tespit yuzunden dusuk kalir (kare basina ~4.2/5 oyuncu izleniyor)
#   -> asil olcu: oyuncu basina m/dk = mesafe / izlenen oyuncu-dakika. Varsayim:
#   izlenemeyen oyuncu da benzer hizda hareket ediyor.
# - hiz bolgeleri (yuruyus <2, tempo 2-4, kosu 4-6, sprint >6 m/s); hiz 1 sn'lik yer
#   degistirmeden, kare-kare adimdan degil (konum gurultusu bolge sinirinda titrer)
# - gurultu tabani: olu topta (oyuncular neredeyse duruyor) medyan hiz raporlanir;
#   yuruyus bolgesinin bir kismi bu gurultudur, keyfi olarak cikarilmaz.
# - gecerlilik: canli oyunda yakin (kutu >=90 px) / uzak (<90 px) oyuncu m/dk ayri.
#   dev'de uzak ~%10-12 yuksek, iki takimda ayni -> uzak sahanin cozunurluk
#   gurultusu (1 px ~15 cm) mesafeyi bu kadar sisiriyor; yakin deger gercege daha yakin.
# kullanim: python scripts/18_mesafe.py dev  -> out/dev/mesafe.json
AD  = sys.argv[1] if len(sys.argv) > 1 else "smoke"
OUT = Path("out") / AD
FPS = 30.0
BOLGE = [("yuruyus", 0, 2), ("tempo", 2, 4), ("kosu", 4, 6), ("sprint", 6, 99), ("belirsiz", -1, 0)]
MIN_YARIM = 3          # hiz penceresi en az +-3 kare; daha kisa: bolge "belirsiz"
TAKIMLAR = ("kirmizi", "lacivert")

kayit = json.load(open(OUT / "izler_saha.json"))
n_kare = max(k["kare"] for k in kayit) + 1
olu_kare = {k["kare"] for k in kayit if k["olu_top"]}
by = defaultdict(list)
for k in kayit:
    if k["takim"] in TAKIMLAR and -1 <= k["sx"] <= 29 and -1 <= k["sy"] <= 16:
        by[k["iz"]].append(k)

# m: mesafe, s: izlenen oyuncu-saniye, b: bolge mesafesi
ist = {t: {d: {"m": 0.0, "s": 0.0, "b": defaultdict(float)} for d in ("canli", "olu")}
       for t in TAKIMLAR}
olu_hiz = []
yu = defaultdict(lambda: [0.0, 0.0])          # (takim, yakin/uzak) -> [m, s], canli oyun
for ks in by.values():
    ks.sort(key=lambda k: k["kare"])
    t = ks[0]["takim"]
    f = np.array([k["kare"] for k in ks])
    p = np.array([[k["sx"], k["sy"]] for k in ks])
    idx = {fr: i for i, fr in enumerate(f)}
    boy = None
    for i in range(len(ks)):
        d = "olu" if f[i] in olu_kare else "canli"
        ist[t][d]["s"] += 1 / FPS
        if ks[i]["kutu"]:
            boy = ks[i]["kutu"][3] - ks[i]["kutu"][1]
        g = None if boy is None else ("uzak" if boy < 90 else "yakin")
        if d == "canli" and g:
            yu[(t, g)][1] += 1 / FPS
        if i == 0 or f[i] - f[i - 1] != 1:
            continue
        adim = float(np.linalg.norm(p[i] - p[i - 1]))
        # 1 sn'lik ortalanmis yer degistirme hizi; iz kenarinda iz icinde kalan en genis
        # ortalanmis pencere. Kare-kare adim x 30'a dusmek 20 cm titremeyi "sprint"
        # yapiyordu (olu topta kirmizi mesafenin %38'i sprint cikmisti).
        yarim = min(15, i, len(ks) - 1 - i)
        a, b = idx.get(f[i] - yarim), idx.get(f[i] + yarim)
        if yarim >= MIN_YARIM and a is not None and b is not None:
            hiz = float(np.linalg.norm(p[b] - p[a])) * FPS / (f[b] - f[a])
        else:
            hiz = -0.5                                   # belirsiz
        ist[t][d]["m"] += adim
        if d == "canli" and g:
            yu[(t, g)][0] += adim
        bolge = next(ad for ad, lo, hi in BOLGE if lo <= hiz < hi)
        ist[t][d]["b"][bolge] += adim
        if d == "olu" and hiz >= 0:
            olu_hiz.append(hiz)

sure_dk = {"canli": (n_kare - len(olu_kare)) / FPS / 60, "olu": len(olu_kare) / FPS / 60}
print(f"video {n_kare / FPS / 60:.1f} dk  (canli {sure_dk['canli']:.1f} dk, olu top {sure_dk['olu']:.1f} dk)")
print(f"gurultu tabani: olu topta medyan hiz {np.median(olu_hiz):.2f} m/s "
      f"(oyuncular neredeyse duruyor; yuruyus bolgesinin bir kismi gurultu)\n")
cikti = {"video_dk": round(n_kare / FPS / 60, 2), "gurultu_tabani_ms": round(float(np.median(olu_hiz)), 2),
         "takimlar": {}}
print(f"{'':10s}{'durum':7s}{'mesafe':>9s}{'izlenen':>9s}{'oyuncu/kare':>12s}{'m/dk/oyuncu':>12s}  bolgeler (mesafe payi)")
for t in TAKIMLAR:
    cikti["takimlar"][t] = {}
    for d in ("canli", "olu"):
        x = ist[t][d]
        oyuncu_dk = x["s"] / 60
        ort_oyuncu = oyuncu_dk / sure_dk[d] if sure_dk[d] else 0
        hiz = x["m"] / oyuncu_dk if oyuncu_dk else 0
        pay = {b: x["b"][b] / x["m"] * 100 if x["m"] else 0 for b, _, _ in BOLGE}
        print(f"{t:10s}{d:7s}{x['m']:8.0f}m{oyuncu_dk:8.1f}dk{ort_oyuncu:12.2f}{hiz:12.0f}  " +
              " ".join(f"{b} %{v:.0f}" for b, v in pay.items()))
        cikti["takimlar"][t][d] = {"mesafe_m": round(x["m"], 1), "oyuncu_dk": round(oyuncu_dk, 2),
                                   "ort_izlenen_oyuncu": round(ort_oyuncu, 2),
                                   "m_dk_oyuncu": round(hiz, 1),
                                   "bolge_m": {b: round(x["b"][b], 1) for b, _, _ in BOLGE}}
    # 5 oyunculu tam takim icin kestirim (eksik tespit duzeltilmis, varsayima dayali)
    tahmin = sum(cikti["takimlar"][t][d]["m_dk_oyuncu"] * 5 * sure_dk[d] for d in ("canli", "olu"))
    cikti["takimlar"][t]["tam_takim_tahmini_m"] = round(tahmin, 0)
    print(f"{'':10s}5 oyunculu tam takim tahmini: {tahmin:.0f} m  (izlenen: "
          f"{sum(ist[t][d]['m'] for d in ('canli', 'olu')):.0f} m)")
print("\ngecerlilik (canli oyun, m/dk/oyuncu): uzak sistematik yuksekse fark gurultudur")
for t in TAKIMLAR:
    v = {g: yu[(t, g)][0] / (yu[(t, g)][1] / 60) if yu[(t, g)][1] else 0 for g in ("yakin", "uzak")}
    cikti["takimlar"][t]["canli_m_dk_yakin_uzak"] = {g: round(x, 1) for g, x in v.items()}
    print(f"  {t:9s} yakin {v['yakin']:4.0f}   uzak {v['uzak']:4.0f}   "
          f"(uzak/yakin {v['uzak'] / v['yakin']:.2f})" if v["yakin"] else f"  {t}: yakin veri yok")
json.dump(cikti, open(OUT / "mesafe.json", "w"), indent=1)
print(f"\nkaydedildi -> {OUT / 'mesafe.json'}")
