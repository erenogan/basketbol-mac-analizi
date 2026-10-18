import json, sys, numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# Takipci kapanmalarda kimligi rakip takima (veya hakeme) aktariyor.
# Izi, takim etiketinin kalici olarak degistigi karede boluyoruz.
# kullanim: python scripts/10_iz_bol.py dev  -> out/dev/izler_bolunmus.json
AD  = sys.argv[1] if len(sys.argv) > 1 else "smoke"
OUT = Path("out") / AD
MIN_PARCA = 10      # bundan kisa takim parcasi gurultu, oncekine eklenir

kayit = json.load(open(OUT / "izler_takim.json"))
iz_index = defaultdict(list)
for i, k in enumerate(kayit):
    iz_index[k["id"]].append(i)


def parcalar(idx):
    # [(etiket, [indeksler])]; belirsiz kareler bulundugu parcaya katilir
    out = []
    for i in idx:
        e = kayit[i]["takim"]
        if e == "belirsiz":
            if out:
                out[-1][1].append(i)
            else:
                out.append([None, [i]])     # iz belirsizle basliyor
            continue
        if out and out[-1][0] in (e, None):
            out[-1][0] = e
            out[-1][1].append(i)
        else:
            out.append([e, [i]])
    # kisa parcalari oncekine (yoksa sonrakine) kat, ayni etiketli komsulari birlestir
    temiz = []
    for e, ii in out:
        if temiz and (len(ii) < MIN_PARCA or temiz[-1][0] == e):
            temiz[-1][1].extend(ii)
        else:
            temiz.append([e, ii])
    if len(temiz) > 1 and len(temiz[0][1]) < MIN_PARCA:
        temiz[1][1][:0] = temiz[0][1]
        temiz.pop(0)
    return temiz


yeni_id = 0
kesim = []          # (eski id, kare, sicrama px)
normal = []         # bolunmeyen ardisik karelerdeki hareket (px/kare)
for tid in sorted(iz_index):
    idx = sorted(iz_index[tid], key=lambda i: kayit[i]["kare"])
    p = parcalar(idx)
    for j, (e, ii) in enumerate(p):
        yeni_id += 1
        for i in ii:
            kayit[i]["iz"] = yeni_id
            kayit[i]["iz_takim"] = e or "belirsiz"
        if j > 0:
            a, b = kayit[p[j - 1][1][-1]], kayit[ii[0]]
            dk = max(b["kare"] - a["kare"], 1)
            kesim.append((tid, b["kare"],
                          np.hypot(b["x"] - a["x"], b["y"] - a["y"]) / dk))
    for a, b in zip(idx, idx[1:]):
        if kayit[a]["iz"] == kayit[b]["iz"]:
            dk = max(kayit[b]["kare"] - kayit[a]["kare"], 1)
            normal.append(np.hypot(kayit[b]["x"] - kayit[a]["x"],
                                   kayit[b]["y"] - kayit[a]["y"]) / dk)

json.dump(kayit, open(OUT / "izler_bolunmus.json", "w"))

# --- rapor ---
n_kare = max(k["kare"] for k in kayit) + 1
print(f"iz sayisi : {len(iz_index)} -> {yeni_id}  ({len(kesim)} kesim)")

def kapsama(anahtar):
    c = Counter(k[anahtar] for k in kayit)
    return sum(v >= 300 for v in c.values()), max(c.values())
for ad, anahtar in (("once", "id"), ("sonra", "iz")):
    n300, en = kapsama(anahtar)
    print(f"  {ad:5s}: 300+ kare iz {n300}, en uzun {en} kare (%{en/n_kare*100:.0f})")

# Bagimsiz kontrol: gercek ID degisiminde ayak noktasi baska oyuncuya atlar.
normal = np.array(normal)
sicrama = np.array([s for _, _, s in kesim])
print("\nayak noktasi hareketi (px/kare):")
print(f"  normal ardisik kare : medyan {np.median(normal):5.1f}  p95 {np.percentile(normal, 95):5.1f}"
      f"  p99 {np.percentile(normal, 99):5.1f}")
if len(sicrama):
    p95 = np.percentile(normal, 95)
    print(f"  kesim noktasi       : medyan {np.median(sicrama):5.1f}  "
          f"normalin p95'inden buyuk: {(sicrama > p95).sum()}/{len(sicrama)}")

# Takim bazli piksel mesafesi: kesimden once/sonra (kesim sahte sicramayi siler)
def takim_mesafe(iz_anahtar, takim_anahtar):
    m = Counter()
    by = defaultdict(list)
    for k in kayit:
        by[k[iz_anahtar]].append(k)
    for ks in by.values():
        ks.sort(key=lambda k: k["kare"])
        for a, b in zip(ks, ks[1:]):
            m[b[takim_anahtar]] += np.hypot(b["x"] - a["x"], b["y"] - a["y"])
    return m
once  = takim_mesafe("id", "takim")
sonra = takim_mesafe("iz", "iz_takim")
print("\ntakim bazli toplam ayak hareketi (piksel, saha metresi degil):")
for t in ("kirmizi", "lacivert"):
    print(f"  {t:9s}: {once[t]:9.0f} -> {sonra[t]:9.0f}  ({(sonra[t]/once[t]-1)*100:+.1f}%)")
print(f"\nkaydedildi -> {OUT / 'izler_bolunmus.json'}")
