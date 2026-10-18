import json, sys, numpy as np
from collections import defaultdict, Counter
from pathlib import Path

# Deney karsilastirmasi: hattin sonunda (takip edilen, saha ici kutular) oyuncu recall'u.
# 1) data/tespit_ref_<video>.json: 16 karede KESIN oyuncular -> IoU>=0.5 izle eslesen orani
#    (kolay ornege yanli; yanlis pozitiften etkilenmez)
# 2) tum video: kare basina min(kirmizi,5)+min(lacivert,5) (etiketsiz; sinir, fazladan
#    kutunun eksik oyuncuyu gizlemesini engeller ama tamamen degil)
# kullanim: python scripts/17_tespit_olc.py dev dev_m1920 dev_11l ...
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix * iy / u if u else 0

print(f"{'deney':14s} {'model':11s} {'imgsz':>5s} {'ms/kare':>7s} {'ref recall':>11s} "
      f"{'yakin':>7s} {'uzak':>7s} {'sinirli/10':>10s} {'10/10 kare':>10s} {'ID':>5s}")
for ad in sys.argv[1:]:
    out = Path("out") / ad
    meta = json.load(open(out / "meta.json")) if (out / "meta.json").exists() else {}
    video = Path(meta.get("video", f"data/{ad}.mp4")).stem
    ref = json.load(open(f"data/tespit_ref_{video}.json"))["kareler"]
    izler = json.load(open(out / "izler.json"))
    by = defaultdict(list)
    for k in izler:
        by[k["kare"]].append(k["kutu"])
    sayac = Counter()
    for r in ref:
        for g in r["oyuncular"]:
            bulundu = any(iou(g, b) >= 0.5 for b in by[r["kare"]])
            grup = "uzak" if g[3] - g[1] < 90 else "yakin"
            sayac[(grup, bulundu)] += 1
    def oran(*gr):
        b = sum(sayac[(g, True)] for g in gr); n = b + sum(sayac[(g, False)] for g in gr)
        return f"{b}/{n} %{b/n*100:.0f}" if n else "-"
    takim = json.load(open(out / "izler_takim.json"))
    kt = defaultdict(Counter)
    for k in takim:
        kt[k["kare"]][k["takim"]] += 1
    n_kare = max(k["kare"] for k in takim) + 1
    sinirli = np.array([min(kt[f]["kirmizi"], 5) + min(kt[f]["lacivert"], 5) for f in range(n_kare)])
    print(f"{ad:14s} {meta.get('model', '?'):11s} {meta.get('imgsz', 0):5d} {meta.get('ms_kare', 0):7.1f} "
          f"{oran('yakin', 'uzak'):>11s} {oran('yakin').split()[-1]:>7s} {oran('uzak').split()[-1]:>7s} "
          f"{sinirli.mean():10.2f} {(sinirli == 10).mean()*100:9.0f}% {len({k['id'] for k in izler}):5d}")
