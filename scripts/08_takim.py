import cv2, json, sys, numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from ortak import video_yolu
from takim_renk import govde, kare_etiketi   # kurallar orada (06 takip, final_sec de kullaniyor)

# kullanim: python scripts/08_takim.py dev  -> data/dev.mp4, out/dev/izler.json
AD    = sys.argv[1] if len(sys.argv) > 1 else "smoke"
VIDEO = video_yolu(AD)
OUT   = Path("out") / AD

PENCERE = 15           # iz icinde +-15 kare cogunluk (yarim saniye)
# -1 "belirsiz": forma kaniti yok (yandan gorunum) -> komsu karelerden doldurulur
# -2 "disarida" : kanit var, takim degil (koyu kiyafet) -> doldurulmaz, oy vermez
ETIKET = {0: "kirmizi", 1: "lacivert", -1: "belirsiz", -2: "disarida"}


kayit = json.load(open(OUT / "izler.json"))
kare_index = defaultdict(list)
for i, k in enumerate(kayit):
    kare_index[k["kare"]].append(i)

# 1) kare bazinda ham etiket
ham = np.full(len(kayit), -1)
cap = cv2.VideoCapture(VIDEO)
kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    for i in kare_index.get(kare, []):
        g = govde(frame, kayit[i]["kutu"])
        if g is None:
            continue
        ham[i], r, n = kare_etiketi(g)
        kayit[i]["oran"] = [round(r, 2), round(n, 2)]
    kare += 1
cap.release()

# 2) iz icinde kayan pencere cogunlugu (belirsizler oy vermez)
iz_index = defaultdict(list)
for i, k in enumerate(kayit):
    iz_index[k["id"]].append(i)

son = ham.copy()        # -2 oldugu gibi kalir
for tid, idx in iz_index.items():
    idx.sort(key=lambda i: kayit[i]["kare"])
    kareler = np.array([kayit[i]["kare"] for i in idx])
    etk = ham[idx]
    for j, i in enumerate(idx):
        if etk[j] == -2:
            continue
        m = (np.abs(kareler - kareler[j]) <= PENCERE) & (etk >= 0)
        son[i] = Counter(etk[m].tolist()).most_common(1)[0][0] if m.any() else -1

for i, k in enumerate(kayit):
    k["takim"] = ETIKET[int(son[i])]
json.dump(kayit, open(OUT / "izler_takim.json", "w"))

# 3) rapor
print(f"tespit: {len(kayit)}")
for ad, dizi in (("ham", ham), ("yumusatilmis", son)):
    c = Counter(ETIKET[int(v)] for v in dizi)
    print(f"  {ad:13s}: " + "  ".join(f"{e}={c[e]} (%{c[e]/len(dizi)*100:.0f})"
                                      for e in ("kirmizi", "lacivert", "belirsiz", "disarida")))

per_kare = defaultdict(Counter)
for i, k in enumerate(kayit):
    per_kare[k["kare"]][int(son[i])] += 1
kr = np.array([per_kare[f][0] for f in range(kare)])
lc = np.array([per_kare[f][1] for f in range(kare)])
print(f"\nkare basina ortalama  kirmizi={kr.mean():.1f}  lacivert={lc.mean():.1f}"
      f"  (beklenen 5 / 5)")
print(f"  kirmizi=5 olan kare : %{(kr == 5).mean()*100:.0f}   "
      f"lacivert=5 olan kare: %{(lc == 5).mean()*100:.0f}")

print("\niz icinde takim degisimi (takimlar arasi ID degisimi supheleri):")
for tid, idx in sorted(iz_index.items()):
    seg = []
    for i in idx:
        e = int(son[i])
        if e < 0:
            continue
        if not seg or seg[-1][0] != e:
            seg.append([e, kayit[i]["kare"], 0])
        seg[-1][2] += 1
    birlesik = []
    for s_ in seg:                         # 10 kareden kisa parcalar gurultu
        if s_[2] < 10:
            continue
        if birlesik and birlesik[-1][0] == s_[0]:
            birlesik[-1][2] += s_[2]
        else:
            birlesik.append(s_)
    seg = birlesik
    if len({s[0] for s in seg}) > 1:
        print(f"  #{tid:>3}: " + " -> ".join(
            f"{ETIKET[s[0]]}@{s[1]}({s[2]})" for s in seg))

# 4) referansa karsi olcum (data/takim_referans.json, kor etiketlendi)
def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - ix*iy
    return ix * iy / u if u else 0

TAHMIN = {0: "kirmizi", 1: "lacivert", -1: "disarida", -2: "disarida"}   # belirsiz = takim yok
# data/takim_ref_<video>_<set>.json, kor etiketli (09_referans_pano.py)
# smoke_gelistirme: kurallar bu setteki hatalara bakilarak kuruldu (iyimser)
# diger setler: kurallar kesinlestikten sonra etiketlendi, ayar yapilmadi
REF_SETLER = [(p.stem.split("_", 3)[-1], str(p))
              for p in sorted(Path("data").glob(f"takim_ref_{Path(VIDEO).stem}_*.json"))]
for (set_ad, yol), (ad, dizi) in (
        (s, d) for s in REF_SETLER for d in (("ham", ham), ("yumusatilmis", son))):
    ref = json.load(open(yol))
    ad = f"{set_ad}, {ad}"
    mat, dogru, n = Counter(), 0, 0
    boy = Counter()                       # (uzak/yakin, dogru mu) -> sayi
    for r in ref:
        if r["takim"] == "karisik":
            continue
        aday = [(iou(r["kutu"], kayit[i]["kutu"]), i) for i in kare_index[r["kare"]]]
        en, i = max(aday, default=(0, -1))
        t = TAHMIN[int(dizi[i])] if en > 0.5 else "eslesmedi"
        mat[(r["takim"], t)] += 1
        dogru += r["takim"] == t
        n += 1
        grup = "uzak" if r["kutu"][3] - r["kutu"][1] < 90 else "yakin"
        boy[(grup, r["takim"] == t)] += 1
    print(f"\nreferans dogrulugu ({ad}): {dogru}/{n} = %{dogru/n*100:.0f}")
    for g in ("yakin", "uzak"):           # kutu boyu <90 px = uzak saha
        gn = boy[(g, True)] + boy[(g, False)]
        if gn:
            print(f"  {g:5s} (kutu {'>=' if g == 'yakin' else '<'}90px): "
                  f"{boy[(g, True)]}/{gn} = %{boy[(g, True)]/gn*100:.0f}")
    sut = ("kirmizi", "lacivert", "disarida")
    print("  gercek / tahmin " + "".join(f"{c:>10s}" for c in sut))
    for g in sut:
        print(f"  {g:16s}" + "".join(f"{mat[(g, c)]:>10d}" for c in sut))

print(f"\nkaydedildi -> {OUT / 'izler_takim.json'}")
