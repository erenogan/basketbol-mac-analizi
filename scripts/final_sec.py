import argparse, json, re, subprocess, threading, time
from pathlib import Path
import cv2, numpy as np
import imageio_ffmpeg
from scipy.optimize import linear_sum_assignment
from takim_renk import govde, kare_etiketi

# Tam mactan final.mp4 secimi: saniyede 1 kare orneklenir, her pencereyi puanlar,
# en iyisini keser. Tum hat tam macta calismaz; sadece hafif tarama (CLAUDE.md kurali).
# Kriterler (final.mp4'ten ne istedigimiz):
#   - gercek mac oynaniyor   : sahada 7-12 takim oyuncusu, iki takimdan da >=2 (isinma /
#                              devre arasi / mola elenir)
#   - iki yari saha          : oyuncu ortalamasi sol ve sag yarida; kolay yakin sahada
#                              olcum sahte iyi cikar
#   - olu top var            : olu top tespitini sinamak icin >=5 sn yavas an
#   - kamera yerinde         : kalibrasyon karesine gore kayma < 3 px
#   - dev/smoke ile cakismaz : kurallar dev'de ayarlandi; ayni goruntuyle olcmek ezber olur
# Tarama out/final_sec_tarama.json'a yazilir; yeniden puanlama saniyeler surer.
# kullanim: python scripts/final_sec.py data/mac_tam.mp4 [--kes] [--sure 420]
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--sure", type=int, default=420, help="pencere, saniye (7 dk)")
ap.add_argument("--adim", type=int, default=15, help="pencere kaydirma, saniye")
ap.add_argument("--haric", nargs="*", default=["data/dev.mp4", "data/smoke.mp4"],
                help="maçtan kesilmis, final'le cakismamasi gereken klipler")
ap.add_argument("--cikti", default="data/final.mp4")
ap.add_argument("--kes", action="store_true", help="secilen pencereyi cikti dosyasina kes")
ap.add_argument("--yeniden-tara", action="store_true", help="onbellegi yok say")
ap.add_argument("--model", default="yolo11l.pt")
ap.add_argument("--baslangic", type=float, help="puanlamanin 1.'si yerine bu pencere (sn); --neden ile")
ap.add_argument("--neden", default="", help="elle secimin gerekcesi (final_sec.json'a yazilir)")
a = ap.parse_args()

EXE = imageio_ffmpeg.get_ffmpeg_exe()
W, H = 1280, 720                     # tarama cozunurlugu (1920x1080'in 2/3'u)
OLCEK = 1920 / W
TARAMA = Path("out") / "final_sec_tarama.json"
saha = np.array(json.load(open("data/saha.json"))["saha"], dtype=np.int32)
Hm = np.array(json.load(open("data/kalibrasyon.json"))["H_piksel_saha"])


def ornek_tara(video, w, h, isle, hiz=1):
    """Saniyede `hiz` kare ornekler, her kare icin isle(i, kare) cagirir; doner: zamanlar (sn).
    Anahtar kare atlama kullanilmiyor: anahtar kare araligi dosyaya bagli (dev.mp4 ~1 sn,
    tam mac 5.61 sn) -> ornekleme hizi kodlamaya bagli kaliyordu. Tam cozum 2 saatte ~2 dk."""
    p = subprocess.Popen([EXE, "-hide_banner", "-i", video,
                          "-vf", f"fps={hiz},scale={w}:{h},showinfo", "-an",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pts = []
    def oku_hata():
        for satir in p.stderr:
            m = re.search(rb"pts_time:([\d.]+)", satir)
            if m:
                pts.append(float(m.group(1)))
    th = threading.Thread(target=oku_hata, daemon=True); th.start()
    n, i = w * h * 3, 0
    while len(buf := p.stdout.read(n)) == n:
        isle(i, np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        i += 1
    p.wait(); th.join()
    return pts[:i]


def kucuk(kare):
    # cakisma eslestirmesi icin kucuk gri iz. INTER_AREA: alan ortalamasi, sonuc
    # baslangic cozunurlugunden bagimsiz (varsayilan bilinear'da ayni kare 1280'den
    # ve 320'den farkli iz veriyordu, MSE ~72)
    return cv2.resize(cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY), (64, 36),
                      interpolation=cv2.INTER_AREA).astype(np.float32)


def kamera_kaymasi(ref, kare, orb, bf):
    """Ust yari (duvar/tavan, oyuncu yok) ORB eslesmesi; kayma 1920 px cinsinden."""
    m = np.zeros(ref.shape, np.uint8); m[: H * 2 // 5] = 255
    k0, d0 = orb.detectAndCompute(ref, m); k1, d1 = orb.detectAndCompute(kare, m)
    if d0 is None or d1 is None:
        return 99.0
    e = bf.match(d0, d1)
    if len(e) < 30:
        return 99.0
    A, _ = cv2.estimateAffinePartial2D(np.float32([k0[x.queryIdx].pt for x in e]),
                                       np.float32([k1[x.trainIdx].pt for x in e]),
                                       method=cv2.RANSAC, ransacReprojThreshold=2)
    return 99.0 if A is None else float(np.hypot(A[0, 2], A[1, 2]) * OLCEK)


# ---------- 1) tarama (onbellekli) ----------
onbellek = json.loads(TARAMA.read_text()) if TARAMA.exists() else {}
if onbellek.get("video") == a.video and not a.yeniden_tara:
    print(f"onbellekten: {TARAMA}")
    tarama = onbellek
else:
    from ultralytics import YOLO
    model = YOLO(a.model)
    kal = json.load(open("data/kalibrasyon.json"))
    c = cv2.VideoCapture(kal["video"]); c.set(cv2.CAP_PROP_POS_FRAMES, kal["kare"])
    ref = cv2.cvtColor(cv2.resize(c.read()[1], (W, H)), cv2.COLOR_BGR2GRAY)
    orb, bf = cv2.ORB_create(3000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ornek = []
    t0 = time.perf_counter()
    def isle(i, kare):
        r = model.predict(kare, imgsz=W, conf=0.25, iou=0.5, classes=[0], device=0,
                          half=True, verbose=False)[0]
        nk = nl = 0; konum = []
        for b in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = b
            ayak = ((x1 + x2) / 2 * OLCEK, y2 * OLCEK)
            if cv2.pointPolygonTest(saha, ayak, False) < 0:
                continue
            g = govde(kare, b)
            etk = kare_etiketi(g)[0] if g is not None else -1
            if etk in (0, 1):
                nk += etk == 0; nl += etk == 1
                konum.append(cv2.perspectiveTransform(np.float32([[ayak]]), Hm)[0, 0].round(2).tolist())
        kayma = kamera_kaymasi(ref, cv2.cvtColor(kare, cv2.COLOR_BGR2GRAY), orb, bf) if i % 60 == 0 else None
        ornek.append({"nk": int(nk), "nl": int(nl), "konum": konum, "iz": kucuk(kare).tolist(),
                      "kayma": kayma})
        if i % 600 == 0:
            print(f"  {i} ornek  ({time.perf_counter() - t0:.0f} sn)")
    pts = ornek_tara(a.video, W, H, isle)
    tarama = {"video": a.video, "pts": pts, "ornek": ornek}
    TARAMA.parent.mkdir(exist_ok=True)
    TARAMA.write_text(json.dumps(tarama))
    print(f"tarama: {len(pts)} ornek, {time.perf_counter() - t0:.0f} sn -> {TARAMA}")

# haric klipler her seferinde (ucuz, 320x180): --haric degisirse onbellek bayatlamaz
haric = {}
for v in a.haric:
    if Path(v).exists():
        _iz = []
        _pts = ornek_tara(v, 320, 180, lambda i, k: _iz.append(kucuk(k).tolist()))
        haric[v] = {"pts": _pts, "iz": _iz}

pts = np.array(tarama["pts"]); ornek = tarama["ornek"]; T = float(pts[-1]) if len(pts) else 0
ORNEK_HIZI = len(pts) / T if T else 1.0
nk = np.array([o["nk"] for o in ornek]); nl = np.array([o["nl"] for o in ornek])
oyun = (nk >= 2) & (nl >= 2) & (nk + nl >= 7) & (nk + nl <= 12)
ort_x = np.array([np.mean([p[0] for p in o["konum"]]) if o["konum"] else np.nan for o in ornek])

# ---------- 2) hiz: ardisik anahtar karelerde oyuncu eslestirme ----------
hiz = np.full(len(ornek), np.nan)
for i in range(1, len(ornek)):
    A_, B_ = np.array(ornek[i - 1]["konum"]), np.array(ornek[i]["konum"])
    dt = pts[i] - pts[i - 1]
    if len(A_) < 4 or len(B_) < 4 or not 0 < dt < 3:
        continue
    D = np.linalg.norm(A_[:, None] - B_[None], axis=2)
    r_, c_ = linear_sum_assignment(D)
    d = D[r_, c_]; d = d[d < 6]                      # 6 m'den uzak eslesme ayni kisi degil
    if len(d) >= 4:
        hiz[i] = np.median(d) / dt

# ---------- 3) haric kliplerin mactaki yeri ----------
izler = np.array([o["iz"] for o in ornek], np.float32)
cakisma = []
for v, h in haric.items():
    # Zamanlama tutarliligi: klip mactan kesildiyse her karesinin mactaki en yakin ani
    # ayni sabit kaymayi verir. Tek kare benzerlik esigi yetmiyor: kamera sabit, iz
    # hep ayni salon; 0.5 sn ornek kaymasinda oran testi karelerin %28'inde tutuyordu.
    # Olculdu (dev'de): gercek parca %71-88 tutarli, icinde olmayan klip %14 -> esik %40.
    kayma = []
    for t_klip, iz in zip(h["pts"], h["iz"]):
        j = int(((izler - np.array(iz, np.float32)) ** 2).mean(axis=(1, 2)).argmin())
        kayma.append(round(pts[j] - t_klip))
    kayma = np.array(kayma)
    degerler, sayi = np.unique(kayma, return_counts=True)
    mod = int(degerler[sayi.argmax()]); tutarli = np.mean(np.abs(kayma - mod) <= 1)
    # %25-40 belirsiz: yine de dislanir. Yanlis dislama ~klip suresi kadar aday kaybettirir;
    # yanlis dahil etme final'i ayar yapilan goruntuyle cakistirir -> bedeller esit degil.
    if tutarli >= 0.25:
        bas, son = float(mod), float(mod + h["pts"][-1])
        cakisma.append((bas, son))
        durum = "bulundu" if tutarli >= 0.4 else "OLASI (belirsiz, ihtiyaten dislandi)"
        print(f"{v}: macta {bas:.0f}-{son:.0f} sn, {durum} (karelerin %{tutarli * 100:.0f}'i tutarli)")
    else:
        print(f"{v}: macta bulunamadi (tutarli kare %{tutarli * 100:.0f}) - farkli kayit olabilir")

# "mac oynaniyor" kapisi: tarama ayarlarinda (1280, conf 0.25) kesin mac oynanan dev
# araliginda bile oran %70 (ust uste binme, uzak saha kaybi). Devre arasi %0-16.
# Ilk surumdeki %80 olculmemisti ve 455 pencerenin 453'unu eliyordu.
OYUN_KAPI = 0.6
kontrol = [(pts[i], o["kayma"]) for i, o in enumerate(ornek) if o["kayma"] is not None]

# ---------- 4) pencere puanlama ----------
def puanla(bas):
    m = (pts >= bas) & (pts < bas + a.sure)
    if m.sum() < a.sure * ORNEK_HIZI * 0.5:              # yeterli ornek yok (video sonu vb.)
        return None
    oy = oyun[m].mean()
    x = ort_x[m & oyun]
    sol = np.mean(x < 14) if len(x) else 0
    yari = min(sol, 1 - sol)
    yavas = (hiz[m] < 1.0) & oyun[m]                   # olu top: oyun var ama herkes yavas
    en_uzun, cur = 0, 0
    for v in yavas:
        cur = cur + 1 if v else 0; en_uzun = max(en_uzun, cur)
    olu = en_uzun >= 5
    yakin = [k for t, k in kontrol if bas - 60 <= t <= bas + a.sure + 60]
    kamera = max(yakin) if yakin else 0.0
    cak = any(bas < s + 30 and bas + a.sure > b - 30 for b, s in cakisma)
    kapi = oy >= OYUN_KAPI and yari >= 0.2 and kamera < 3 and not cak
    puan = 0.4 * oy + 0.4 * min(yari / 0.5, 1) + 0.2 * olu
    return dict(bas=bas, puan=puan, oyun=oy, sol=sol, yari=yari, olu_sn=en_uzun,
                kamera_px=kamera, cakisma=cak, gecer=kapi)

pencereler = [p for b in np.arange(0, max(T - a.sure, 0) + 1, a.adim) if (p := puanla(float(b)))]
gecen = sorted([p for p in pencereler if p["gecer"]], key=lambda p: -p["puan"])
print(f"\nmac {T / 60:.0f} dk | oyun oynanan ornek %{oyun.mean() * 100:.0f} | "
      f"{len(pencereler)} pencere, {len(gecen)} kapidan gecti")
def zaman(s):
    return f"{int(s // 3600)}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"
print(f"{'baslangic':>9s} {'puan':>5s} {'oyun':>5s} {'sol/sag':>8s} {'olu top':>7s} {'kamera':>6s}")
for p in gecen[:5]:
    print(f"{zaman(p['bas']):>9s} {p['puan']:5.2f} {p['oyun']:5.0%} "
          f"{p['sol']:4.0%}/{1 - p['sol']:<3.0%} {p['olu_sn']:5d}sn {p['kamera_px']:5.1f}px")
if not gecen:
    elenme = {k: sum(not c(p) for p in pencereler) for k, c in (
        ("oyun<kapi", lambda p: p["oyun"] >= OYUN_KAPI), ("tek yari", lambda p: p["yari"] >= 0.2),
        ("kamera", lambda p: p["kamera_px"] < 3), ("cakisma", lambda p: not p["cakisma"]))}
    raise SystemExit(f"hicbir pencere kapidan gecmedi; elenme sebepleri: {elenme}")

sec, neden = gecen[0], "puanlamanin 1. adayi"
if a.baslangic is not None:
    eslesen = [p for p in gecen if abs(p["bas"] - a.baslangic) < a.adim / 2]
    if not eslesen:
        raise SystemExit(f"{a.baslangic} sn kapidan gecen pencereler arasinda yok")
    sec, neden = eslesen[0], a.neden or "elle secildi (gerekce verilmedi)"
    print(f"elle secim: sira {gecen.index(sec) + 1}/{len(gecen)}, puan {sec['puan']:.2f} "
          f"(1. aday {gecen[0]['puan']:.2f})")
json.dump({"video": a.video, "baslangic_sn": sec["bas"], "sure_sn": a.sure, "secim": sec,
           "secim_nedeni": neden, "ilk_5": gecen[:5], "cakisma_araliklari": cakisma},
          open("data/final_sec.json", "w"), indent=1)
print(f"\nsecilen: {zaman(sec['bas'])} - {zaman(sec['bas'] + a.sure)}  -> data/final_sec.json")

# ---------- 5) zaman cizelgesi ----------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 1, figsize=(14, 6.5), sharex=True)
dk = pts / 60
ax[0].fill_between(dk, 0, oyun.astype(float), step="mid", color="#2F6B4F", alpha=.6, lw=0)
ax[0].set_ylabel("maç\noynanıyor"); ax[0].set_yticks([])
ax[1].plot(dk, ort_x, ".", ms=2, color="#22417A"); ax[1].axhline(14, color="#999", lw=.8)
ax[1].set_ylabel("oyuncu ort.\nx (m)"); ax[1].set_ylim(0, 28)
ax[2].plot(dk, hiz, ".", ms=2, color="#8B2C3B"); ax[2].axhline(1.0, color="#999", lw=.8)
ax[2].set_ylabel("medyan hız\n(m/s)"); ax[2].set_ylim(0, 6); ax[2].set_xlabel("maç süresi (dk)")
for e in ax:
    for b, s in cakisma:
        e.axvspan(b / 60, s / 60, color="#999", alpha=.35, lw=0)
    e.axvspan(sec["bas"] / 60, (sec["bas"] + a.sure) / 60, color="#D9AE5B", alpha=.35, lw=0)
ax[0].set_title("final.mp4 seçimi — sarı: seçilen pencere, gri: dev/smoke (dışlandı)", loc="left")
fig.tight_layout(); fig.savefig("out/final_sec.png", dpi=110)
print("zaman cizelgesi -> out/final_sec.png")

# ---------- 6) kes ----------
if a.kes:
    print(f"kesiliyor -> {a.cikti}")
    subprocess.run([EXE, "-y", "-loglevel", "error", "-ss", str(sec["bas"]), "-i", a.video,
                    "-t", str(a.sure), "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                    "-an", a.cikti], check=True)
    print(f"kaydedildi -> {a.cikti}")
