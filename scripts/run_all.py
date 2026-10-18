import argparse, subprocess, sys, time
from pathlib import Path

# Tum hat tek komutla: tespit/takip -> takim -> iz bolme -> yumusatma -> isi haritasi
# (-> videolar). Bir adim hata verirse durur: bozuk ara ciktinin ustune devam etmez.
# kullanim: python scripts/run_all.py data/dev.mp4 [--etiket x] [--video] [-- 06 secenekleri]
#   ornek : python scripts/run_all.py data/dev.mp4 --etiket l1920 -- --model yolov8l.pt --imgsz 1920
ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--etiket", default="")
ap.add_argument("--video-ciktilari", "--video", action="store_true",
                help="takim videosu + minimap da uret (yavas)")
ap.add_argument("--atla-takip", action="store_true", help="izler.json varsa 06'yi atla")
a, takip_ek = ap.parse_known_args()
takip_ek = [x for x in takip_ek if x != "--"]

AD = Path(a.video).stem + (f"_{a.etiket}" if a.etiket else "")
PY = sys.executable
adimlar = []
if not a.atla_takip:
    adimlar.append(("06 tespit+takip", ["scripts/06_takip.py", a.video]
                    + (["--etiket", a.etiket] if a.etiket else []) + takip_ek))
adimlar += [("08 takim", ["scripts/08_takim.py", AD]),
            ("10 iz bolme", ["scripts/10_iz_bol.py", AD]),
            ("14 yumusatma", ["scripts/14_yumusat.py", AD]),
            ("15 isi haritasi", ["scripts/15_isi_haritasi.py", AD])]
if a.video_ciktilari:
    adimlar += [("11 takim videosu", ["scripts/11_takim_video.py", AD]),
                ("13 minimap", ["scripts/13_minimap.py", AD])]

ozet = []
for ad, komut in adimlar:
    print(f"\n===== {ad} =====", flush=True)
    t = time.perf_counter()
    r = subprocess.run([PY] + komut)
    sure = time.perf_counter() - t
    ozet.append((ad, sure, r.returncode))
    if r.returncode != 0:
        print(f"\n!! {ad} hata verdi (kod {r.returncode}), hat durduruldu")
        break

print(f"\n===== ozet: out/{AD} =====")
for ad, sure, kod in ozet:
    print(f"  {ad:18s} {sure:7.1f} sn  {'OK' if kod == 0 else 'HATA'}")
print(f"  {'toplam':18s} {sum(s for _, s, _ in ozet):7.1f} sn")
sys.exit(max(k for _, _, k in ozet))
