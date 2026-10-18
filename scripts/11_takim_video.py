import cv2, json, sys, shutil, subprocess, numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from ortak import video_yolu

# Takim etiketini gozle kontrol etmek icin: kutular kare bazli takim rengiyle.
# kullanim: python scripts/11_takim_video.py dev
AD    = sys.argv[1] if len(sys.argv) > 1 else "smoke"
VIDEO = video_yolu(AD)
OUT   = Path("out") / AD
RENK  = {"kirmizi": (40, 40, 230), "lacivert": (230, 120, 20),
         "disarida": (150, 150, 150), "belirsiz": (255, 255, 255)}

kayit = json.load(open(OUT / "izler_takim.json"))
saha  = np.array(json.load(open("data/saha.json"))["saha"], dtype=np.int32)
kare_index = defaultdict(list)
for k in kayit:
    kare_index[k["kare"]].append(k)

cap = cv2.VideoCapture(VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(3)), int(cap.get(4))
ham = str(OUT / "takim_video.mp4")
yazici = cv2.VideoWriter(ham, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

kare = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    say = Counter()
    for k in kare_index.get(kare, []):
        x1, y1, x2, y2 = k["kutu"]
        t = k["takim"]
        say[t] += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), RENK[t], 2 if t != "belirsiz" else 1)
        if t in ("kirmizi", "lacivert"):
            cv2.circle(frame, ((x1 + x2) // 2, y2), 4, RENK[t], -1)
    cv2.polylines(frame, [saha], True, (0, 200, 255), 1)
    cv2.rectangle(frame, (10, 10), (560, 60), (0, 0, 0), -1)
    cv2.putText(frame, f"kare {kare}  kirmizi {say['kirmizi']}  lacivert {say['lacivert']}"
                f"  disari {say['disarida']}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    yazici.write(frame)
    kare += 1
cap.release()
yazici.release()
print(f"kaydedildi -> {ham}")

# Windows oynaticisi mp4v acmiyor -> H.264 kopya (kozmetik, hata isi oldurmesin)
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
