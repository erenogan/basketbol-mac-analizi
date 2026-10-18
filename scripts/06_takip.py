import argparse, cv2, json, queue, threading, time, numpy as np
from pathlib import Path
from ultralytics import YOLO

# Tespit + takip -> out/<video>[_<etiket>]/izler.json (+ meta.json).
# Gorsellestirme 11_takim_video / 13_minimap'te; burada video yazmak istege bagli
# (--video): yazma + ffmpeg smoke'ta kare basina ~13 ms ekliyordu.
# kullanim: python scripts/06_takip.py data/dev.mp4 [--etiket fp32 --fp32 --model yolov8l.pt]
ap = argparse.ArgumentParser()
ap.add_argument("video", nargs="?", default="data/smoke.mp4")
ap.add_argument("--model", default="yolov8m.pt")
ap.add_argument("--imgsz", type=int, default=1280)
ap.add_argument("--fp32", action="store_true", help="fp16 yerine tam hassasiyet")
ap.add_argument("--etiket", default="", help="deney ciktisi icin klasor eki")
ap.add_argument("--video-yaz", action="store_true", help="kimlik renkli takip videosu da yaz")
a = ap.parse_args()

CONF, IOU = 0.05, 0.5          # conf modele dusuk, eleme takipcide (bkz. CLAUDE.md)
OUT = Path("out") / (Path(a.video).stem + (f"_{a.etiket}" if a.etiket else ""))
OUT.mkdir(parents=True, exist_ok=True)

saha  = np.array(json.load(open("data/saha.json"))["saha"], dtype=np.int32)
model = YOLO(a.model)
cap = cv2.VideoCapture(a.video)
if not cap.isOpened():
    raise SystemExit(f"acilamadi: {a.video}")
fps, TOPLAM = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W, H = int(cap.get(3)), int(cap.get(4))

# Video cozme ayri thread'de: GPU calisirken bir sonraki kare hazirlanir.
kuyruk = queue.Queue(maxsize=16)
def okuyucu():
    while True:
        ok, f = cap.read()
        kuyruk.put(f if ok else None)
        if not ok:
            return
threading.Thread(target=okuyucu, daemon=True).start()

yazici = None
if a.video_yaz:
    yazici = cv2.VideoWriter(str(OUT / "takip.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

kayit, gorulen, kare = [], {}, 0
t0 = time.perf_counter()
while (frame := kuyruk.get()) is not None:
    r = model.track(frame, imgsz=a.imgsz, conf=CONF, iou=IOU, classes=[0], device=0,
                    half=not a.fp32, persist=True, tracker="bytetrack_ozel.yaml",
                    verbose=False)[0]
    if r.boxes.id is not None:
        for b, tid, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.id.cpu().numpy().astype(int),
                             r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, b)
            ayak = (float((x1 + x2) / 2), float(y2))       # zemin noktasi
            if cv2.pointPolygonTest(saha, ayak, False) < 0:
                continue
            gorulen[tid] = gorulen.get(tid, 0) + 1
            kayit.append({"kare": kare, "id": int(tid), "x": int(ayak[0]), "y": int(ayak[1]),
                          "kutu": [x1, y1, x2, y2], "conf": round(float(c), 3)})
            if yazici:
                renk = tuple(int(v) for v in np.random.default_rng(int(tid)).integers(60, 255, 3))
                cv2.rectangle(frame, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(frame, f"#{tid}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, renk, 2)
    if yazici:
        yazici.write(frame)
    kare += 1
    if kare % 600 == 0:
        print(f"  {kare}/{TOPLAM}")
sure = time.perf_counter() - t0
cap.release()

# ana sonuc once diske; video donusumu kozmetik, sonra ve try icinde
json.dump(kayit, open(OUT / "izler.json", "w"))
meta = {"video": a.video, "model": a.model, "imgsz": a.imgsz, "half": not a.fp32,
        "conf": CONF, "iou": IOU, "tracker": "bytetrack_ozel.yaml",
        "kare": kare, "ms_kare": round(sure / max(kare, 1) * 1000, 1)}
json.dump(meta, open(OUT / "meta.json", "w"), indent=1)
print(f"\nislenen kare      : {kare}  ({meta['ms_kare']} ms/kare, dongu)")
print(f"toplam farkli ID  : {len(gorulen)}")
print(f"yarisindan fazla  : {sum(n > kare * 0.5 for n in gorulen.values())}  (kararli izler)")
print(f"kaydedildi -> {OUT / 'izler.json'}")

if yazici:
    yazici.release()
    try:
        import shutil, subprocess
        exe = shutil.which("ffmpeg")
        if exe is None:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", str(OUT / "takip.mp4"),
                        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                        str(OUT / "takip_izle.mp4")], check=True)
    except Exception as e:
        print(f"donusturme basarisiz: {e}")
