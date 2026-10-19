# Basketbol Maç Analizi

Sabit kameralı amatör basketbol kaydından oyuncu takibi, takım ayrımı, saha koordinatı
ve takım bazlı koşu metrikleri çıkaran bir görüntü işleme hattı.

Tek bir geniş açı kamera, 1920×1080, 30 fps. Hazır saha modeli yok: çok amaçlı bir salonda
birkaç sporun çizgileri üst üste. Top görüntüde ~20 piksel, uzak oyuncular ~60 piksel.

## Ne yapıyor

```
video ─► oyuncu tespiti ─► takip (+ forma rengi kısıtı) ─► takım ayrımı
             YOLO11l            ByteTrack alt sınıfı          HSV piksel sayımı
                                                                   │
   ısı haritası ◄─ mesafe ◄─ ölü top ◄─ iz yumuşatma ◄─ saha koordinatı (homography)
```

| Adım | Yöntem | Ölçülen sonuç |
|---|---|---|
| Takım ayrımı | Gövde kırpıntısında HSV piksel sayımı, iz içinde zamansal oylama | %97 (kör referans, 78 tespit) |
| Takip | ByteTrack + forma rengine göre eşleşme yasağı (kırmızı / lacivert / takım dışı) | Takımlar arası kimlik geçişi ~85 → ~10–30 (3 dk) |
| Saha koordinatı | 15 elle işaretlenmiş FIBA noktasından homography | Medyan 0.16 m hata (birini-dışarıda) |
| İz yumuşatma | Medyan filtre + Savitzky–Golay | İmkânsız sıçrama 813 → 49, gerçek hareket korunuyor |
| Ölü top | Bütün oyuncuların medyan hızı < 1 m/s, ≥ 3 sn | Geliştirme klibinde 2/2 aralık doğru |
| Takım mesafesi | Oyuncu başına m/dk, canlı oyun ve ölü top ayrı, hız bölgeleri | Uzak sahada ~%10 şişme ölçüldü ve raporlanıyor |

Sınırların tamamı ve nasıl okunması gerektiği: [LIMITASYONLAR.md](LIMITASYONLAR.md)

## Ölçme yaklaşımı

Her karar bir referansa karşı ölçüldü. Takım ayrımı ve oyuncu bulma için kareler elle
etiketlendi, etiketleme sırasında sistemin tahmini gösterilmedi (`data/takim_ref_*.json`,
`data/tespit_ref_*.json`). Kurallar bir sette kurulup hiç dokunulmamış ayrı bir sette ölçüldü.
Denenip işe yaramayan yöntemler de kayıtlı: Lab rengi + KMeans, BoT-SORT ReID, iz bölme,
hareket kapısı, düşük yeni iz eşiği.

## Çalıştırma

```bash
# GPU'lu torch (PyPI sürümü CPU'dur)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install ultralytics opencv-python scipy matplotlib imageio-ffmpeg

# tüm hat: tespit/takip -> takım -> yumuşatma -> mesafe -> ısı haritası [-> videolar]
python scripts/run_all.py data/video.mp4 --video

# teslim paketi: tracks.json, tracks.csv, ozet.json, ısı haritası, video
python scripts/19_teslim.py video
```

Saha kalibrasyonu kamera başına bir kez yapılır: `python scripts/12_kalibrasyon.py`
(bilinen saha noktalarını tıklama arayüzü; birini-dışarıda hatasını raporlar).

Çıktılar `out/<video>/` altına yazılır. Her koşunun ayarları `meta.json`'da.
Deney için: `python scripts/run_all.py data/video.mp4 --etiket deneme -- --model yolov8m.pt`

## Klasörler

| Yol | İçerik |
|---|---|
| `scripts/06–18` | Hattın adımları (numara sırası çalışma sırası) |
| `scripts/saha_model.py` | FIBA saha geometrisi (metre) |
| `scripts/takim_renk.py` | Forma rengi kuralları |
| `scripts/takim_tracker.py` | Takım kısıtlı ByteTrack |
| `data/*.json` | Saha poligonu, kalibrasyon, elle hazırlanmış referanslar |
| `bytetrack_ozel.yaml` | Takipçi ayarları |

Videolar ve model ağırlıkları depoda değil.

## Kapsam dışı

Top görüntüde tespit edilemediği için sayı, pas ve top sahipliği yok. Forma numarası
okunmadığı ve takip kimlikleri oyuncu başına güvenilir olmadığı için metrikler oyuncu
bazında değil, takım bazında.
