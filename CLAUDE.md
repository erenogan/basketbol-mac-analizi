# Basketbol Maç Analizi — Proje Bağlamı

Sabit kameralı amatör basketbol kayıtlarından oyuncu takibi, takım ayrımı ve
saha koordinatı çıkarımı.

> Özel bağlam `CLAUDE.local.md`'de (git'e girmez). Bu dosya herkese açık depoda;
> kişi/kulüp adı ve kişisel bağlam buraya yazılmaz.

---

## Çalışma tarzı (buna uy)

- Türkçe, samimi dil.
- Her teknik kararın **nedenini** söyle, sadece kodu verme.
- Hata görürsen **anında** söyle, nazikleştirme.
- Bir şeyin **gerçek bir mühendislik deseni mi yoksa geçici çözüm mü** olduğunu belirt.
- Genel cesaretlendirme yok, doğrudan geri bildirim var.
- Kullanıcı görüntü işlemede sıfırdan başladı; Python ve yazılım mimarisi biliyor.
  Python öğretme, CV kavramlarını açıkla.

## Yöntem kuralı

**Ölçmeden karar verme.** Bu projede birkaç kez "sayı yükseldi, iyiye gidiyor"
tuzağına düşmenin eşiğinden dönüldü. Referans (ground truth) olmadan üretilen
hiçbir sayı kanıt değildir. Parametre değiştirirken tek seferde bir şey değiştir
ve etkisini ölç.

---

## Donanım / ortam

- RTX 4060 Laptop, 8 GB VRAM (ASUS ROG Strix G614JV), Windows
- Python 3.14.7, venv: `C:\basket\.venv`
- torch 2.11.0+cu128 (GPU doğrulandı), opencv 5.0.0, ultralytics, supervision
- ffmpeg PATH'te YOK → `imageio-ffmpeg` paketi üzerinden kullanılıyor

> torch'un CUDA sürümü PyPI'de yok, `--index-url https://download.pytorch.org/whl/cu128`
> ile kuruldu. `pip install ultralytics` tek başına CPU sürümü getiriyor — tuzak bu.

## Klasör yapısı

```
C:\basket\
├── data\         smoke.mp4, dev.mp4 (git dışı); saha.json, kalibrasyon.json,
│                 takim_ref_*.json, tespit_ref_*.json (elle, depoda)
├── out\          <video>[_<etiket>]\ : izler.json, meta.json, izler_takim.json,
│                 izler_bolunmus.json, izler_saha.json, isi_haritasi.png, *_izle.mp4
├── scripts\      06 takip → 08 takım → 10 iz bölme → 14 yumuşatma → 15 ısı haritası
│                 11 takım videosu, 13 minimap, 12 kalibrasyon, 16/17 tespit ölçümü,
│                 09 kör referans panosu, run_all.py (tek komut)
│                 ortak modüller: saha_model, takim_renk, takim_tracker, ortak
│                 01–05, 07: ilk keşif scriptleri
├── tracker_ayar\ deney eşikleri (ntt045/035 — elendi)
├── bytetrack_ozel.yaml
└── CLAUDE.local.md  özel bağlam (git dışı)
```

GitHub: `erenogan/basketbol-mac-analizi` (herkese açık). Commit'lerde Claude
imzası yok; yazar e-postası depoya özel noreply adresi.

> Not: `scripts/08_kirpinti, 09_oznitelik, 10_kume, 11_etiketle` kullanıcının
> kendi denemeleri (numaraları çakışıyor, dokunma; `out/smoke/izler.json` okurlar).

`out/`, videolar (`*.mp4`) ve model ağırlıkları (`*.pt`) git'e girmiyor. `data/` altındaki
elle hazırlanmış JSON referansları (takım/tespit etiketleri, kalibrasyon, saha) **bilerek
depoda** — kaybolursa hiçbir ölçüm yeniden doğrulanamaz.

---

## Video kaynağı

İki video var, **sadece birincisi kullanılıyor**:

| video | kamera | durum |
|---|---|---|
| A — geniş açı, tribün üstü | **sabit** (25 sn'de 21 px kayma) | ✅ kullanılıyor |
| B — alçak açı, yer seviyesi | sürekli dönüyor (2566 px) | ❌ kullanılmıyor |

İkincisi elendi: kamera yer seviyesinde, sahanın yarısı kadraj dışı, oyuncular
sürekli kadrajdan çıkıyor, elimizdeki kesit ısınma anı. Homography kurulamaz.

### Kullanılan videonun (A) özellikleri

- 1920x1080, ~30 fps
- Çok amaçlı salon → **saha çizgi çorbası** (siyah, kırmızı, mavi, sarı, kesikli).
  Hazır saha-keypoint modelleri burada çalışmaz → **elle 4 nokta kalibrasyon zorunlu**
  (geçici çözüm değil, bu görüntü için doğru karar).
- Takımlar kırmızı vs lacivert → renk ayrımı net, takım sınıflandırma kolay olmalı
- Yakın oyuncu ~200 px, uzak oyuncu ~60 px, top ~20 px
- Sağ altta korkuluk sahayı kapatıyor
- Kadrajda kulübe, masa görevlileri, seyirci, hakem var

---

## Referans (ground truth) — elle sayıldı

`out/kare.jpg` karesinde:

```
10 oyuncu + 1 hakem + 19 kulübe/seyirci = 30 kişi
```

Bütün ölçümler buna göre. **Önemsenen tek metrik: 10 oyuncunun kaçı bulundu.**
Toplam tespit sayısı anlamsız — 30'un 20'si zaten kulübede oturan insanlar.

---

## Seçilen ayarlar (hepsi ölçümle)

```python
model  = "yolo11l.pt"   # 2026-10-18: yolov8m'den geçildi (aşağıda "Optimizasyon turu")
imgsz  = 1280
half   = True           # fp16: döngü ~40 -> ~23-27 ms/kare, doğruluk aynı
conf   = 0.05           # modele düşük ver, eleme takipçide
iou    = 0.50
classes = [0]           # COCO person
tracker = "bytetrack_ozel.yaml" + takım kısıtı (scripts/takim_tracker.py)
filtre  = saha poligonu + ayak noktası testi (data/saha.json)
```

Tek komut: `python scripts/run_all.py data/dev.mp4 [--video] [--etiket x -- <06 seçenekleri>]`
(06 → 08 → 10 → 14 → 15 [→ 11 → 13]; hata veren adımda durur). Deney çıktıları
`out/<video>_<etiket>/`, her koşunun ayarları `meta.json`'da.

`bytetrack_ozel.yaml`:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.35
track_low_thresh: 0.08
new_track_thresh: 0.55
track_buffer: 90
match_thresh: 0.85
fuse_score: True
```

### Neden bu değerler

- **imgsz=1280**: varsayılan 640, 1920'lik kareyi 3 kat küçültüyor; uzak oyuncular
  (60 px) ve top (20 px) modele ulaşmadan yok oluyor. Küçük nesne tespitinde
  imgsz yükseltmek standart ayardır.
- **conf=0.05 (modelde)**: `conf` parametresi takipçiye değil modele gidiyor.
  0.25 verilince düşük güvenli kutular ByteTrack'e hiç ulaşmıyor ve algoritmanın
  ikinci-tur kurtarma mekanizması devre dışı kalıyor. Bu hata yapıldı, 61 ID
  üretti; düzeltilince 36'ya indi.
- **new_track_thresh=0.55**: zayıf kutular izi *sürdürebilsin* ama *yeni kimlik
  açamasın*. Bu ayrım kritik — öncesinde takipçi 310+ sahte iz açıyordu.
- **track_buffer=90**: kaybolan iz 3 sn hafızada (varsayılan 1 sn, basketbol
  kapanmaları için kısa).
- **iou=0.50**: varsayılan 0.70'te üst üste binen oyuncularda çift kutu kalıyordu
  (NMS eşiğin altında kaldığı için birleştiremiyordu).
- **Ayak noktası (kutu alt-orta)**: oyuncu zeminde, saha zeminde. Kutu merkezi
  havada kalıyor ve perspektifte metrelerce sapıyor. Aynı nokta homography'de de
  kullanılacak.

---

## Ölçülmüş sonuçlar

dev.mp4 (5543 kare, ~3 dk, oyun çoğunlukla uzak sahada):
```
takip                 : 162 farklı ID, %50+ kapsamalı iz 0, 300+ kare 62 iz
kare başına kişi      : 9.8 (saha içi), takım: 4.3 kırmızı / 4.0 lacivert
```
smoke'a göre parçalanma çok daha kötü → oyuncu bazlı metrik kesinlikle yok.

smoke.mp4 (647 kare, ~21.6 sn):

```
oyuncu tespiti        : 10/10
saha filtresi         : 30 kişiden 17'si elendi (hakem dahil)
kare başına kişi      : 10.3 ortalama
takip                 : 36 farklı ID, 7 iz %50+ kapsama, 8 iz 300+ kare
top tespiti           : 0/1 — hiçbir ayarda bulunamadı
hız                   : ~90 ms/kare (3 dk klip ≈ 8 dk işlem)
```

---

## Takım sınıflandırma (08_takim.py) — ölçüldü

`izler.json` artık kutu (`kutu`) ve `conf` da tutuyor (06_takip güncellendi,
36 ID / 7 kararlı iz aynen tekrarlandı). Çıktı: `out/izler_takim.json`
(her tespitte `takim`: kirmizi / lacivert / belirsiz / disarida).

Yöntem: gövde kırpıntısı (kutunun üst %15–45'i, orta %50) → HSV piksel sayımı
→ kare etiketi → iz içinde ±15 kare çoğunluk.

| kural | değer | neden (ölçümle) |
|---|---|---|
| kırmızı piksel | (H<2 \| H>165), S≥120, V≥60 | ten/zemin de kırmızı tonda ama S≈84; **H 2–10 hakemin turuncusu** |
| lacivert piksel | H 100–160, S≥70, V 30–140 | kamera laciverti morumsu kaydediyor; siyah kıyafet V<30, S≈50 |
| kırmızı / lacivert eşiği | oran ≥0.25 / ≥0.05 | lacivert koyu, kol/gölge daha çok yiyor |
| takım dışı | V<35 oranı ≥0.25 **ve** lacivert <0.05 | gölgedeki lacivert oyuncu da koyu ama forma kanıtı var |

Referans: `data/takim_ref_smoke_gelistirme.json` (seed 0), `..._smoke_test.json`
(seed 1), `..._dev_test.json` (seed 2). Hepsi kör etiketlendi
(`09_referans_pano.py`), 80'er tespit, "karışık" olanlar puanlamaya girmiyor.
Geliştirme dışındakiler kurallar bittikten sonra etiketlendi, ayar yapılmadı.

```
                          ham    yumuşatılmış
smoke geliştirme (76)     %88    %97
smoke test       (76)     %87    %96
dev test         (78)     %87    %97   <- raporlanacak; uzak saha (<90px) 47/47
```

dev.mp4'te oyunun çoğu uzak sahada; kurallar smoke'ta kurulup dev'de
dokunulmadan ölçüldü → gerçek genelleme ölçümü. Poligon dev'de de geçerli
(kamera kaymamış), ama uzak oyuncuların ayağı üst kenara 10–20 px — marj ince.

Denenip elenen: **Lab medyan rengi + KMeans** → lacivertlerin çoğu kırmızı
kümeye düştü. Forma kolsuz; gövde kırpıntısına ten + zemin giriyor, medyan
kayıyor (lacivert oyuncunun gövde piksellerinin %53'ü kırmızı tonda).

Sınırlar:
- Eşikler bu videonun ışığına ve formalarına göre elle ölçüldü →
  başka maça taşınmaz, yeniden ölçmek gerekir.
- Yandan görünen lacivert oyuncu kare bazında "belirsiz" (forma görünmüyor),
  yumuşatma dolduruyor.
- Sahada turuncu-siyah giyen **iki** kişi var (muhtemelen iki hakem), ikisi de
  doğru elendi.

**Yan bulgu — takımlar arası ID değişimi:** smoke'ta 11, **dev'de 60 izde** iz
ortasında takım değişiyor (smoke #5, #11; dev #36, #63 gözle doğrulandı: kapanma
anında lacivert oyuncunun ID'si kırmızı oyuncuya geçiyor). Sınıflandırıcı kare
bazında ~%97 doğru ve ≥10 karelik parçalar sayıldığı için bunlar titreme değil.

**İz bölme denendi (`10_iz_bol.py`) — ana hatta alınmadı.** dev'de 193 kesim,
iz sayısı 162 → 355. Rastgele 20 kesim gözle: 4 doğru yerde gerçek değişim,
7 gerçek ama geç (hakemin belirsiz kareleri önceki parçaya katılıyor), 4 titreme
(yandaki rakibin forması / uzak siyah giyen), 5 belirsiz → ~%55 isabet.
Ayak sıçraması kontrolü işe yaramadı: değişim kapanmada olduğu için ayaklar
aynı yerde, sıçrama yok. Takım mesafesine etkisi ≤%2. **Karar:** takım
metrikleri/ısı haritası/renkler kare bazlı `takim` etiketinden üretilecek;
bölme sadece rapora "3 dk'da en az ~100 takımlar arası ID değişimi" için.
Takipçi rakip takım oyuncularını karıştırıyor. İz bazında tek oy bunu gizlerdi;
etiket kare bazında tutuluyor. (İz bölme denendi, aşağıya bak — ana hatta yok.)

---

## Saha kalibrasyonu (12_kalibrasyon.py) — ölçüldü

- Basketbol sahası **bordo** çizgiler (modern FIBA: dikdörtgen boyalı alan,
  6.75 m yay). Siyahlar eski kuşak (yamuk boyalı alan, çift orta çember), aynı
  28×15 sahaya ait. Diğer renkler başka sporlar.
- Koordinat: orijin sol-uzak köşe (panolar tarafı), x boyuna 0–28, y kameraya
  doğru 0–15. `data/kalibrasyon.json` → `H_piksel_saha` (ayak noktası → metre).
- 15 elle tıklanmış nokta, en küçük kareler. 1 nokta çıkarıldı (köşeyle aynı
  piksel, fiziksel olarak imkânsız; gerekçe JSON'da).

```
birini-dışarıda hata : medyan 0.16 m, en kötü 0.73 m
sol yarı (yakın)     : ~0.05–0.15 m
orta                 : ~0.4 m
sağ uzak             : ~0.7 m   (orada 1 px ≈ 15–17 cm; tıklama hatası ~5 px)
```

- Piksel cinsinden hatalar 1–7 px, yönleri rastgele → **lens bükmesi yok**,
  tıklama gürültüsü.
- Kamera dev boyunca **kaymıyor** (ORB, 5 kare, <0.1 px) → tek kalibrasyon yeter.
- Oyuncu ayaklarının %99.1'i 28×15 içinde (1 m payla %100) — poligon filtresi
  yüzünden kısmen kendiliğinden; kaba hata yok demek, doğruluk kanıtı değil.
- Tuzak (öğretici): aynı çizgi üzerindeki noktalar homography'yi belirlemez;
  ilk denemede 4 nokta sol dip çizgideydi → bir köşe çıkınca 5000 m hata.
- **Sınır:** uzak sahada konum hassasiyeti ~0.5 m'nin altına inemez (çözünürlük).
  Kutu alt kenarı titremesi → aşağıdaki yumuşatma bölümü.

## İz yumuşatma (14_yumusat.py) — ölçüldü

Ham ayak noktası (metre): titreme ~9 cm, adımların %1.1'i tek karede >1 m sıçrama
(kutu birleşme/bölünme), iz içi kopmaların %68'i ≤10 kare.
Yöntem (iz anahtarı `iz` — takım değişiminde bölünmüş; `id` ile yumuşatsak
kimlik rakibe geçince iki oyuncu arasında hayalet nokta çizilirdi):
≤10 karelik kopmayı doğrusal doldur (`ara_doldurma: true`, tahmin olarak işaretli)
→ medyan(7) sıçramaları siler → Savitzky–Golay(15, 2) titremeyi alır.
Çıktı: `out/<video>/izler_saha.json` (sx, sy metre; ham_sx/ham_sy da duruyor).

```
dev                         ham      yumuşak
sıçrama >1 m/kare           596       32
hız p99 (m/s)               31.7      13.5    (insan ≤ ~9 → kalıntı var)
1 sn yer değiştirme (m)     1.17      1.17    <- gerçek hareket korunuyor
mesafe kırmızı / lacivert   3385/2650 2122/1617 m (3 dk)
```

Mesafe ≈ kırmızı 141, lacivert 108 m/dk/oyuncu (literatür ~110–125). Ground truth
yok → "makul aralıkta" denebilir, "doğru" denemez. Kırmızı yüksekliğinin olası
sebebi kenardaki kırmızı yedeklerin poligon içinde kalması (kare 2771).
Kalan 32 sıçrama büyük ihtimalle ribaunt yığılması; filtreyi sıkmak gerçek
hızlanmayı kırpar, referanssız yapılmadı.

Minimap (`13_minimap.py`): sol kamera + kalibre saha overlay, sağ kuşbakışı
tahta; dolu nokta = ölçüm, içi boş halka = ara doldurma.

**Ölü top tespiti** (14_yumusat içinde, `olu_top` alanı): top yok, ama ölü topta
herkes yavaşlar. Oyuncuların medyan hızı (2 sn ort.) < 1.0 m/s ve ≥3 sn →
ölü top. **Sezgisel kural**; eşik dev'de iki küme arasındaki boşluktan (ölü
~0.6–0.8, canlı ≥1.25; hareketsizken bile ~0.6 = konum gürültü tabanı).
dev: 2 aralık, 26 sn (1578–2099 bekleme/mola, 2215–2478 serbest atış dizilişi),
ikisi de gözle doğru; sınırdaki canlı an (3342, top yukarı taşınıyor) doğru.
Sadece 2+1 örnek → oran verilemez, final.mp4'te tekrar kontrol.

## Isı haritası (15_isi_haritasi.py)

Birim: takım süresinin %'si / m², her takım kendi toplamına normalize (lacivert
daha az tespit ediliyor; mutlak süre farkı tespitten gelirdi). Gauss σ=0.75 m
(= uzak saha kalibrasyon hatası; daha ince detay gürültü). Tek renkli rampalar,
iki takım aynı ölçek. **Ölü top hariç** — dahil edilince 17 sn'lik bekleme
haritanın en koyu iki lekesini tek başına üretiyordu.
Çıktı: `out/<video>/isi_haritasi.png`, `bolge_dagilimi.json`.

Yorum sınırları: top olmadığı için **hücum ve savunma karışık**; hangi takımın
hangi potaya hücum ettiği haritadan söylenemez. Konum kaybı kalabalık
bölgelerde (boyalı alan) daha sık → orası olduğundan az görünebilir.
Kırmızıda sol-uzak köşede (~2, 1.5 m) leke var — kulübe önündeki kırmızı yedekler
olabilir, kontrol edilmedi.

---

## Optimizasyon turu (2026-10-18) — ölçüldü

**Hız.** Profil: model+takip 36 ms, mp4 yazma + ffmpeg ~13 ms/kare, kütüphane
yükleme ~12 sn sabit. fp16 + 06'da video yazmayı isteğe bağlı yapmak (`--video-yaz`)
→ döngü 23 ms/kare (v8m). fp16 doğrulandı: smoke'ta 36→37 ID, takım doğruluğu aynı.
Tüm hat dev'de 148 sn (önce ~15 dk, ama o ölçümde GPU paylaşılıyordu).
Ders: ölçüm yaparken başka iş paralel çalışmasın.

**Tespit referansı** (`data/tespit_ref_dev.json`, `16_tespit_referans.py`): dev'den
ölü top dışı 16 kare, 128 **kesin** oyuncu kutusu (yolo11x+v8x@1920 adaylarından,
gözle). Kalabalık uzak kümelerde belirsizler hariç → kolay örneğe yanlı.
Ölçüm `17_tespit_olc.py`: ref recall (yakın/uzak), tüm videoda sınırlı oyuncu
min(k,5)+min(l,5), fazla kutu/kare, takımlar arası kesim, ID.

**Model / imgsz** (hepsi fp16, ms/kare dev döngüsü):
```
                 ms    ref   uzak  oyuncu/10
v8m  1280       21.5   %87   %77   8.21
v8m  1920       38.4   %88   %79   8.43
v8l  1280       32.4   %87   %77   8.35
11l  1280       25.8   %88   %80   8.48   <- seçildi (maliyet/fayda)
11x  1280       41.6   %89   %81   8.54
11l  1920       48.6   %89   %80   8.61
```
Fark küçük (128'de en fazla 3 oyuncu). Uzak oyuncuların ham tespiti ≥0.08'de %89
(v8m) ama izlerde %77: kayıp takipçide. v8m'de uzak oyuncunun ≥0.55 (new_track)
oranı %70, 11l'de %81 → 11l açığı kendiliğinden kapattı.

**new_track_thresh** 0.45 / 0.35 (11l): uzak %80→%81, +21/+58 ID → **0.55 kaldı**.

**Takım kısıtlı takip** (`takim_tracker.py`): BYTETracker alt sınıfı; tespitin
güçlü takım etiketi (`takim_renk.guclu_takim`: kırmızı / lacivert / takım dışı /
bilinmiyor) `cls` alanında; izin takımı son 30 güçlü etiketin çoğunluğu (≥3 oy);
çelişen çift maliyeti 1 (eşleşmez), birinci ve ikinci turda. Taban aynı yoldan
kısıt kapalı (`--kisitsiz`) — predict+BYTETracker yolu model.track ile aynı
eşleşme kararlarını veriyor (koordinatlarda %1.8 kayıtta ≤22 px fark, kaynağı
ultralytics iç detayı, eşleşmeye etkisi yok).
```
                  kesim  fazla/kare  ID   ref/uzak/oyuncu
taban (11l)        211     0.160    141   %88/%80/8.48
kırmızı/lacivert   129     0.131    189   %88/%80/8.46
+ takım dışı       104     0.123    226   %88/%80/8.43   <- varsayılan
```
Gözle 10'ar rastgele kesim: oyuncu→rakip oyuncu geçişi taban 4/10, iki sınıf 1/10,
üç sınıf 1/10 (+2 belirsiz); hakeme geçiş iki sınıfta 2/10, üç sınıfta 0/10.
Kaba tahmin: oyuncu→rakip ~85 → ~10–30 (3 dk). Bedel: ID artıyor (iz kopuyor) —
takım metriklerini etkilemiyor. Sıçrama/mesafe değişmedi (beklenen: geçiş
kapanmada olur, sıçrama üretmez). Kalan kesimlerin çoğu uzak kenarda koyu giyimli
bir kişinin lacivert/takım dışı titremesi (sınıflandırma, takip değil).
Takım doğruluğu tüm deneylerde %97.

## Takım mesafesi (18_mesafe.py) — ölçüldü

Girdi `izler_saha.json`. Canlı oyun / ölü top ayrı. Asıl ölçü **oyuncu başına m/dk**
(= mesafe / izlenen oyuncu-dakika): toplam metre eksik tespit yüzünden düşük kalır
(kare başına ~4.2–4.7/5 oyuncu). Varsayım: izlenemeyen oyuncu da benzer hızda.
Hız bölgeleri 2/4/6 m/s, hız 1 sn'lik ortalanmış yer değiştirmeden; iz kenarında
pencere ±3 kareden kısaysa "belirsiz" (önce kare-kare adım×30'a düşüyordu → ölü
topta kırmızı mesafenin %38'i "sprint" çıkıyordu; hata düzeltildi).

dev (yolo11l + takım kısıtı, 3.1 dk, canlı 2.6 dk):
```
            canlı m/dk/oyuncu   ölü top   tam takım tahmini   bölgeler (canlı)
kırmızı           180              86          2559 m        yürüyüş 34 tempo 31 koşu 17 sprint 6 belirsiz 12
lacivert          149              53          2077 m        yürüyüş 38 tempo 34 koşu 13 sprint 5 belirsiz 10
gürültü tabanı: ölü topta medyan hız 0.28 m/s
```
**Geçerlilik kontrolü (çıktıda kalıcı):** canlı oyunda yakın (kutu ≥90 px) / uzak
oyuncu m/dk: kırmızı 169/186, lacivert 141/158 → uzak **~%10–12 yüksek, iki takımda
aynı** = uzak sahanın çözünürlük gürültüsü mesafeyi bu kadar şişiriyor. Takım farkı
(~28 m/dk) hem yakında hem uzakta duruyor → gürültü değil gibi; ground truth yok,
"kanıt" denmeyecek. Literatür ~110–140 m/dk; değerlerimiz üst tarafta → mutlak
değer değil, **takımlar arası karşılaştırma** olarak sunulmalı.

## Denendi, elendi

- **COCO `sports ball` ile top tespiti** — 8 farklı imgsz/conf kombinasyonunda
  sıfır tespit. COCO örnekleri yakın plan toplar; buradaki top ~20 px.
  Sayı sayımı bu yüzden kapsam dışı.
- **BoT-SORT + ReID** — 36 → 39 ID, iyileşme yok. Sebep: ReID "insanlar farklı
  görünür" varsayımına dayanıyor. Üniformalı sporda aynı takımın oyuncuları
  görsel olarak ayırt edilemez. ReID en fazla takım ayırt edebilir, oyuncu
  ayırt edemez. (Negatif sonuç, rapora yazılacak.)
- **Hareket kapısı** (yumuşatmadan önce iz kesme; metre eşiği 0.6 m/kare ve kutu
  boyuna oranlı eşik) — 1165 kesim, sıçrama 49→71, gerçek hareket kesildi. Kalan
  ~49 "sıçrama" piksel uzayında normal adımdan ayrışmıyor (adım/kutu boyu medyanı
  0.057, normalle aynı) → uzak sahanın çözünürlük sınırı, kimlik atlaması değil.
  Mesafeye etkisi ~%1.5.
- **new_track_thresh'i düşürmek** (0.45, 0.35; yolo11l ile) — uzak recall +1
  puan, sahte ID +21/+58. yolo11l'e geçiş açığı zaten kapatmıştı.
- **conf eşiğini yükselterek çöp eleme** — gerçek ama kapalı kalmış oyuncuyu
  (conf 0.38) da atıyor. Eleme takipçiye bırakıldı.

---

## Bilinen açık sorunlar

1. **İz parçalanması** — ~12 kişi için 36 ID. Her kişi ortalama 2-3 parçaya
   bölünüyor. Kare 241'deki ID'ler kare 481'de yeni numaralarla görünüyor.
   → Oyuncu başına mesafe metriği bu yüzden **güvenilmez**, vaat edilmeyecek.
2. **Tespit kaybı** — kapanma olmayan karelerde bile ara ara oyuncu kaçıyor
   (kare 481'de ortadaki lacivert oyuncunun kutusu yok).
3. ~~**Saha kenarındaki ayakta duranlar**~~ — takım sınıflandırmada "disarida"
   olarak eleniyor (referansta 25/25 doğru).
5. **Takımlar arası ID değişimi** — 11 izde ölçüldü (yukarıya bak).
4. **Ribaunt yığılması** — pota altında 3-4 oyuncu üst üste binince kutular
   iç içe geçiyor, en zor senaryo.

---

## Teslimat planı

### Verilecek
- `tracked.mp4` — takım renkli kutular, kalıcı ID, hareket izleri
- `tracks.json` — kare başına id/takım/saha koordinatı (asıl değerli çıktı)
- Takım ve bölge bazlı **ısı haritaları**
- Kuşbakışı **minimap** (yan yana görünüm)
- Toplam/takım bazlı mesafe
- `LIMITASYONLAR.md`

### Verilmeyecek (bilinçli kapsam kararı)
- Sayı sayımı — top tespit edilemiyor
- Forma numarası / oyuncu adı — OCR gerekiyor, kapsam dışı
- **Oyuncu başına** mesafe — iz parçalanması ölçüldü, güvenilmez
- Pas / top sahipliği — topa bağlı

Metrikler **takım bazında** verilecek, oyuncu bazında değil. Bu, ölçüme dayalı
bir kapsam kararı; raporda parçalanma oranıyla birlikte gerekçelendirilecek.

---

## Sıradaki adımlar

1. ~~**Takım sınıflandırma**~~ — yapıldı, dev test %97. İz bölme denendi, ana
   hatta alınmadı.
2. ~~**Saha kalibrasyonu**~~ — yapıldı, birini-dışarıda medyan 0.16 m.
3. ~~**Minimap + yumuşatma + ısı haritası + takım mesafesi**~~ — yapıldı.
   Kalan: kırmızı köşe lekesinin kontrolü.
4. **Metrik + rapor** — `final.mp4` (5-8 dk, iki yarı saha da dahil) üzerinde
   son koşu, doğruluk ölçümü, limitasyonlar.

> `final.mp4` seçilirken iki yarı saha da dahil edilecek. Geliştirme yakın yarı
> sahada yapıldı (oyuncular büyük), ama ölçüm kolay senaryoda yapılırsa sahte
> olur. Uzak sahada doğruluğun düştüğü raporda açıkça yazılacak.

---

## Çalışma notları

- Uzun süren işlerde **ana sonucu önce diske yaz**, yan işleri (ffmpeg dönüşümü
  gibi) sonra ve `try/except` içinde dene. Kozmetik bir adım 1 saatlik işi
  öldürmemeli.
- OpenCV `VideoWriter` mp4v ile yazıyor; Windows'un yerleşik oynatıcısı açamıyor.
  İzlemek için H.264'e dönüştür: `-c:v libx264 -crf 20 -pix_fmt yuv420p`
- Commit mesajları ne/neden söylesin: "imgsz 1280'e çıkarıldı, uzak oyuncular
  11'den 19'a" gibi. "update" / "fix" yazma.
- Geliştirme `smoke.mp4` (20 sn) üzerinde, doğrulama `dev.mp4` (3 dk) üzerinde,
  sunum `final.mp4` üzerinde. 2 saatlik dosyaya script yöneltme.
