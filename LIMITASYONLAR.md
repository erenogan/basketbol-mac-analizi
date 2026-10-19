# Sınırlar ve doğruluk

Bu belge, sistemin ne ölçtüğünü, ne kadar doğru ölçtüğünü ve neyi ölçemediğini anlatır.
Buradaki sayılar maçın ikinci devresinden alınan 7 dakikalık sunum klibinde (`final.mp4`)
ölçüldü. Sistemin bütün ayarları maçın birinci devresinden başka bir klipte yapıldı; sunum
klibine hiç ayar yapılmadı. Yani sayılar, sistemin daha önce görmediği görüntüdeki başarısı.

Bütün doğruluk sayıları, görüntüye bakılarak elle hazırlanmış referanslara karşı ölçüldü.
Referanslar etiketlenirken sistemin tahmini gösterilmedi.

## Sistem ne veriyor

| Çıktı | Açıklama |
|---|---|
| `tracked.mp4` | Kamera görüntüsü ile kuşbakışı saha tahtası yan yana; oyuncular takım renginde |
| `tracks.json`, `tracks.csv` | Her karede her oyuncunun takımı ve saha üzerindeki konumu (metre) |
| `isi_haritasi.png` | Her takımın canlı oyunda sahanın neresinde zaman geçirdiği |
| `ozet.json` | Takım bazlı koşu mesafesi, hız bölgeleri, saha bölgesi dağılımı |

## Ölçülen doğruluk

| Ne | Değer | Nasıl ölçüldü |
|---|---|---|
| Takım ayrımı (kırmızı / lacivert / takım dışı) | %93 (yakın saha %100, uzak %89) | 73 rastgele oyuncu kutusu, elle etiketlendi |
| Oyuncu bulma, kameraya yakın saha | %97 | 16 karede 30 oyuncu |
| Oyuncu bulma, uzak saha | %81 | 16 karede 74 oyuncu |
| Saha üzerindeki konum hatası | medyan 0.16 m, en kötü 0.73 m | 15 bilinen saha noktası |

## Sistemin yapamadıkları

**Top görülmüyor.** Top görüntüde yaklaşık 20 piksel; kullanılan modeller bu boyutta topu
bulamadı. Bu yüzden şunlar yok: sayı, pas, top kimde, hücum mu savunma mı.
Isı haritasında hücum ve savunma anları birbirine karışıktır.

**Oyuncu kimliği yok, takım var.** Sistem "bu kırmızı takımdan biri" der, "bu 7 numara" demez.
Forma numarası okunmuyor. Takip numaraları (`iz`) oyuncu kimliği değildir: iki oyuncu üst üste
bindiğinde takip kopar ve oyuncu yeni bir numarayla devam eder. Bu yüzden **oyuncu başına
mesafe verilmiyor**, bütün metrikler takım bazında.

## Bilinmesi gereken sınırlar

**Uzak sahada her 5 oyuncudan 1'i bulunamıyor.** Uzaktaki oyuncular görüntüde 60 piksel
civarında. Kalabalık anlarda (ribauntlar, serbest atış dizilişi) bu oran daha da düşer.
Isı haritasında boyalı alan ve kalabalık bölgeler olduğundan biraz az görünebilir.

**Uzak sahada konum daha belirsiz.** Görüntüde 1 piksel, kameraya yakın sahada yaklaşık 1 cm,
uzak sahada 15 cm ediyor. Konum hatası yakında birkaç santimetre, uzakta 0.5–0.7 metre.

**Koşu mesafesi mutlak değil, karşılaştırmalı okunmalı.** Uzak sahadaki konum oynamaları
mesafeye ekleniyor: aynı sistemde uzaktaki oyuncular yakındakilerden yaklaşık %10 daha hızlı
görünüyor ve bu fark iki takımda da aynı. Bulunan değerler (oyuncu başına dakikada 160–170 m)
basketbol literatüründeki değerlerin üst sınırında. "Kırmızı takım laciverden fazla koştu"
güvenle söylenebilir; "kırmızı takım 2.559 m koştu" kesin bir ölçüm değildir.

**Kenardaki kişiler.** Saha kenarında duran yedek oyuncular ve antrenörler bazen sahadaki
oyuncu sanılabiliyor. Bir takıma kare başına ortalama 0.2 fazladan kişi düşüyor.
Hakemler ve siyah giyen kişiler büyük ölçüde doğru ayıklanıyor.

**Ölü top tahmini.** Top görülmediği için oyunun durduğu anlar, bütün oyuncuların aynı anda
yavaşlamasından çıkarılıyor. Sunum klibinde 12 aralık, toplam 2 dakika ölü top sayıldı. Gözle
bakılınca yarısı kesin ölü top (serbest atışlar, sakatlık duraklaması); kalanların bir kısmı
yavaş tempolu set hücumu olabilir. Bu yüzden "canlı oyun" metrikleri bir miktar yavaş oyunu
dışarıda bırakıyor olabilir.

**Takım renkleri bu maça göre ayarlı.** Renk kuralları bu salonun ışığına ve bu iki takımın
formalarına göre ölçüldü. Başka bir maçta, farklı formalarla ya da farklı bir salonda yeniden
ayarlanması gerekir.

**Kamera sabit olmalı.** Saha koordinatları kameranın hiç oynamadığı varsayımına dayanıyor.
Bu kayıtta kamera 3 dakika boyunca 0.1 pikselden az oynadı. Kamera oynarsa saha yeniden
ölçülmelidir.

## Bilerek kapsam dışı bırakılanlar

- Sayı sayımı, pas ve top sahipliği: top görülmediği için
- Forma numarası ve oyuncu adı: numara okuma ayrı bir sistem gerektirir
- Oyuncu başına mesafe ve ısı haritası: takip kimlikleri oyuncu başına güvenilir değil
