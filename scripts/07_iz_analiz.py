import json, sys
from pathlib import Path
from collections import Counter

# kullanim: python scripts/07_iz_analiz.py dev   -> out/dev/izler.json
AD = sys.argv[1] if len(sys.argv) > 1 else "smoke"
kayit = json.load(open(Path("out") / AD / "izler.json"))
sayac = Counter(k["id"] for k in kayit)
kare_sayisi = max(k["kare"] for k in kayit) + 1

print(f"kare: {kare_sayisi} | toplam ID: {len(sayac)}\n")

for esik in (1, 5, 15, 30, 60, 150, 300):
    n = sum(1 for v in sayac.values() if v >= esik)
    print(f"  >= {esik:>3} kare goruldu : {n:>3} ID")

print("\nen uzun 15 iz:")
for tid, n in sayac.most_common(15):
    print(f"  #{tid:>3}: {n:>4} kare  (%{n / kare_sayisi * 100:.0f})")

per = Counter(k["kare"] for k in kayit)
print(f"\nkare basina ortalama kisi: {sum(per.values()) / kare_sayisi:.1f}")