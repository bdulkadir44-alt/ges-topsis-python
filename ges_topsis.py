# -*- coding: utf-8 -*-
"""
GES Arazi Seçiminde TOPSIS Uygulaması (Malatya, 7 alternatif x 4 kriter)

Bu betik makaledeki Tablo 1, Tablo 2, Formül (1)-(11), Tablo 3 ve Şekil 1
sonuçlarını yeniden üretir ve makalede yayımlanan değerlerle karşılaştırır.

Gereksinimler : Python 3.9+, numpy, pandas, matplotlib
Kurulum       : pip install numpy pandas matplotlib
Çalıştırma    : python ges_topsis.py
Çıktılar      : betiğin bulunduğu klasöre
                sekil1_ci_siralama.png, topsis_sonuc.csv, topsis_duyarlilik.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # ekransız ortamda da çalışsın
import matplotlib.pyplot as plt

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

CIKTI_KLASORU = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# ---------------------------------------------------------------
# 1) VERİ (Tablo 1 ve Tablo 2)
# ---------------------------------------------------------------
alternatifler = [f"GES-{i}" for i in range(1, 8)]

# Kriter sırası Tablo 1 / Tablo 2 ile aynıdır:
#   1) Arazinin yıllık ortalama güneşlenme süresi (saat)      -> fayda
#   2) Dağıtım şebekesine olan uzaklık (km)                   -> maliyet (1/x ile fayda yapılır)
#   3) Panel dizileri arasındaki mesafe (metre)               -> fayda
#   4) Panel verimi (oran; Tablo 2'de % olarak verilmiştir)   -> fayda
kriterler = ["Güneşlenme", "Şebeke Uzaklığı", "Dizi Mesafesi", "Panel Verimi"]

# Tablo 2
ham = np.array([
    [7.80, 0.30, 6.00, 0.24],   # GES-1
    [7.80, 0.50, 4.70, 0.24],   # GES-2
    [7.80, 0.70, 5.20, 0.24],   # GES-3
    [7.50, 0.70, 4.00, 0.24],   # GES-4
    [7.50, 1.00, 5.00, 0.18],   # GES-5
    [7.50, 0.90, 4.20, 0.24],   # GES-6
    [7.50, 1.50, 4.00, 0.18],   # GES-7
], dtype=float)

# Tablo 1 (%): güneşlenme, şebeke uzaklığı, dizi mesafesi, panel verimi
w = np.array([56.74, 9.71, 9.18, 24.37])
assert abs(w.sum() - 100.0) < 1e-9, "Ağırlıklar toplamı 100 olmalı"

# Makalede "mevcut üretim miktarı" karşılaştırması için kullanılan yıllık
# üretim değerleri (kWh x 1000). TOPSIS hesabına GİRMEZ; yalnızca 6. ve 7.
# bölümdeki sıralama karşılaştırması içindir.
yillik_uretim = np.array([1987, 1842, 1900, 1795, 1678, 1725, 1580], dtype=float)


# ---------------------------------------------------------------
# 2) TOPSIS FONKSİYONU
# ---------------------------------------------------------------
def topsis(veri, agirlik, fayda):
    """Vektör normalizasyonlu klasik TOPSIS.

    veri    : (m x n) karar matrisi
    agirlik : (n,) ağırlık vektörü
    fayda   : (n,) bool; True = fayda, False = maliyet kriteri
    """
    R = veri / np.sqrt((veri ** 2).sum(axis=0))                 # Formül 2-3
    V = R * agirlik                                             # Formül 4
    a_poz = np.where(fayda, V.max(axis=0), V.min(axis=0))       # Formül 5
    a_neg = np.where(fayda, V.min(axis=0), V.max(axis=0))       # Formül 6
    s_poz = np.sqrt(((V - a_poz) ** 2).sum(axis=1))             # Formül 7
    s_neg = np.sqrt(((V - a_neg) ** 2).sum(axis=1))             # Formül 8
    c = s_neg / (s_poz + s_neg)                                 # Formül 11
    return R, V, a_poz, a_neg, s_poz, s_neg, c


# ---------------------------------------------------------------
# 3) ANA ANALİZ: şebeke uzaklığına ters orantı (1/x) dönüşümü
# ---------------------------------------------------------------
A = ham.copy()
A[:, 1] = 1.0 / A[:, 1]                       # Formül 1
fayda = np.ones(4, dtype=bool)                # 1/x sonrası tüm kriterler fayda
R, V, a_poz, a_neg, s_poz, s_neg, c = topsis(A, w, fayda)

print("A matrisi (Formül 1):")
print(pd.DataFrame(A, index=alternatifler, columns=kriterler))
print("\nR matrisi (Formül 3):")
print(pd.DataFrame(R, index=alternatifler, columns=kriterler))
print("\nV matrisi (Formül 4):")
print(pd.DataFrame(V, index=alternatifler, columns=kriterler))
print("\nTablo 3 - İdeal çözümler:")
print(pd.DataFrame([a_poz, a_neg], index=["A+", "A-"], columns=kriterler))

sonuc = pd.DataFrame({"Si+": s_poz, "Si-": s_neg, "Ci": c}, index=alternatifler)
sonuc["Sıra"] = sonuc["Ci"].rank(ascending=False, method="min").astype(int)
print("\nSonuç tablosu (Formül 9-11):")
print(sonuc.sort_values("Sıra"))

# ---------------------------------------------------------------
# 4) DOĞRULAMA: makalede yayımlanan değerlerle karşılaştırma
# ---------------------------------------------------------------
# Makaledeki yuvarlanmış değerler (Formül 1, 3, 4, 9, 10, Tablo 3, Şekil 1)
makale_A = np.array([
    [7.80, 3.333, 6.00, 0.24],
    [7.80, 2.000, 4.70, 0.24],
    [7.80, 1.429, 5.20, 0.24],
    [7.50, 1.429, 4.00, 0.24],
    [7.50, 1.000, 5.00, 0.18],
    [7.50, 1.111, 4.20, 0.24],
    [7.50, 0.667, 4.00, 0.18],
])
makale_R = np.array([
    [0.3864, 0.7127, 0.4747, 0.4041],
    [0.3864, 0.4276, 0.3718, 0.4041],
    [0.3864, 0.3055, 0.4114, 0.4041],
    [0.3715, 0.3055, 0.3165, 0.4041],
    [0.3715, 0.2138, 0.3956, 0.3030],
    [0.3715, 0.2376, 0.3323, 0.4041],
    [0.3715, 0.1425, 0.3165, 0.3030],
])
makale_V = np.array([
    [21.9235, 6.9208, 4.3576, 9.8470],
    [21.9235, 4.1525, 3.4134, 9.8470],
    [21.9235, 2.9661, 3.7766, 9.8470],
    [21.0803, 2.9661, 2.9051, 9.8470],
    [21.0803, 2.0762, 3.6313, 7.3852],
    [21.0803, 2.3069, 3.0503, 9.8470],
    [21.0803, 1.3842, 2.9051, 7.3852],
])
makale_Apoz = np.array([21.9235, 6.9208, 4.3576, 9.8470])        # Tablo 3
makale_Aneg = np.array([21.0803, 1.3842, 2.9052, 7.3852])        # Tablo 3 (basıldığı gibi)
makale_Spoz = np.array([0.0000, 2.9249, 3.9972, 4.2966, 5.5469, 4.8691, 6.2877])  # Formül 9
makale_Sneg = np.array([6.2877, 3.8332, 3.1675, 2.9262, 1.0032, 2.6330, 0.0000])  # Formül 10
makale_C = np.array([1.000, 0.567, 0.442, 0.405, 0.153, 0.351, 0.000])            # Şekil 1
makale_sira = ["GES-1", "GES-2", "GES-3", "GES-4", "GES-6", "GES-5", "GES-7"]
makale_uretim_sira = ["GES-1", "GES-3", "GES-2", "GES-4", "GES-6", "GES-5", "GES-7"]


def yuvarlak_uyumlu(hesap, yayim, basamak, ek_tol=0.0):
    """Hesaplanan değer, makaledeki 'basamak' ondalığa yuvarlanmış değerle
    yuvarlama payı (yarım son basamak) içinde uyuşuyor mu?"""
    tol = 0.5 * 10 ** (-basamak) + 1e-9 + ek_tol
    return bool(np.all(np.abs(np.asarray(hesap) - np.asarray(yayim)) <= tol))


sira_hesap = list(sonuc.sort_values("Ci", ascending=False).index)
uretim_sira = [alternatifler[i] for i in np.argsort(-yillik_uretim)]

kontroller = {
    "A matrisi (Formül 1)": yuvarlak_uyumlu(A, makale_A, 3),
    "R matrisi (Formül 3)": yuvarlak_uyumlu(R, makale_R, 4),
    "V matrisi (Formül 4)": yuvarlak_uyumlu(V, makale_V, 4),
    "A+ (Tablo 3)": yuvarlak_uyumlu(a_poz, makale_Apoz, 4),
    "A- (Tablo 3)": yuvarlak_uyumlu(a_neg, makale_Aneg, 4, ek_tol=1e-4),
    "Si+ (Formül 9)": yuvarlak_uyumlu(s_poz, makale_Spoz, 4),
    "Si- (Formül 10)": yuvarlak_uyumlu(s_neg, makale_Sneg, 4),
    "Ci (Şekil 1)": yuvarlak_uyumlu(c, makale_C, 3),
    "TOPSIS sıralaması": sira_hesap == makale_sira,
    "Üretim sıralaması": uretim_sira == makale_uretim_sira,
}

print("\nDOĞRULAMA (makale değerleri ile karşılaştırma):")
for ad, tamam in kontroller.items():
    print(f"  {ad:<22}: {'UYUMLU' if tamam else 'FARKLI'}")

# Tablo 3'teki A- üçüncü sütun notu: makalede 2,9052 basılmış, ancak V matrisindeki
# (Formül 4) karşılığı 2,9051'dir; hesaplanan değer aşağıda gösterilir.
print(f"\nNot: A- (3. sütun) hesaplanan = {a_neg[2]:.5f}; "
      f"makalede Tablo 3'te {makale_Aneg[2]:.4f}, Formül 4'te {makale_V[3, 2]:.4f}.")

# TOPSIS sıralaması ile üretim sıralaması farkı (6. ve 7. bölüm)
uretim_sirasi_no = pd.Series(yillik_uretim, index=alternatifler).rank(ascending=False, method="min").astype(int)
karsilastirma = pd.DataFrame({
    "TOPSIS Ci": c,
    "TOPSIS sırası": sonuc["Sıra"],
    "Yıllık üretim": yillik_uretim,
    "Üretim sırası": uretim_sirasi_no,
})
print("\nTOPSIS ve mevcut üretim sıralamalarının karşılaştırılması:")
print(karsilastirma)
yer_degistiren = list(karsilastirma.index[karsilastirma["TOPSIS sırası"] != karsilastirma["Üretim sırası"]])
print("Sıralaması değişen lokasyonlar:", yer_degistiren)
kontroller["Yer değiştirenler (GES-2, GES-3)"] = yer_degistiren == ["GES-2", "GES-3"]

if not all(kontroller.values()):
    basarisiz = [ad for ad, ok in kontroller.items() if not ok]
    raise SystemExit(f"HATA: makale değerleriyle uyuşmazlık: {basarisiz}")
print("\nTüm kontroller başarılı: sonuçlar makaledeki değerlerle uyumludur.")

# ---------------------------------------------------------------
# 5) DUYARLILIK KONTROLÜ (isteğe bağlı, makalede yoktur)
# (a) Şebeke uzaklığı 1/x yerine doğrudan MALİYET kriteri
# (b) Eşit ağırlıklar
# ---------------------------------------------------------------
fayda_b = np.array([True, False, True, True])
c_maliyet = topsis(ham, w, fayda_b)[-1]
c_esit = topsis(A, np.full(4, 25.0), fayda)[-1]
duyarlilik = pd.DataFrame({
    "Ana (1/x)": c,
    "Maliyet kriteri": c_maliyet,
    "Eşit ağırlık": c_esit,
}, index=alternatifler)
print("\nDuyarlılık - Ci değerleri:")
print(duyarlilik)
print("\nDuyarlılık - Sıralamalar:")
print(duyarlilik.rank(ascending=False, method="min").astype(int))

# ---------------------------------------------------------------
# 6) ŞEKİL 1 VE ÇIKTI DOSYALARI
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.5))
cubuklar = ax.bar(alternatifler, c, color="#2E75B6")
ax.bar_label(cubuklar, labels=[f"{v:.3f}".replace(".", ",") for v in c], padding=3)
ax.set_xlabel("GES LOKASYONLARI")
ax.set_ylabel("KARAR DEĞERLERİ (Ci)")
ax.set_title("Şekil 1. GES Lokasyonu Karar Değerleri")
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(CIKTI_KLASORU / "sekil1_ci_siralama.png", dpi=200)
plt.close(fig)

sonuc.to_csv(CIKTI_KLASORU / "topsis_sonuc.csv", encoding="utf-8-sig")
duyarlilik.to_csv(CIKTI_KLASORU / "topsis_duyarlilik.csv", encoding="utf-8-sig")
print("\nDosyalar kaydedildi: sekil1_ci_siralama.png, topsis_sonuc.csv, topsis_duyarlilik.csv")
