# 📊 Analisis Program Pemilahan Sampah Berbasis Data & AI

> Dokumentasi lengkap **Data Analyst & AI Engineer** dalam program pemilahan sampah berbasis reward di masyarakat di desa Sidowayah.

---

## 📋 Daftar Isi

1. [Gambaran Umum Program](#1-gambaran-umum-program)
2. [Peran & Tanggung Jawab](#2-peran--tanggung-jawab)
3. [Struktur Data](#3-struktur-data)
4. [Alur Kerja Lengkap](#4-alur-kerja-lengkap)
5. [Tahap 1 — Analisis Survei Pertama (Pre-Program)](#5-tahap-1--analisis-survei-pertama-pre-program)
6. [Tahap 2 — Analisis Data Pengambilan Sampah per Minggu](#6-tahap-2--analisis-data-pengambilan-sampah-per-minggu)
7. [Tahap 3 — Analisis Survei Kedua (Post-Program)](#7-tahap-3--analisis-survei-kedua-post-program)
8. [Tahap 4 — Integrasi Data & Kesimpulan](#8-tahap-4--integrasi-data--kesimpulan)
9. [Strategi Data Sintetis](#9-strategi-data-sintetis)
10. [Evaluasi Model](#10-evaluasi-model)
11. [Output & Deliverables](#11-output--deliverables)
12. [Etika Data](#12-etika-data)
13. [Keterbatasan & Catatan Ilmiah](#13-keterbatasan--catatan-ilmiah)
14. [Struktur Direktori Proyek](#14-struktur-direktori-proyek)
15. [Teknologi yang Digunakan](#15-teknologi-yang-digunakan)

---

## 1. Gambaran Umum Program

Program ini merupakan salah satu program kerja dari kelompok MBKM di desa Sidowayah untuk mendorong perilaku pemilahan sampah di masyarakat melalui sistem **reward** bagi warga yang berhasil memilah sampah dengan baik.

### Tujuan Program
- Meningkatkan kesadaran dan perilaku pemilahan sampah di masyarakat
- Mengevaluasi efektivitas reward sebagai pendorong perubahan perilaku
- Mengidentifikasi faktor-faktor yang paling memengaruhi keberhasilan pemilahan sampah

### Pertanyaan Utama yang Harus Dijawab
1. **Apakah program ini berhasil?**
2. **Apa pengaruh program terhadap perilaku pemilahan sampah warga?**
3. **Berapa banyak orang yang sudah bisa memilah sampah setelah program berjalan?**
4. **Faktor apa yang paling menentukan keberhasilan pemilahan sampah?**
5. **Apakah reward mendorong perilaku warga?**

---

## 2. Peran & Tanggung Jawab

Sebagai penanggung jawab **pengolahan data dan pembuatan model AI**, berikut cakupan peran secara lengkap:

### 2.1 Sebagai Data Analyst
| Tanggung Jawab | Deskripsi |
|---|---|
| **Data Collection Review** | Memastikan data survei dan pantauan terkumpul dengan benar |
| **Data Cleaning** | Menangani missing values, duplikat, outlier, dan inkonsistensi data |
| **EDA (Exploratory Data Analysis)** | Mengeksplorasi distribusi, korelasi, dan pola tersembunyi dalam data |
| **Feature Engineering** | Membuat variabel baru yang lebih informatif dari data mentah |
| **Statistical Testing** | Melakukan uji statistik untuk memvalidasi temuan secara ilmiah |
| **Visualisasi** | Menyajikan temuan dalam bentuk grafik dan visualisasi yang komunikatif |

### 2.2 Sebagai AI Engineer
| Tanggung Jawab | Deskripsi |
|---|---|
| **Label Engineering** | Mendefinisikan dan membuat label target secara metodologis (untuk data sintetis) |
| **Synthetic Data Generation** | Membuat data sintetis yang representatif berbasis distribusi data asli |
| **Model Training** | Melatih model klasifikasi menggunakan data sintetis |
| **Model Inference** | Menjalankan model pada 42 data asli sebagai inference |
| **Model Evaluation** | Mengevaluasi performa model secara menyeluruh |
| **Feature Importance Analysis** | Mengidentifikasi faktor paling berpengaruh dari model |

### 2.3 Sebagai Komunikator Data
| Tanggung Jawab | Deskripsi |
|---|---|
| **Interpretasi Hasil** | Menerjemahkan output model ke dalam bahasa yang dapat dipahami stakeholder (kelompok MBKM, LPJ, warga desa Sidowayah) |
| **Rekomendasi** | Memberikan rekomendasi berbasis data untuk program selanjutnya kepada pengelola desa Sidowayah |
| **Dokumentasi** | Mendokumentasikan seluruh proses agar dapat direproduksi |
| **Transparansi** | Menyampaikan keterbatasan dan asumsi analisis |

---

## 3. Struktur Data

### 3.1 Data Survei Pertama (Pre-Program)
- **Jumlah responden:** 42 orang
- **Waktu pengumpulan:** Sebelum program berjalan
- **Konten:** Pengetahuan, sikap, perilaku dan kebiasaan, dan hambatan
- **Tipe jawaban:** Skala Likert 1–5 dan Ya/Tidak
- **Variabel reward:** Belum ada
- **Peran dalam analisis:** EDA + Inference (bukan training)

**Kategori pertanyaan:**

| Kategori | Deskripsi | Tipe Jawaban |
|---|---|---|
| **Pengetahuan** | Seberapa tahu warga tentang pemilahan sampah | Skala 1–5 |
| **Sikap** | Seberapa peduli warga terhadap isu sampah | Skala 1–5 |
| **perilaku dan kebiasaan** | Kebiasaan warga dalam kegiatan yang berkaitan dengan memilah sampah | Skala 1–5 |
| **hambatan** | Apa hambatan yang selama ini dirasakan warga terkait sampah | Ya-Tidak |

### 3.2 Data Keberlangsungan (Minggu 1–3)
- **Jumlah responden:** 42 orang (orang yang sama)
- **Periode:** Minggu 1, 2, dan 3 setelah program berjalan
- **Konten:** Apakah warga berhasil memilah dan mendapatkan reward
- **Peran dalam analisis:** EDA tren + integrasi dengan data survei

### 3.3 Data Survei Kedua (Post-Program)
- **Jumlah responden:** 42 orang (orang yang sama)
- **Waktu pengumpulan:** Setelah program berjalan
- **Konten:** Sama dengan survei pertama + pertanyaan terkait reward
- **Variabel baru:** Persepsi dan pengalaman terhadap reward
- **Peran dalam analisis:** EDA + Pre-Post Comparison + Inference

---

## 4. Alur Kerja Lengkap

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ALUR KERJA ANALISIS                          │
└─────────────────────────────────────────────────────────────────────┘

DATA SURVEI 1 (Pre-Program)
        │
        ├──► [TAHAP 1A] EDA per kategori
        │         └── distribusi, korelasi, kualitas data
        │
        ├──► [TAHAP 1B] Pembuatan Label
        │         └── composite scoring (equal weighting) → threshold
        │
        ├──► [TAHAP 1C] Generate Data Sintetis
        │         └── CTGAN/Gaussian Copula berbasis distribusi asli
        │
        ├──► [TAHAP 1D] Training Model Klasifikasi
        │         └── input: data sintetis + label buatan
        │
        └──► [TAHAP 1E] Inference pada 42 Data Asli
                  └── output: label "sudah memilah" / "belum memilah" per orang

DATA KEBERLANGSUNGAN (Minggu 1–3)
        │
        ├──► [TAHAP 2A] EDA Tren Mingguan
        │         └── siapa yang melakukan pemilahan, siapa yang konsisten, siapa yang dropout
        │
        └──► [TAHAP 2B] Integrasi dengan Hasil Tahap 1
                  └── apakah yang "memilah" benar-benar berhasil memilah?

DATA SURVEI 2 (Post-Program)
        │
        ├──► [TAHAP 3A] EDA + Perbandingan dengan Survei 1
        │
        ├──► [TAHAP 3B] Analisis Pre-Post (Uji Statistik)
        │
        ├──► [TAHAP 3C] Generate Sintetis → Training → Inference
        │
        └──► [TAHAP 3D] Analisis Pengaruh Reward

INTEGRASI SEMUA DATA
        │
        ├──► [TAHAP 4A] Jawab 5 Pertanyaan Utama Program
        │
        ├──► [TAHAP 4B] Feature Importance Analysis
        │         └── kategori mana yang paling menentukan keberhasilan?
        │
        ├──► [TAHAP 4C] Proyeksi Minggu ke Depan untuk Program Selanjutnya
        │
        └──► [TAHAP 4D] Rekomendasi untuk Program Selanjutnya
```

---

## 5. Tahap 1 — Analisis Survei Pertama (Pre-Program)

### 5A. Exploratory Data Analysis (EDA)

**Tujuan:** Memahami kondisi awal pengetahuan dan perilaku warga sebelum program berjalan.

**Langkah-langkah:**

```python
# 1. Cek kualitas data
- Jumlah missing values per kolom
- Identifikasi duplikat
- Cek tipe data tiap kolom

# 2. Statistik deskriptif
- Mean, median, std, min, max per pertanyaan
- Distribusi jawaban per kategori (Pengetahuan, Sikap, Perilaku dan Kebiasaan, Hambatan)

# 3. Visualisasi distribusi
- Histogram tiap pertanyaan
- Boxplot per kategori
- Heatmap korelasi antar pertanyaan

# 4. Analisis per kategori
- Skor rata-rata tiap kategori
- Kategori mana yang paling tinggi/rendah secara rata-rata
- Apakah ada kategori yang sangat bervariasi (std tinggi)?

# 5. Analisis demografis (jika tersedia)
- Distribusi usia, pendidikan, dll.
- Apakah ada perbedaan skor berdasarkan demografi?
```

**Output EDA:**
- Ringkasan kualitas data
- Visualisasi distribusi per kategori
- Insight awal tentang kondisi warga sebelum program

---

### 5B. Pembuatan Label

**Tujuan:** Membuat label target (sudah memilah / belum memilah) secara metodologis karena data asli tidak memiliki label.

**Pendekatan: Composite Scoring dengan Equal Weighting**

```python
# Step 1: Normalisasi semua pertanyaan ke skala 0-1
# Step 2: Hitung skor rata-rata per kategori
skor_pengetahuan         = rata_rata(semua pertanyaan kategori pengetahuan)
skor_sikap               = rata_rata(semua pertanyaan kategori sikap)
skor_perilaku_kebiasaan  = rata_rata(semua pertanyaan kategori perilaku dan kebiasaan)
skor_hambatan            = rata_rata(semua pertanyaan kategori hambatan)

# Step 3: Hitung skor total (equal weighting)
skor_total = rata_rata(skor_pengetahuan, skor_sikap,
                       skor_perilaku_kebiasaan, skor_hambatan)

# Step 4: Tentukan threshold
# Gunakan median skor total sebagai threshold awal
# atau gunakan 60% dari skor maksimum sebagai ambang batas "sudah memilah"
label = "sudah memilah" jika skor_total >= threshold
label = "belum memilah" jika skor_total <  threshold
```

> **Catatan:** Threshold ini akan divalidasi dan mungkin direvisi setelah melihat distribusi hasilnya. Pendekatan equal weighting digunakan di awal karena belum ada bukti empiris kategori mana yang paling menentukan — hasil analisis nantinya yang akan menjawab ini.

---

### 5C. Generate Data Sintetis

**Tujuan:** Membuat data latih yang cukup untuk melatih model klasifikasi, karena 42 data asli terlalu sedikit untuk training.

**Metode yang digunakan:**

| Metode | Kelebihan | Kapan Digunakan |
|---|---|---|
| **CTGAN** | Menangkap distribusi kompleks dan korelasi antar fitur | Data dengan campuran numerik dan kategorikal |
| **Gaussian Copula** | Sederhana, mempertahankan korelasi antar variabel | Data yang mendekati distribusi normal |
| **SMOTE** | Cepat, untuk menangani imbalanced class | Jika satu kelas jauh lebih sedikit dari kelas lain |

**Prosedur:**

```python
# Step 1: Fit generator pada 42 data asli
generator.fit(data_asli_42)

# Step 2: Generate data sintetis
# Jumlah rekomendasi: 500–1000 sampel
data_sintetis = generator.sample(n=1000)

# Step 3: Validasi data sintetis
# Bandingkan distribusi data sintetis vs data asli
# Cek apakah korelasi antar fitur terjaga
# Visualisasi: overlay histogram asli vs sintetis

# Step 4: Assign label ke data sintetis
# Gunakan metode composite scoring yang sama dengan step 5B
data_sintetis['label'] = composite_scoring(data_sintetis)
```

> ⚠️ **Penting:** Data sintetis hanya digunakan untuk **training model**. Data asli (42 orang) **tidak boleh masuk ke training set** — murni untuk inference.

---

### 5D. Training Model Klasifikasi

**Input:** Data sintetis (500–1000 sampel) + label hasil composite scoring

**Model yang dilatih dan dibandingkan:**

| Model | Alasan Dipilih |
|---|---|
| **Random Forest** | Akurasi tinggi, menghasilkan feature importance |
| **Gradient Boosting (XGBoost)** | Performa terbaik untuk data tabular |

**Prosedur Training:**

```python
# Split data sintetis: 80% train, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    data_sintetis, labels, test_size=0.2, stratify=labels
)

# Training semua model
for model in [RandomForest, XGBoost]:
    model.fit(X_train, y_train)
    evaluasi(model, X_val, y_val)

# Pilih model terbaik berdasarkan F1-Score
model_terbaik = pilih_berdasarkan_f1(hasil_evaluasi)
```

---

### 5E. Inference pada 42 Data Asli

```python
# Jalankan model terbaik pada 42 data asli
hasil_inference = model_terbaik.predict(data_asli_42)
probabilitas    = model_terbaik.predict_proba(data_asli_42)

# Output per orang:
# - Label: "sudah memilah" atau "belum memilah"
# - Probabilitas kepercayaan model
# - Skor per kategori (Pengetahuan, Sikap, Perilaku dan Kebiasaan, Hambatan)
```

---

## 6. Tahap 2 — Analisis Data Keberlangsungan

### 6A. EDA Tren Mingguan

**Tujuan:** Memahami pola keberhasilan pemilahan sampah selama 3 minggu program berjalan.

**Analisis yang dilakukan:**

```python
# 1. Berapa orang yang berhasil memilah tiap minggu?
tren_mingguan = data.groupby('minggu')['berhasil'].sum()

# 2. Siapa yang konsisten berhasil (minggu 1, 2, dan 3)?
konsisten = data[data['minggu_1'] == 1 & data['minggu_2'] == 1 & data['minggu_3'] == 1]

# 3. Siapa yang dropout (berhasil minggu 1 tapi tidak minggu 2/3)?
dropout = identifikasi_pola_dropout(data)

# 4. Visualisasi tren
plot_tren_mingguan(tren_mingguan)
plot_sankey_perubahan_status(data)  # alur perubahan status tiap minggu
```

---

### 6B. Integrasi dengan Hasil Survei 1

**Tujuan:** Memvalidasi apakah hasil klasifikasi survei 1 sesuai dengan perilaku nyata warga.

```python
# Gabungkan hasil inference survei 1 dengan data keberlangsungan
data_gabungan = merge(hasil_inference_survei1, data_keberlangsungan, on='id_responden')

# Pertanyaan yang dijawab:
# 1. Apakah yang "sudah memilah" di survei 1 → benar-benar berhasil memilah?
crosstab(label_survei1, berhasil_memilah)

# 2. Kategori mana yang paling membedakan yang berhasil vs tidak?
analisis_per_kategori_vs_keberhasilan(data_gabungan)
```

---

## 7. Tahap 3 — Analisis Survei Kedua (Post-Program)

### 7A. EDA + Perbandingan dengan Survei 1

```python
# Statistik deskriptif survei 2 (sama seperti survei 1)
eda_survei2(data_survei2)

# Bandingkan rata-rata skor tiap kategori: survei 1 vs survei 2
perbandingan = {
    'Pengetahuan':            [mean_survei1, mean_survei2],
    'Sikap':                  [mean_survei1, mean_survei2],
    'Perilaku dan Kebiasaan': [mean_survei1, mean_survei2],
    'Hambatan':               [mean_survei1, mean_survei2],
}
visualisasi_grouped_bar(perbandingan)
```

---

### 7B. Analisis Pre-Post (Uji Statistik)

**Tujuan:** Membuktikan secara ilmiah bahwa perubahan yang terjadi bukan karena kebetulan.

**Uji yang digunakan:**

| Kondisi Data | Uji yang Dipilih |
|---|---|
| Data berdistribusi normal | **Paired t-test** |
| Data tidak berdistribusi normal | **Wilcoxon Signed-Rank Test** |
| Data kategorikal (Ya/Tidak) | **McNemar's Test** |

```python
# Cek normalitas data
shapiro_wilk_test(skor_survei1, skor_survei2)

# Pilih dan jalankan uji yang sesuai
if normal:
    hasil = paired_ttest(skor_survei1, skor_survei2)
else:
    hasil = wilcoxon_test(skor_survei1, skor_survei2)

# Hitung effect size (seberapa besar perubahan)
cohen_d = hitung_effect_size(skor_survei1, skor_survei2)

# Interpretasi:
# p-value < 0.05 → perubahan signifikan secara statistik
# Cohen's d: 0.2 = kecil, 0.5 = sedang, 0.8 = besar
```

---

### 7C. Generate Sintetis → Training → Inference

Prosedur sama dengan Tahap 1C, 1D, 1E — dilakukan untuk data survei 2.

> **Perbedaan:** Data survei 2 memiliki variabel tambahan terkait reward, sehingga model yang dilatih akan menyertakan variabel tersebut sebagai fitur tambahan.

---

### 7D. Analisis Pengaruh Reward

```python
# Korelasi antara persepsi reward dengan perilaku aktual
korelasi_reward_perilaku = korelasi(
    data_survei2['skor_reward'],
    data_keberlangsungan['berhasil_memilah']
)

# Apakah yang merasa reward menarik → lebih konsisten memilah?
analisis_reward_vs_konsistensi(data_survei2, data_keberlangsungan)
```

---

## 8. Tahap 4 — Integrasi Data & Kesimpulan

### 8A. Jawab 5 Pertanyaan Utama

**Pertanyaan 1: Apakah program ini berhasil?**
```
Sumber data  : Uji statistik pre-post (Tahap 3B) + tren keberlangsungan (Tahap 2A)
Kriteria     : Program dianggap berhasil jika:
               - Terdapat peningkatan skor yang signifikan (p < 0.05) antara survei 1 dan 2
               - Jumlah warga yang berhasil memilah meningkat atau stabil dari minggu 1 ke 3
```

**Pertanyaan 2: Apa pengaruh program terhadap warga?**
```
Sumber data  : Analisis pre-post per kategori + analisis pengaruh reward
Output       : Kategori mana (pengetahuan/sikap/perilaku dan kebiasaan/hambatan) 
               yang paling banyak berubah setelah program
```

**Pertanyaan 3: Berapa yang sudah bisa memilah sampah?**
```
Sumber data  : Inference survei 2 + data keberlangsungan minggu 3
Output       : Jumlah dan persentase warga yang dikategorikan "sudah memilah" 
               dan terbukti memilah sampah secara konsisten
```

**Pertanyaan 4: Apa yang paling menentukan keberhasilan pemilahan sampah?**
```
Sumber data  : Inference survei 2
Output       : Fitur dari kategori mana (pengetahuan/sikap/perilaku dan kebiasaan/hambatan) 
               yang menentukan keberhasilan pemilahan sampah
```

**Pertanyaan 5: Apakah reward mendorong perilaku warga?**
```
Sumber data  : Inference survei 2
Output       : Persentase seberapa besar pengaruh reward
               dalam program pemilahan sampah
```

---

### 8B. Feature Importance Analysis

**Tujuan:** Menjawab — *faktor apa yang paling menentukan keberhasilan pemilahan sampah?*

```python
# Ambil feature importance dari Random Forest / XGBoost
feature_importance = model_terbaik.feature_importances_

# Kelompokkan per kategori
importance_per_kategori = {
    'Pengetahuan':            sum(importance untuk fitur pengetahuan),
    'Sikap':                  sum(importance untuk fitur sikap),
    'Perilaku dan Kebiasaan': sum(importance untuk fitur perilaku dan kebiasaan),
    'Hambatan':               sum(importance untuk fitur hambatan),
}

# Visualisasi
plot_bar_importance(importance_per_kategori)
```

**Output:** Ranking kategori berdasarkan kontribusinya terhadap keberhasilan pemilahan sampah.

---

### 8C. Proyeksi Minggu ke Depan

**Pendekatan:** Sederhana dan transparan, mengingat hanya ada 3 titik data (minggu 1–3).

```python
# Opsi 1: Linear Trend
from numpy.polynomial import polynomial as P
trend = P.polyfit(x=[1,2,3], y=jumlah_berhasil_per_minggu, deg=1)

# Opsi 2: Exponential Smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
model_ts = SimpleExpSmoothing(jumlah_berhasil_per_minggu).fit()
proyeksi = model_ts.forecast(4)  # proyeksi 4 minggu ke depan (1 bulan)

# Selalu sertakan confidence interval
# Sampaikan sebagai "proyeksi awal" bukan prediksi pasti
```

> ⚠️ **Catatan Penting:** Proyeksi dengan 3 titik data memiliki uncertainty yang sangat besar. Hasil ini harus disajikan dengan interval kepercayaan yang lebar dan dikomunikasikan sebagai **estimasi awal**, bukan prediksi definitif.

---

### 8D. Rekomendasi untuk Program Selanjutnya

Berdasarkan seluruh temuan, buat rekomendasi berbasis data:

```
┌─────────────────────────────────────────────────────────┐
│              FRAMEWORK REKOMENDASI                       │
├─────────────────────────────────────────────────────────┤
│ 1. Fokus Intervensi                                      │
│    → Kategori dengan importance tertinggi perlu          │
│      mendapat porsi program yang lebih besar             │
│                                                          │
│ 2. Target Sasaran                                        │
│    → Profil warga yang paling sulit berubah              │
│      memerlukan pendekatan berbeda                       │
│                                                          │
│ 3. Desain Reward                                         │
│    → Apakah tipe reward saat ini efektif?                │
│      Atau perlu disesuaikan berdasarkan temuan?          │
│                                                          │
│ 4. Durasi Program                                        │
│    → Berdasarkan tren, berapa lama program               │
│      perlu berjalan agar perubahan perilaku              │
│      menjadi kebiasaan permanen?                         │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Strategi Data Sintetis

### Prinsip Utama

> Data sintetis yang baik adalah data yang **tidak bisa dibedakan secara statistik** dari data asli, bukan data yang dibuat secara acak.

### Proses Pembuatan

```
42 Data Asli
     │
     ├──► Analisis distribusi tiap fitur
     ├──► Analisis korelasi antar fitur
     ├──► Analisis proporsi kelas (imbalanced?)
     │
     ▼
Fit Synthetic Data Generator (CTGAN / Gaussian Copula)
     │
     ▼
Generate 500–1000 Sampel Sintetis
     │
     ▼
Validasi Kualitas Data Sintetis:
     ├──► Kolmogorov-Smirnov Test (distribusi serupa?)
     ├──► Korelasi matrix: asli vs sintetis
     ├──► Visual overlay histogram
     └──► Train-on-synthetic, test-on-real baseline
     │
     ▼
Data Sintetis Siap untuk Training
```

### Ketentuan Penggunaan

| | Data Sintetis | Data Asli (42 orang) |
|---|---|---|
| **Training model** | ✅ Ya | ❌ Tidak |
| **Validasi model** | ✅ Ya (20% split) | ❌ Tidak |
| **Inference** | ❌ Tidak | ✅ Ya |
| **Laporan hasil** |  Harus disebutkan  | ✅ Utama |

---

## 10. Evaluasi Model

### Metrik yang Digunakan

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Prioritaskan F1-Score karena data mungkin tidak seimbang
# Jika kelas tidak seimbang, gunakan F1-macro atau F1-weighted
```

### Interpretasi Confusion Matrix

```
                  Prediksi
                  memilah   Belum memilah
Aktual  memilah  [ TP ]     [ FN ]
        Belum    [ FP ]     [ TN ]

TP = True Positive  → prediksi memilah, aslinya memilah       
TN = True Negative  → prediksi belum memilah, aslinya belum memilah     
FP = False Positive → prediksi memilah, aslinya belum memilah     
FN = False Negative → prediksi belum memilah, aslinya memilah       
```

### Threshold Keberterimaan Model

| Metrik | Minimum Acceptable | Target |
|---|---|---|
| Accuracy | > 70% | > 80% |
| F1-Score | > 0.65 | > 0.75 |
| Precision | > 0.65 | > 0.75 |
| Recall | > 0.65 | > 0.75 |

> Jika model tidak mencapai threshold minimum → review ulang proses pembuatan label dan kualitas data sintetis.

---

## 11. Output & Deliverables

### Dokumen yang Dihasilkan

| Deliverable | Format | Tujuan |
|---|---|---|
| Laporan EDA Survei 1 | Python + Jupyter Notebook + PDF | Dokumentasi teknis |
| Laporan EDA Survei 2 | Python + Jupyter Notebook + PDF | Dokumentasi teknis |
| Analisis Pre-Post | Python + Jupyter Notebook + PDF | Dokumentasi teknis |
| Visualisasi Tren Keberlangsungan | PNG / Dashboard | Presentasi stakeholder |
| Hasil Klasifikasi (per orang) | CSV | Data operasional |
| Feature Importance Report | PDF | Rekomendasi program |
| Proyeksi Minggu ke Depan | PDF + Grafik | Perencanaan program |
| **Laporan Final Terintegrasi** | PDF | Stakeholder utama |

### Format Hasil Inference (per orang)

```csv
id_responden, label_survei1, prob_memilah_s1, skor_pengetahuan, skor_sikap, 
skor_perilaku_kebiasaan, skor_hambatan, skor_total, berhasil_minggu1, 
berhasil_minggu2, berhasil_minggu3, label_survei2, prob_memilah_s2
```

---

## 12. Etika Data

### Prinsip yang Dipegang

- **Anonimisasi:** Identitas responden (nama) tidak masuk ke dalam analisis. Gunakan ID numerik.
- **Transparansi:** Semua asumsi, keterbatasan, dan penggunaan data sintetis **wajib disebutkan** dalam laporan.
- **Tidak Manipulatif:** Threshold label dan parameter model tidak disesuaikan hanya agar hasil terlihat bagus.
- **Kepemilikan Data:** Data responden hanya digunakan untuk tujuan program ini dan tidak dibagikan ke pihak ketiga.
- **Persetujuan:** Pastikan seluruh responden memahami bahwa data mereka digunakan untuk evaluasi program.

---

## 13. Keterbatasan & Catatan Ilmiah

### Keterbatasan yang Harus Dikomunikasikan

| Keterbatasan | Dampak | Mitigasi |
|---|---|---|
| **n = 42** terlalu kecil untuk training | Model mungkin tidak cukup general | Gunakan data sintetis + cross-validation |
| **Data sintetis** bukan data nyata | Hasil model tidak sekuat data asli | Validasi di data asli + sampaikan secara transparan |
| **3 minggu** terlalu singkat untuk time series | Proyeksi memiliki uncertainty besar | Gunakan confidence interval yang lebar |
| **Label dibuat sendiri** dari data survei | Label mungkin tidak sempurna | Validasi dengan domain expert + uji sensitivitas threshold |
| **Survei** rawan bias respons | Jawaban survei ≠ perilaku nyata | Triangulasi dengan data pantauan langsung |

### Cara Mengkomunikasikan Keterbatasan

> Setiap laporan harus menyertakan bagian **"Catatan Metodologis"** yang menjelaskan keterbatasan di atas secara jujur. Hasil analisis adalah **panduan berbasis data** — bukan kebenaran mutlak.

---

## 14. Struktur Direktori Proyek

```
program-pemilahan-sampah/
│
├── README.md                         
│
├── data/
│   ├── raw/
│   │   ├── survei_1_raw.csv           
│   │   ├── keberlangsungan_raw.csv   
│   │   └── survei_2_raw.csv         
│   │
│   ├── processed/
│   │   ├── survei_1_clean.csv      
│   │   ├── keberlangsungan_clean.csv
│   │   └── survei_2_clean.csv
│   │
│   ├── synthetic/
│   │   ├── sintetis_survei_1.csv  
│   │   └── sintetis_survei_2.csv
│   │
│   └── output/
│       ├── hasil_inference_survei1.csv
│       ├── hasil_inference_survei2.csv
│       └── data_integrasi_final.csv
│
├── Python/
│   ├── 01_eda_survei1.py
│   ├── 02_label_engineering.py
│   ├── 03_synthetic_data_generation.py
│   ├── 04_model_training_survei1.py
│   ├── 05_inference_survei1.py
│   ├── 06_eda_keberlangsungan.py
│   ├── 07_eda_survei2.py
│   ├── 08_prepost_analysis.py
│   ├── 09_model_training_survei2.py
│   ├── 10_inference_survei2.py
│   └── 11_integrasi_kesimpulan.py
│
├── models/
│   ├── model_survei1_best.pkl
│   └── model_survei2_best.pkl
│
├── reports/
│   ├── eda_survei1_report.pdf
│   ├── eda_survei2_report.pdf
│   ├── prepost_analysis_report.pdf
│   ├── feature_importance_report.pdf
│   └── laporan_final.pdf
│
└── requirements.txt
```

---

## 15. Teknologi yang Digunakan

```txt
# Core Data Analysis
pandas>=2.0
numpy>=1.24
scipy>=1.10
statsmodels>=0.14

# Machine Learning
scikit-learn>=1.3
xgboost>=2.0
lightgbm>=4.0

# Synthetic Data Generation
sdv>=1.9                    # CTGAN & Gaussian Copula
imbalanced-learn>=0.11      # SMOTE

# Visualization
matplotlib>=3.7
seaborn>=0.12
plotly>=5.15

# Jupyter
jupyter>=1.0
nbformat>=5.9
```

---
