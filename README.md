# Analisis Pemilahan Sampah

Kerangka analisis yang disederhanakan untuk melihat **apakah warga bergerak dari sekadar
paham menjadi benar-benar memilah**, dan bagaimana **reward** (yang baru diperkenalkan di
Survei 2) menyertai perubahan itu. Model dilatih dan diuji
langsung di data asli lewat cross-validation.

## Susunan skrip

| File | Isi |
|---|---|
| `common.py` | Definisi kolom, pembersihan, aturan label, palet warna (diimpor semua skrip) |
| `01_eda_survei1.py` | EDA Survei 1: sebaran jawaban, korelasi, kualitas data |
| `02_pelabelan_survei1.py` | Label `sudah/belum` + tiga tingkat deskriptif |
| `03_model_survei1.py` | Random Forest + leave-one-out CV + Gini importance + inference |
| `04_eda_survei2.py` | EDA Survei 2, termasuk sebaran empat item reward |
| `05_pelabelan_survei2.py` | Label dengan aturan yang sama persis |
| `06_model_survei2.py` | Random Forest + reward + CV + Gini (di sinilah peran reward dinilai) |
| `07_perbandingan.py` | Bandingkan % pemilah S1 vs S2 dan pergeserannya |
| `08_keberlangsungan.py` | Data lapangan: peringkat, status, tren — berdiri sendiri |
| `09_validasi.py` | Cocokkan label survei dengan perilaku nyata di lapangan (uji validitas eksternal) |

## Hasil utama yang diharapkan

Bintang analisis ini adalah angka deskriptif di `07`: **persen warga yang benar-benar
memilah** di Survei 1 versus Survei 2, dan berapa yang berpindah dari belum ke sudah.
Kenaikan ini terjadi berbarengan dengan diperkenalkannya reward. Model + Gini melengkapi
dengan cerita faktor: di Survei 2, seberapa besar peran reward dibanding pengetahuan,
sikap, dan hambatan dalam memprediksi siapa yang memilah.

## Catatan

- **Kelas Survei 1 sangat timpang.** Dengan aturan ketat, hanya sedikit warga yang lolos
  sebagai "sudah memilah" di Survei 1. Model prediksi Survei 1 karena itu **tidak stabil**
  dan sebaiknya dibaca sebagai catatan, bukan temuan. Analisis faktor yang bermakna ada di
  Survei 2 (kelas lebih seimbang, dan reward hanya ada di sana).
- **Reward hanya diukur di Survei 2.** Jadi importance reward S1-vs-S2 tidak bisa
  dibandingkan. Yang bisa: posisi reward di dalam Survei 2, plus kenaikan pemilah yang
  menyertai kehadiran reward (ini kebersamaan waktu, bukan bukti sebab).
- **Reward cenderung seragam tinggi.** Kalau hampir semua warga menilai reward tinggi,
  Gini importance-nya bisa kecil bukan karena tak penting, tapi karena tak ada variasi
  untuk dibedakan. Bukti reward terkuat tetap dari lonjakan pemilah deskriptif.
- **Ini perilaku laporan-diri.** Fitur (sikap) dan label (perilaku) sama-sama diisi orang
  yang sama dalam satu sesi, sehingga hubungan yang ditemukan sebagian bisa berasal dari
  kecenderungan menjawab konsisten, bukan tentu hubungan nyata.

## Data lapangan

Data pemantauan lapangan (kg sampah) kini dianalisis di `08` dan `09`.
`08` berdiri sendiri: peringkat warga (paling banyak vs paling rajin — dua hal berbeda),
status keberlangsungan (konsisten / sudah tapi belum konsisten / tidak memilah), dan tren
program. `09` menautkannya ke label survei untuk menguji hal paling penting: apakah yang
survei-nya "sudah memilah" benar-benar memilah di lapangan.