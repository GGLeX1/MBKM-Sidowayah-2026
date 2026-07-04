import re
import pandas as pd
 
GELAP = "#1a5e37"
SEDANG = "#2d7a4e"
MUDA = "#a8d5b8"
AKSEN = "#f0a500"
MERAH = "#c0392b"
ABU = "#95a5a6"
 
item = {
    "P1_organik": "Saya mengetahui perbedaan sampah organik dan anorganik.",
    "P2_cara": "Saya mengetahui cara memilah sampah dengan benar.",
    "P3_dampak": "Saya mengetahui dampak sampah terhadap lingkungan.",
    "S1_tanggungjawab": "Saya merasa bertanggung jawab terhadap lingkungan.",
    "S2_penting": "Saya percaya bahwa memilah sampah itu penting.",
    "S3_sekitar": "Orang di sekitar saya memilah sampah dengan baik.",
    "PK1_niat": "Saya berniat memilah sampah secara rutin.",
    "PK2_sering": "Saya sering melakukan pemilahan sampah.",
    "PK3_kegiatan": "Saya mengikuti kegiatan yang berkaitan dengan lingkungan.",
    "PK4_rutin": "Saya rutin membuang sampah sesuai kategori dalam kehidupan sehari-hari.",
    "H1_repot": "Memilah sampah merupakan kegiatan yang merepotkan.",
    "H2_waktu": "Memilah sampah merupakan kegiatan yang membuang waktu.",
    "H3_fasilitas": "Fasilitas pemilihan sampah sudah tersedia di sekitar lingkungan Anda.",
    "H4_dukungan": "Lingkungan (warga sekitar dan fasilitas) mendukung untuk dilakukan kegiatan pemilahan sampah.",
}
 
reward = {
    "RW1_tertarik": "Apakah Anda tertarik jika ada Reward/hadiah untuk memilah sampah.",
    "RW2_pengaruh": "Seberapa besar Reward/hadiah memengaruhi keputusan Anda.",
    "RW3_menarik": "Apakah Reward/hadiah saat ini menarik.",
    "RW4_sebanding": "Apakah usaha memilah sampah sebanding dengan Reward/hadiah yang diberikan.",
}
 
pengetahuan = ["P1_organik", "P2_cara", "P3_dampak"]
sikap = ["S1_tanggungjawab", "S2_penting", "S3_sekitar"]
hambatan = ["H1_repot", "H2_waktu", "H3_fasilitas", "H4_dukungan"]
kolom_reward = list(reward.keys())
 
fitur_survei1 = pengetahuan + sikap + hambatan
fitur_survei2 = pengetahuan + sikap + hambatan + kolom_reward
 
ITEM_AKSI = ["PK2_sering", "PK4_rutin"]
AMBANG_AKSI = 4
AMBANG_PAHAM = 4
 
 
def _rapikan(teks):
    return teks.strip().replace("\n", " ")
 
 
def muat_survei(path, pakai_reward=False):
    """Baca CSV survei, ganti pertanyaan panjang jadi nama pendek, dan anonimkan responden."""
    df = pd.read_csv(path)
    df.columns = [_rapikan(c) for c in df.columns]
 
    peta = {}
    kamus = {**item, **reward} if pakai_reward else item
    for pendek, panjang in kamus.items():
        target = _rapikan(panjang)
        cocok = next((c for c in df.columns if c == target), None)
        if cocok is not None:
            peta[cocok] = pendek
    df = df.rename(columns=peta)
 
    for h in hambatan:
        df[h] = df[h].astype(str).str.strip().map({"Ya": 1, "Tidak": 0}).astype(int)
 
    df.insert(0, "id", [f"R{str(i + 1).zfill(2)}" for i in range(len(df))])
    df = df.drop(columns=["Nama Lengkap"], errors="ignore")
    return df
 
 
def beri_label(df):
    """Tandai 'sudah memilah' hanya bila benar-benar melakukan aksi pemilahan (PK2 & PK4 tinggi)."""
    aksi_tinggi = pd.concat([df[k] >= AMBANG_AKSI for k in ITEM_AKSI], axis=1).all(axis=1)
    df = df.copy()
    df["label"] = aksi_tinggi.map({True: "sudah_memilah", False: "belum_memilah"})
    df["label_num"] = aksi_tinggi.astype(int)
    return df
 
 
def tingkat_perilaku(df):
    """Tiga tingkat deskriptif: belum paham, paham tapi belum memilah, dan sudah memilah."""
    paham = df[pengetahuan].mean(axis=1) >= AMBANG_PAHAM
    sudah = df["label_num"] == 1
 
    def pilih(row_sudah, row_paham):
        if row_sudah:
            return "sudah_memilah"
        if row_paham:
            return "paham_belum_memilah"
        return "belum_paham"
 
    return [pilih(s, p) for s, p in zip(sudah, paham)]