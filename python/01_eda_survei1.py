import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent
DATA_RAW   = BASE_DIR / "data" / "raw"        / "survei_1_raw.csv"
DATA_OUT   = BASE_DIR / "data" / "processed"  / "survei_1_clean.csv"
REPORT_DIR = BASE_DIR / "reports" / "eda_survei1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data" / "processed").mkdir(parents=True, exist_ok=True)

GREEN_DARK  = "#1a5e37"
GREEN_MED   = "#2d7a4e"
ACCENT      = "#f0a500"

KOLOM_INFO = {
    "Saya mengetahui perbedaan sampah organik dan anorganik." : ("P1_tau_organik",       "Pengetahuan"),
    "Saya mengetahui cara memilah sampah dengan benar."       : ("P2_tau_cara",           "Pengetahuan"),
    "Saya mengetahui dampak sampah terhadap lingkungan."      : ("P3_tau_dampak",         "Pengetahuan"),
    "Saya merasa bertanggung jawab terhadap lingkungan."      : ("S1_rasa_jwb",           "Sikap"),
    "Saya percaya bahwa memilah sampah itu penting."          : ("S2_percaya_penting",    "Sikap"),
    "Orang di sekitar saya memilah sampah dengan baik."       : ("S3_sekitar_milah",      "Sikap"),
    "Saya berniat memilah sampah secara rutin."               : ("PK1_niat_rutin",        "Perilaku & Kebiasaan"),
    "Saya sering melakukan pemilahan sampah."                 : ("PK2_sering_milah",      "Perilaku & Kebiasaan"),
    "Saya mengikuti kegiatan yang berkaitan dengan lingkungan.": ("PK3_ikut_kegiatan",    "Perilaku & Kebiasaan"),
    "Saya rutin membuang sampah sesuai kategori dalam kehidupan sehari-hari.": ("PK4_rutin_buang", "Perilaku & Kebiasaan"),
    "Memilah sampah merupakan kegiatan yang merepotkan."      : ("H1_merepotkan",         "Hambatan"),
    "Memilah sampah merupakan kegiatan yang membuang\nwaktu." : ("H2_buang_waktu",        "Hambatan"),
    "Fasilitas pemilihan sampah sudah tersedia di sekitar lingkungan Anda.": ("H3_fasilitas", "Hambatan"),
    "Lingkungan (warga sekitar dan fasilitas) mendukung untuk dilakukan kegiatan pemilahan sampah.": ("H4_lingkungan_dukung", "Hambatan"),
}

KATEGORI_WARNA = {
    "Pengetahuan"          : "#2d7a4e",
    "Sikap"                : "#1a5e37",
    "Perilaku & Kebiasaan" : "#f0a500",
    "Hambatan"             : "#c0392b",
}

print("=" * 65)
print("  EDA SURVEI 1 — Program Pemilahan Sampah | Desa Sidowayah")
print("=" * 65)

df_raw = pd.read_csv(DATA_RAW)
print(f"\n[LOAD] Data mentah: {df_raw.shape[0]} baris x {df_raw.shape[1]} kolom")

df = df_raw.copy()
df.columns = df.columns.str.strip()

df.insert(0, "ID", [f"R{str(i+1).zfill(2)}" for i in range(len(df))])
df.drop(columns=["Nama Lengkap"], inplace=True)

pend_map = {
    "sma":"SMA","Sma":"SMA","smp":"SMP","smk":"SMK",
    "s1 ":"S1","s1":"S1","-":"Tidak Diketahui","tidak sekolah":"Tidak Sekolah"
}
df["Pendidikan Terakhir"] = df["Pendidikan Terakhir"].str.strip().replace(pend_map)

rename_map = {}
for kol_asli, (alias, _) in KOLOM_INFO.items():
    kol_norm = kol_asli.strip().replace("\n", " ")
    for col in df.columns:
        if col.strip().replace("\n", " ") == kol_norm:
            rename_map[col] = alias
            break
df.rename(columns=rename_map, inplace=True)

alias_hambatan = [v[0] for v in KOLOM_INFO.values() if v[1] == "Hambatan"]
for col in alias_hambatan:
    if col in df.columns:
        df[col] = df[col].map({"Ya": 1, "Tidak": 0})

alias_likert = [v[0] for v in KOLOM_INFO.values() if v[1] != "Hambatan"]

print(f"\n[CLEANING] Missing values  : {df.isnull().sum().sum()}")
print(f"[CLEANING] Duplikat        : {df.duplicated().sum()}")
print(f"[CLEANING] Kolom akhir     : {df.shape[1]}")

df.to_csv(DATA_OUT, index=False)
print(f"[SAVE]     Data bersih     → {DATA_OUT}")

print("\n" + "-" * 65)
print("  STATISTIK DESKRIPTIF")
print("-" * 65)

print(f"\n  Jumlah responden : {len(df)}")
print(f"  Rerata usia      : {df['Umur'].mean():.1f} th  |  Median: {df['Umur'].median():.0f} th  |  Std: {df['Umur'].std():.1f}")
print(f"  Range usia       : {df['Umur'].min()}–{df['Umur'].max()} tahun")

print("\n  Distribusi Pendidikan:")
for pend, cnt in df["Pendidikan Terakhir"].value_counts().items():
    print(f"    {pend:<20}: {cnt:>3} orang ({cnt/len(df)*100:.1f}%)")

print("\n  Statistik Likert per Pertanyaan:")
stats = df[alias_likert].agg(["mean","median","std","min","max"]).T
stats.columns = ["Mean","Median","Std","Min","Max"]
print(stats.round(3).to_string())

print("\n  Skor Rata-rata per Kategori (skala 1-5):")
kat_means = {}
for kat in ["Pengetahuan", "Sikap", "Perilaku & Kebiasaan"]:
    cols = [v[0] for v in KOLOM_INFO.values() if v[1] == kat]
    kat_means[kat] = df[cols].mean().mean()
    bar = "█" * int(kat_means[kat] * 5)
    print(f"    {kat:<25}: {kat_means[kat]:.3f}  {bar}")

print("\n  Frekuensi Hambatan (% menjawab 'Ya'):")
for col in alias_hambatan:
    pct = df[col].mean() * 100
    label_h = next(k for k, v in KOLOM_INFO.items() if v[0] == col)
    print(f"    {col}: {pct:.0f}%  — {label_h[:55]}")

print("\n" + "-" * 65)
print("  MEMBUAT VISUALISASI")
print("-" * 65)

plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Demografi Responden", fontsize=14, fontweight="bold", color=GREEN_DARK)
axes[0].hist(df["Umur"], bins=8, color=GREEN_MED, edgecolor="white")
axes[0].axvline(df["Umur"].mean(), color=ACCENT, linestyle="--", linewidth=1.5, label=f"Mean={df['Umur'].mean():.1f}")
axes[0].axvline(df["Umur"].median(), color="red", linestyle=":", linewidth=1.5, label=f"Median={df['Umur'].median():.0f}")
axes[0].set(xlabel="Usia (tahun)", ylabel="Jumlah", title="Distribusi Usia")
axes[0].legend(fontsize=8)
pend_cnt = df["Pendidikan Terakhir"].value_counts()
bars = axes[1].barh(pend_cnt.index, pend_cnt.values, color=GREEN_MED, edgecolor="white")
for bar, val in zip(bars, pend_cnt.values):
    axes[1].text(val+0.1, bar.get_y()+bar.get_height()/2, f"{val} ({val/len(df)*100:.0f}%)", va="center", fontsize=8)
axes[1].set(xlabel="Jumlah Responden", title="Distribusi Pendidikan Terakhir")
plt.tight_layout()
plt.savefig(REPORT_DIR/"01_demografi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 01_demografi.png")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Distribusi Jawaban Likert (Skala 1-5)", fontsize=14, fontweight="bold", color=GREEN_DARK, y=1.01)
for i, col in enumerate(alias_likert):
    ax = axes[i//5][i%5]
    kat = next(v[1] for v in KOLOM_INFO.values() if v[0] == col)
    cnt = df[col].value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0)
    bars_ = ax.bar(cnt.index, cnt.values, color=KATEGORI_WARNA[kat], edgecolor="white")
    for b in bars_:
        if b.get_height() > 0:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2, str(int(b.get_height())), ha="center", fontsize=8)
    ax.set(title=col, xlabel="Skor", ylabel="n", xticks=[1,2,3,4,5])
    ax.set_ylim(0, cnt.max()+4)
    ax.set_title(col, fontsize=7.5, wrap=True)
    patch = mpatches.Patch(color=KATEGORI_WARNA[kat], label=kat)
    ax.legend(handles=[patch], fontsize=6, loc="upper left")
plt.tight_layout()
plt.savefig(REPORT_DIR/"02_distribusi_likert.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 02_distribusi_likert.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Boxplot Skor per Kategori", fontsize=14, fontweight="bold", color=GREEN_DARK)
for ax, kat in zip(axes, ["Pengetahuan","Sikap","Perilaku & Kebiasaan"]):
    cols = [v[0] for v in KOLOM_INFO.values() if v[1] == kat]
    bp = ax.boxplot([df[c].values for c in cols], patch_artist=True,
                    medianprops=dict(color=ACCENT, linewidth=2),
                    boxprops=dict(facecolor=KATEGORI_WARNA[kat], alpha=0.6))
    ax.set_xticklabels(cols, rotation=20, ha="right", fontsize=8)
    ax.set(title=kat, ylabel="Skor (1-5)")
    ax.set_ylim(0, 6)
    ax.axhline(df[cols].values.mean(), color="gray", linestyle="--", linewidth=1,
               label=f"Mean={df[cols].values.mean():.2f}")
    ax.legend(fontsize=8)
    ax.set_title(kat, fontweight="bold", color=KATEGORI_WARNA[kat])
plt.tight_layout()
plt.savefig(REPORT_DIR/"03_boxplot_kategori.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 03_boxplot_kategori.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Ringkasan Skor per Kategori", fontsize=14, fontweight="bold", color=GREEN_DARK)
cats  = list(kat_means.keys())
means = list(kat_means.values())
bars_ = axes[0].barh(cats, means, color=[KATEGORI_WARNA[c] for c in cats], edgecolor="white", height=0.5)
for bar, val in zip(bars_, means):
    axes[0].text(val+0.03, bar.get_y()+bar.get_height()/2, f"{val:.2f}", va="center", fontweight="bold")
axes[0].set_xlim(0, 5.5)
axes[0].axvline(3.0, color="gray", linestyle="--", linewidth=1, label="Netral (3.0)")
axes[0].axvline(df[alias_likert].mean().mean(), color=ACCENT, linestyle=":", linewidth=1.5,
                label=f"Grand mean={df[alias_likert].mean().mean():.2f}")
axes[0].set(xlabel="Skor Rata-rata (1-5)", title="Skor Rata-rata per Kategori")
axes[0].legend(fontsize=8)
h_vals = [df[c].mean()*100 for c in alias_hambatan]
bars2  = axes[1].bar(alias_hambatan, h_vals, color=KATEGORI_WARNA["Hambatan"], edgecolor="white", alpha=0.8)
for bar, val in zip(bars2, h_vals):
    axes[1].text(bar.get_x()+bar.get_width()/2, val+1, f"{val:.0f}%", ha="center", fontweight="bold")
axes[1].set_ylim(0, 110)
axes[1].axhline(50, color="gray", linestyle="--", linewidth=1, label="50%")
axes[1].set(xlabel="Item Hambatan", ylabel="% menjawab 'Ya'", title="Proporsi Hambatan")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(REPORT_DIR/"04_ringkasan_kategori.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 04_ringkasan_kategori.png")

all_alias = alias_likert + alias_hambatan
corr = df[all_alias].corr()
fig, ax = plt.subplots(figsize=(13, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap=sns.diverging_palette(145,10,as_cmap=True),
            center=0, annot=True, fmt=".2f", square=True, linewidths=0.5,
            cbar_kws={"shrink":0.7}, ax=ax, annot_kws={"size":7})
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    col_name = lbl.get_text()
    for v in KOLOM_INFO.values():
        if v[0] == col_name:
            lbl.set_color(KATEGORI_WARNA[v[1]])
ax.set_title("Heatmap Korelasi Antar Semua Variabel Survei 1",
             fontsize=13, fontweight="bold", color=GREEN_DARK, pad=15)
ax.legend(handles=[mpatches.Patch(color=KATEGORI_WARNA[k], label=k) for k in KATEGORI_WARNA],
          loc="upper right", fontsize=9, title="Kategori", bbox_to_anchor=(1.18,1))
plt.tight_layout()
plt.savefig(REPORT_DIR/"05_heatmap_korelasi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 05_heatmap_korelasi.png")

fig, ax = plt.subplots(figsize=(14, 7))
palette_skor = ["#c0392b","#e67e22","#f1c40f","#2ecc71","#1a5e37"]
label_skor   = ["Sangat Tidak Setuju (1)","Tidak Setuju (2)","Netral (3)","Setuju (4)","Sangat Setuju (5)"]
bottom = np.zeros(len(alias_likert))
for skor_val, color, lbl in zip([1,2,3,4,5], palette_skor, label_skor):
    vals = [df[col].eq(skor_val).sum()/len(df)*100 for col in alias_likert]
    ax.bar(range(len(alias_likert)), vals, bottom=bottom, color=color, label=lbl,
           edgecolor="white", linewidth=0.5)
    bottom += np.array(vals)
ax.set_xticks(range(len(alias_likert)))
ax.set_xticklabels(alias_likert, rotation=35, ha="right", fontsize=8.5)
ax.set(ylabel="Persentase Responden (%)",
       title="Distribusi Jawaban per Pertanyaan (Stacked 100%)")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(loc="upper right", fontsize=8, bbox_to_anchor=(1.18,1))
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
for tick, col in zip(ax.get_xticklabels(), alias_likert):
    for v in KOLOM_INFO.values():
        if v[0] == col:
            tick.set_color(KATEGORI_WARNA[v[1]])
plt.tight_layout()
plt.savefig(REPORT_DIR/"06_stacked_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 06_stacked_bar.png")

df["skor_pengetahuan"] = df[[v[0] for v in KOLOM_INFO.values() if v[1]=="Pengetahuan"]].mean(axis=1)
df["skor_sikap"]       = df[[v[0] for v in KOLOM_INFO.values() if v[1]=="Sikap"]].mean(axis=1)
df["skor_perilaku"]    = df[[v[0] for v in KOLOM_INFO.values() if v[1]=="Perilaku & Kebiasaan"]].mean(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Hubungan Usia dengan Skor per Kategori", fontsize=13, fontweight="bold", color=GREEN_DARK)
for ax, (kat, col_s) in zip(axes, [("Pengetahuan","skor_pengetahuan"),
                                     ("Sikap","skor_sikap"),
                                     ("Perilaku & Kebiasaan","skor_perilaku")]):
    ax.scatter(df["Umur"], df[col_s], color=KATEGORI_WARNA[kat], alpha=0.7, s=60, edgecolors="white")
    z = np.polyfit(df["Umur"], df[col_s], 1)
    x_line = np.linspace(df["Umur"].min(), df["Umur"].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), color="gray", linestyle="--", linewidth=1.5)
    r = df["Umur"].corr(df[col_s])
    ax.set_title(f"{kat}\n(r = {r:.2f})", fontweight="bold", color=KATEGORI_WARNA[kat])
    ax.set(xlabel="Usia (tahun)", ylabel="Skor Rata-rata")
    ax.set_ylim(0.5, 5.5)
plt.tight_layout()
plt.savefig(REPORT_DIR/"07_usia_vs_skor.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 07_usia_vs_skor.png")
print("\n" + "=" * 65)
print("  RINGKASAN INSIGHT EDA SURVEI 1")
print("=" * 65)
print(f"\n  Responden              : {len(df)} orang")
print(f"  Usia                   : {df['Umur'].min()}-{df['Umur'].max()} th  (mean {df['Umur'].mean():.1f})")
print(f"  Grand mean Likert      : {df[alias_likert].mean().mean():.3f} / 5.00")
print(f"\n  Kategori Tertinggi     : {max(kat_means, key=kat_means.get)} ({max(kat_means.values()):.3f})")
print(f"  Kategori Terendah      : {min(kat_means, key=kat_means.get)} ({min(kat_means.values()):.3f})")
print(f"\n  Missing values         : {df.isnull().sum().sum()}")
print(f"\n  Output → {REPORT_DIR}")
print("\n[SELESAI] 01_eda_survei1.py berhasil dijalankan.\n")
