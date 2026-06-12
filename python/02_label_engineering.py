import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
DATA_IN    = BASE_DIR / "data" / "processed" / "survei_1_clean.csv"
DATA_OUT   = BASE_DIR / "data" / "processed" / "survei_1_labeled.csv"
REPORT_DIR = BASE_DIR / "reports" / "label_engineering"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

GREEN_DARK  = "#1a5e37"
GREEN_MED   = "#2d7a4e"
GREEN_LIGHT = "#a8d5b8"
ACCENT      = "#f0a500"
RED         = "#c0392b"

KOLOM_PENGETAHUAN = ["P1_tau_organik", "P2_tau_cara", "P3_tau_dampak"]
KOLOM_SIKAP       = ["S1_rasa_jwb", "S2_percaya_penting", "S3_sekitar_milah"]
KOLOM_PERILAKU    = ["PK1_niat_rutin", "PK2_sering_milah", "PK3_ikut_kegiatan", "PK4_rutin_buang"]

KOLOM_HAMBATAN_NEGATIF  = ["H1_merepotkan", "H2_buang_waktu"] 
KOLOM_HAMBATAN_POSITIF  = ["H3_fasilitas", "H4_lingkungan_dukung"]  
KOLOM_HAMBATAN_ALL      = KOLOM_HAMBATAN_NEGATIF + KOLOM_HAMBATAN_POSITIF

print("=" * 65)
print("  LABEL ENGINEERING — Composite Scoring | Desa Sidowayah")
print("=" * 65)

df = pd.read_csv(DATA_IN)
print(f"\n[LOAD] Data bersih: {df.shape[0]} baris x {df.shape[1]} kolom")
print("\n" + "-" * 65)
print("  STEP 1: Normalisasi ke Skala 0-1")
print("-" * 65)

all_likert = KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU
for col in all_likert:
    df[f"{col}_norm"] = (df[col] - 1) / (5 - 1) 

for col in KOLOM_HAMBATAN_NEGATIF:
    df[f"{col}_norm"] = 1 - df[col] 

for col in KOLOM_HAMBATAN_POSITIF:
    df[f"{col}_norm"] = df[col].astype(float)

print("  Likert (1-5)    → (nilai-1)/4  [0=terburuk, 1=terbaik]")
print("  Hambatan negatif → 1 - nilai   [0=ada hambatan, 1=tidak ada hambatan]")
print("  Hambatan positif → nilai as-is [1=mendukung/ada fasilitas]")
print("\n" + "-" * 65)
print("  STEP 2: Skor per Kategori (rata-rata dalam kategori)")
print("-" * 65)

norm_pengetahuan = [f"{c}_norm" for c in KOLOM_PENGETAHUAN]
norm_sikap       = [f"{c}_norm" for c in KOLOM_SIKAP]
norm_perilaku    = [f"{c}_norm" for c in KOLOM_PERILAKU]
norm_hambatan    = [f"{c}_norm" for c in KOLOM_HAMBATAN_ALL]

df["skor_pengetahuan"] = df[norm_pengetahuan].mean(axis=1)
df["skor_sikap"]       = df[norm_sikap].mean(axis=1)
df["skor_perilaku"]    = df[norm_perilaku].mean(axis=1)
df["skor_hambatan"]    = df[norm_hambatan].mean(axis=1)

for kat, col in [("Pengetahuan","skor_pengetahuan"),("Sikap","skor_sikap"),
                  ("Perilaku","skor_perilaku"),("Hambatan","skor_hambatan")]:
    print(f"  {kat:<15}: mean={df[col].mean():.3f}  std={df[col].std():.3f}  "
          f"min={df[col].min():.3f}  max={df[col].max():.3f}")

print("\n" + "-" * 65)
print("  STEP 3: Skor Total (Equal Weighting 4 kategori)")
print("-" * 65)

BOBOT = {
    "skor_pengetahuan" : 0.25,
    "skor_sikap"       : 0.25,
    "skor_perilaku"    : 0.25,
    "skor_hambatan"    : 0.25,
}
df["skor_total"] = sum(df[k] * v for k, v in BOBOT.items())

print("  Bobot tiap kategori: 0.25 (equal weighting)")
print(f"\n  Distribusi Skor Total:")
print(f"    Mean   : {df['skor_total'].mean():.4f}")
print(f"    Median : {df['skor_total'].median():.4f}")
print(f"    Std    : {df['skor_total'].std():.4f}")
print(f"    Min    : {df['skor_total'].min():.4f}")
print(f"    Max    : {df['skor_total'].max():.4f}")
print("\n" + "-" * 65)
print("  STEP 4: Penetapan Threshold & Pemberian Label")
print("-" * 65)

THRESHOLD = df["skor_total"].median()
print(f"\n  Threshold (median)    : {THRESHOLD:.4f}")
print(f"  Interpretasi          : skor >= {THRESHOLD:.4f} → 'sudah memilah'")
print(f"                        : skor <  {THRESHOLD:.4f} → 'belum memilah'")

df["label"] = df["skor_total"].apply(lambda x: "sudah_memilah" if x >= THRESHOLD else "belum_memilah")
df["label_num"] = df["label"].map({"sudah_memilah": 1, "belum_memilah": 0})

label_counts = df["label"].value_counts()
print(f"\n  Distribusi Label:")
for lbl, cnt in label_counts.items():
    bar = "█" * cnt
    print(f"    {lbl:<20}: {cnt:>3} orang ({cnt/len(df)*100:.1f}%)  {bar}")

ratio = label_counts.min() / label_counts.max()
if ratio < 0.7:
    print(f"\n  [PERINGATAN] Data mungkin imbalanced (ratio={ratio:.2f})")
    print("  → Pertimbangkan SMOTE saat generate data sintetis (file 03)")
else:
    print(f"\n  [OK] Data relatif seimbang (ratio={ratio:.2f})")

print("\n" + "-" * 65)
print("  STEP 5: Analisis Sensitivitas Threshold")
print("-" * 65)
print("  (Bagaimana distribusi berubah jika threshold berbeda?)\n")

thresholds_test = [0.40, 0.45, 0.50, 0.55, 0.60, THRESHOLD]
thresholds_test = sorted(set([round(t, 4) for t in thresholds_test]))

print(f"  {'Threshold':<12} {'Sudah Memilah':>15} {'Belum Memilah':>15} {'Ratio':>8}")
print(f"  {'-'*52}")
for t in thresholds_test:
    n_sudah = (df["skor_total"] >= t).sum()
    n_belum = (df["skor_total"] < t).sum()
    ratio_t = min(n_sudah, n_belum) / max(n_sudah, n_belum) if max(n_sudah, n_belum) > 0 else 0
    marker  = " ← DIPILIH" if abs(t - THRESHOLD) < 0.0001 else ""
    print(f"  {t:<12.4f} {n_sudah:>15} {n_belum:>15} {ratio_t:>7.2f}{marker}")

print("\n" + "-" * 65)
print("  MEMBUAT VISUALISASI")
print("-" * 65)

plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribusi Skor Total dan Penetapan Label", fontsize=14, fontweight="bold", color=GREEN_DARK)

n, bins, patches = axes[0].hist(df["skor_total"], bins=12, color=GREEN_MED, edgecolor="white", alpha=0.8)
for patch in patches:
    if patch.get_x() >= THRESHOLD:
        patch.set_facecolor(GREEN_DARK)
axes[0].axvline(THRESHOLD, color=ACCENT, linewidth=2.5, linestyle="--", label=f"Threshold={THRESHOLD:.3f} (median)")
axes[0].axvline(df["skor_total"].mean(), color=RED, linewidth=1.5, linestyle=":", label=f"Mean={df['skor_total'].mean():.3f}")
axes[0].set(xlabel="Skor Total (0-1)", ylabel="Jumlah Responden", title="Distribusi Skor Total")
patch_sudah = mpatches.Patch(color=GREEN_DARK, label="Sudah Memilah")
patch_belum = mpatches.Patch(color=GREEN_MED,  label="Belum Memilah")
axes[0].legend(handles=[patch_sudah, patch_belum], fontsize=8, loc="upper left")
axes[0].legend(fontsize=8)

colors_pie = [GREEN_DARK, GREEN_LIGHT]
wedge_props = dict(linewidth=2, edgecolor="white")
axes[1].pie(label_counts.values, labels=label_counts.index,
            autopct="%1.1f%%", colors=colors_pie, wedgeprops=wedge_props,
            textprops={"fontsize": 10})
axes[1].set_title("Distribusi Label Akhir", fontweight="bold")

plt.tight_layout()
plt.savefig(REPORT_DIR/"01_distribusi_skor_label.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 01_distribusi_skor_label.png")

fig, ax = plt.subplots(figsize=(15, 5))
x = np.arange(len(df))
w = 0.2
cats_plot = [("Pengetahuan","skor_pengetahuan"),("Sikap","skor_sikap"),
             ("Perilaku","skor_perilaku"),("Hambatan","skor_hambatan")]
colors_cats = [GREEN_MED, GREEN_DARK, ACCENT, RED]
for i, ((kat, col), color) in enumerate(zip(cats_plot, colors_cats)):
    ax.bar(x + i*w, df[col], width=w, label=kat, color=color, alpha=0.8, edgecolor="white")
ax.axhline(THRESHOLD, color="gray", linestyle="--", linewidth=1.2, label=f"Threshold={THRESHOLD:.3f}")
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(df["ID"], rotation=60, fontsize=6.5)
ax.set(xlabel="Responden", ylabel="Skor (0-1)", title="Skor per Kategori per Responden")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(fontsize=9, ncol=5, loc="upper right")
plt.tight_layout()
plt.savefig(REPORT_DIR/"02_skor_per_responden.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 02_skor_per_responden.png")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Perbandingan Skor antar Label", fontsize=14, fontweight="bold", color=GREEN_DARK)

groups = [df.loc[df["label"]=="sudah_memilah","skor_total"],
          df.loc[df["label"]=="belum_memilah","skor_total"]]
bp = axes[0].boxplot(groups, patch_artist=True,
                     medianprops=dict(color=ACCENT, linewidth=2))
axes[0].set_xticklabels(["Sudah Memilah","Belum Memilah"])
bp["boxes"][0].set_facecolor(GREEN_DARK)
bp["boxes"][1].set_facecolor(GREEN_LIGHT)
axes[0].axhline(THRESHOLD, color="gray", linestyle="--", label=f"Threshold={THRESHOLD:.3f}")
axes[0].set(ylabel="Skor Total (0-1)", title="Distribusi Skor Total per Label")
axes[0].legend(fontsize=8)

kat_cols = ["skor_pengetahuan","skor_sikap","skor_perilaku","skor_hambatan"]
kat_names = ["Pengetahuan","Sikap","Perilaku","Hambatan"]
mean_sudah = df.loc[df["label"]=="sudah_memilah", kat_cols].mean()
mean_belum = df.loc[df["label"]=="belum_memilah", kat_cols].mean()
x_r = np.arange(len(kat_names))
axes[1].bar(x_r - 0.2, mean_sudah, 0.35, label="Sudah Memilah", color=GREEN_DARK, edgecolor="white")
axes[1].bar(x_r + 0.2, mean_belum, 0.35, label="Belum Memilah", color=GREEN_LIGHT, edgecolor=GREEN_MED, linewidth=1)
for i, (v1, v2) in enumerate(zip(mean_sudah, mean_belum)):
    axes[1].text(i-0.2, v1+0.01, f"{v1:.2f}", ha="center", fontsize=8, fontweight="bold")
    axes[1].text(i+0.2, v2+0.01, f"{v2:.2f}", ha="center", fontsize=8)
axes[1].set_xticks(x_r)
axes[1].set_xticklabels(kat_names)
axes[1].set(ylabel="Skor Rata-rata (0-1)", title="Skor Rata-rata per Kategori & Label")
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(REPORT_DIR/"03_perbandingan_label.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 03_perbandingan_label.png")

thresholds_range = np.arange(0.30, 0.75, 0.01)
n_sudah_range = [(df["skor_total"] >= t).sum() for t in thresholds_range]
n_belum_range = [(df["skor_total"] < t).sum() for t in thresholds_range]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(thresholds_range, n_sudah_range, color=GREEN_DARK, linewidth=2, label="Sudah Memilah")
ax.plot(thresholds_range, n_belum_range, color=RED,        linewidth=2, label="Belum Memilah")
ax.axvline(THRESHOLD, color=ACCENT, linestyle="--", linewidth=2, label=f"Threshold dipilih={THRESHOLD:.3f}")
ax.fill_between(thresholds_range, n_sudah_range, n_belum_range, alpha=0.08, color=GREEN_MED)
ax.set(xlabel="Threshold", ylabel="Jumlah Responden",
       title="Sensitivitas Distribusi Label terhadap Perubahan Threshold")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(REPORT_DIR/"04_sensitivitas_threshold.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 04_sensitivitas_threshold.png")

kolom_output = (
    ["ID", "Umur", "Pendidikan Terakhir"] +
    KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU + KOLOM_HAMBATAN_ALL +
    ["skor_pengetahuan","skor_sikap","skor_perilaku","skor_hambatan","skor_total","label","label_num"]
)
df_out = df[[c for c in kolom_output if c in df.columns]]
df_out.to_csv(DATA_OUT, index=False)
print(f"\n[SAVE] Data berlabel disimpan → {DATA_OUT}")

print("\n" + "=" * 65)
print("  RINGKASAN LABEL ENGINEERING")
print("=" * 65)
print(f"\n  Metode          : Composite Scoring — Equal Weighting")
print(f"  Bobot per kat.  : 0.25 (Pengetahuan + Sikap + Perilaku + Hambatan)")
print(f"  Threshold       : {THRESHOLD:.4f} (median skor total)")
print(f"\n  Label 'sudah_memilah'  : {label_counts.get('sudah_memilah',0):>3} orang ({label_counts.get('sudah_memilah',0)/len(df)*100:.1f}%)")
print(f"  Label 'belum_memilah'  : {label_counts.get('belum_memilah',0):>3} orang ({label_counts.get('belum_memilah',0)/len(df)*100:.1f}%)")
print(f"\n  Output → {DATA_OUT}")
print(f"  Viz    → {REPORT_DIR}")
print("\n[SELESAI] 02_label_engineering.py berhasil dijalankan.\n")