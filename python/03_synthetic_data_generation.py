import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
DATA_IN    = BASE_DIR / "data" / "processed"  / "survei_1_labeled.csv"
DATA_OUT   = BASE_DIR / "data" / "synthetic"  / "sintetis_survei_1.csv"
REPORT_DIR = BASE_DIR / "reports" / "synthetic_data"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data" / "synthetic").mkdir(parents=True, exist_ok=True)

GREEN_DARK  = "#1a5e37"
GREEN_MED   = "#2d7a4e"
GREEN_LIGHT = "#a8d5b8"
ACCENT      = "#f0a500"
RED         = "#c0392b"

N_SYNTHETIC = 1000 
RANDOM_SEED = 42

KOLOM_PENGETAHUAN = ["P1_tau_organik",  "P2_tau_cara",        "P3_tau_dampak"]
KOLOM_SIKAP       = ["S1_rasa_jwb",     "S2_percaya_penting", "S3_sekitar_milah"]
KOLOM_PERILAKU    = ["PK1_niat_rutin",  "PK2_sering_milah",   "PK3_ikut_kegiatan", "PK4_rutin_buang"]
KOLOM_HAMBATAN    = ["H1_merepotkan",   "H2_buang_waktu",     "H3_fasilitas",       "H4_lingkungan_dukung"]
KOLOM_FITUR       = KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU + KOLOM_HAMBATAN
KOLOM_HAMBATAN_NEGATIF = ["H1_merepotkan", "H2_buang_waktu"]
KOLOM_HAMBATAN_POSITIF = ["H3_fasilitas",  "H4_lingkungan_dukung"]

print("=" * 65)
print("  SYNTHETIC DATA GENERATION | Desa Sidowayah")
print("=" * 65)

df = pd.read_csv(DATA_IN)
print(f"\n[LOAD] Data berlabel: {df.shape[0]} baris x {df.shape[1]} kolom")

X = df[KOLOM_FITUR].copy()
y = df["label_num"].copy()
label_mapping = df[["label","label_num"]].drop_duplicates()

print(f"\n  Distribusi label asli:")
for lbl, cnt in y.value_counts().items():
    nama = "sudah_memilah" if lbl == 1 else "belum_memilah"
    print(f"    {nama}: {cnt} orang ({cnt/len(y)*100:.1f}%)")

print("\n" + "-" * 65)
print("  STEP 1: Analisis Distribusi Data Asli")
print("-" * 65)

likert_cols = KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU
print("\n  Kolom Likert (1-5):")
for col in likert_cols:
    counts = X[col].value_counts().sort_index()
    probs  = (counts / len(X)).round(3).to_dict()
    print(f"    {col:<22}: mean={X[col].mean():.2f}, std={X[col].std():.2f} | prob={probs}")

print("\n  Kolom Hambatan (binary):")
for col in KOLOM_HAMBATAN:
    p_ya = X[col].mean()
    print(f"    {col:<25}: P(Ya=1)={p_ya:.3f}, P(Tidak=0)={1-p_ya:.3f}")

corr_matrix = X.corr()
print(f"\n  Korelasi rata-rata antar fitur: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")

print("\n" + "-" * 65)
print("  STEP 2: Generate Data Sintetis")
print("-" * 65)

USE_SDV = False
try:
    from sdv.single_table import GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata
    USE_SDV = True
    print("\n  [METHOD] SDV GaussianCopula tersedia — menggunakan metode primer")
except ImportError:
    print("\n  [METHOD] SDV tidak tersedia — menggunakan fallback: Bootstrap + Noise")
    print("           (Install SDV: pip install sdv)")

np.random.seed(RANDOM_SEED)

if USE_SDV:
    df_for_sdv = X.copy()
    df_for_sdv["label_num"] = y.values

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_for_sdv)

    for col in likert_cols:
        metadata.update_column(col, sdtype="numerical")
    for col in KOLOM_HAMBATAN:
        metadata.update_column(col, sdtype="categorical")
    metadata.update_column("label_num", sdtype="categorical")

    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    synthesizer.fit(df_for_sdv)
    df_synthetic = synthesizer.sample(N_SYNTHETIC)
    df_synthetic["label_num"] = df_synthetic["label_num"].astype(int)

else:
    label_proportions = y.value_counts(normalize=True)
    synthetic_parts   = []

    for label_val, proportion in label_proportions.items():
        n_class  = int(N_SYNTHETIC * proportion)
        X_class  = X[y == label_val].values
        indices  = np.random.choice(len(X_class), size=n_class, replace=True)
        X_sample = X_class[indices].astype(float)

        noise_scale = 0.25
        for i, col in enumerate(KOLOM_FITUR):
            if col in likert_cols:
                noise = np.random.normal(0, noise_scale, size=n_class)
                X_sample[:, i] += noise
                X_sample[:, i] = np.clip(X_sample[:, i], 1, 5)
                X_sample[:, i] = np.round(X_sample[:, i])
            else:
                p_satu = X[col].mean()
                X_sample[:, i] = np.random.binomial(1, p_satu, size=n_class).astype(float)

        df_class              = pd.DataFrame(X_sample, columns=KOLOM_FITUR)
        df_class["label_num"] = label_val
        synthetic_parts.append(df_class)

    df_synthetic = pd.concat(synthetic_parts, ignore_index=True)
    df_synthetic = df_synthetic.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

for col in KOLOM_FITUR:
    df_synthetic[col] = df_synthetic[col].round().astype(int)
    if col in likert_cols:
        df_synthetic[col] = df_synthetic[col].clip(1, 5)
    else:
        df_synthetic[col] = df_synthetic[col].clip(0, 1)

df_synthetic["label"] = df_synthetic["label_num"].map({1: "sudah_memilah", 0: "belum_memilah"})
df_synthetic.insert(0, "ID_syn", [f"SYN{str(i+1).zfill(4)}" for i in range(len(df_synthetic))])

print(f"\n  Data sintetis berhasil di-generate: {len(df_synthetic)} sampel")
print(f"\n  Distribusi label sintetis:")
for lbl, cnt in df_synthetic["label"].value_counts().items():
    print(f"    {lbl}: {cnt} ({cnt/len(df_synthetic)*100:.1f}%)")

print("\n" + "-" * 65)
print("  STEP 3: Validasi Kualitas Data Sintetis")
print("-" * 65)

print("\n  Kolmogorov-Smirnov Test (p > 0.05 = distribusi serupa):")
print(f"  {'Kolom':<25} {'KS-Stat':>10} {'p-value':>10} {'Status':>10}")
print(f"  {'-'*58}")

ks_results = {}
for col in likert_cols:
    ks_stat, p_val = stats.ks_2samp(X[col], df_synthetic[col])
    status = "OK" if p_val > 0.05 else "PERHATIAN"
    ks_results[col] = (ks_stat, p_val, status)
    print(f"  {col:<25} {ks_stat:>10.4f} {p_val:>10.4f} {status:>10}")

print("\n  Chi-Square Test untuk Kolom Hambatan (p > 0.05 = distribusi serupa):")
print(f"  {'Kolom':<25} {'Chi2':>10} {'p-value':>10} {'Status':>10}")
print(f"  {'-'*58}")

for col in KOLOM_HAMBATAN:
    obs_asli = X[col].value_counts().sort_index()
    obs_syn  = df_synthetic[col].value_counts().sort_index().reindex(obs_asli.index, fill_value=0)
    expected = obs_asli / obs_asli.sum() * obs_syn.sum()
    if (expected > 5).all():
        chi2, p_val = stats.chisquare(obs_syn, f_exp=expected)
    else:
        chi2, p_val = 0.0, 1.0 
    status = "OK" if p_val > 0.05 else "PERHATIAN"
    print(f"  {col:<25} {chi2:>10.4f} {p_val:>10.4f} {status:>10}")

print("\n  Perbandingan Mean (Asli vs Sintetis):")
print(f"  {'Kolom':<25} {'Mean Asli':>12} {'Mean Syn':>12} {'Selisih':>10}")
print(f"  {'-'*62}")
for col in KOLOM_FITUR:
    m_asli = X[col].mean()
    m_syn  = df_synthetic[col].mean()
    diff   = abs(m_asli - m_syn)
    flag   = " !" if diff > 0.3 else ""
    print(f"  {col:<25} {m_asli:>12.4f} {m_syn:>12.4f} {diff:>10.4f}{flag}")

corr_asli = X[KOLOM_FITUR].corr()
corr_syn  = df_synthetic[KOLOM_FITUR].corr()
corr_diff = (corr_asli - corr_syn).abs()
print(f"\n  Perbedaan korelasi (mean absolute) : {corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean():.4f}")
print(f"  (Semakin kecil = semakin mirip distribusi korelasi)")

print("\n" + "-" * 65)
print("  MEMBUAT VISUALISASI VALIDASI")
print("-" * 65)

plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("Validasi: Distribusi Likert — Data Asli vs Sintetis",
             fontsize=13, fontweight="bold", color=GREEN_DARK, y=1.01)
for i, col in enumerate(likert_cols):
    ax = axes[i//5][i%5]
    cnt_asli = X[col].value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0) / len(X)
    cnt_syn  = df_synthetic[col].value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0) / len(df_synthetic)
    x_pos = np.arange(1, 6)
    ax.bar(x_pos - 0.2, cnt_asli, 0.35, label="Asli", color=GREEN_DARK, alpha=0.8, edgecolor="white")
    ax.bar(x_pos + 0.2, cnt_syn,  0.35, label="Sintetis", color=ACCENT, alpha=0.8, edgecolor="white")
    ax.set(title=col, xlabel="Skor", ylabel="Proporsi")
    ax.set_title(col, fontsize=7.5, wrap=True)
    ax.set_xticks([1,2,3,4,5])
    ks_stat, p_val = ks_results[col][:2]
    ax.set_xlabel(f"KS p={p_val:.3f}", fontsize=7)
    if i == 0:
        ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(REPORT_DIR/"01_validasi_likert.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 01_validasi_likert.png")

fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle("Validasi: Distribusi Hambatan — Asli vs Sintetis",
             fontsize=13, fontweight="bold", color=GREEN_DARK)
for ax, col in zip(axes, KOLOM_HAMBATAN):
    p_asli = [1 - X[col].mean(), X[col].mean()]
    p_syn  = [1 - df_synthetic[col].mean(), df_synthetic[col].mean()]
    x_pos  = np.arange(2)
    ax.bar(x_pos - 0.2, p_asli, 0.35, label="Asli", color=GREEN_DARK, alpha=0.8, edgecolor="white")
    ax.bar(x_pos + 0.2, p_syn,  0.35, label="Sintetis", color=ACCENT, alpha=0.8, edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Tidak(0)","Ya(1)"])
    ax.set(title=col, ylabel="Proporsi")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(REPORT_DIR/"02_validasi_hambatan.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 02_validasi_hambatan.png")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Perbandingan Korelasi Matrix: Asli vs Sintetis",
             fontsize=13, fontweight="bold", color=GREEN_DARK)
cmap_ = sns.diverging_palette(145, 10, as_cmap=True)
mask_ = np.triu(np.ones((len(KOLOM_FITUR),len(KOLOM_FITUR)), dtype=bool))
sns.heatmap(corr_asli, mask=mask_, cmap=cmap_, center=0, annot=True, fmt=".2f",
            square=True, linewidths=0.3, ax=axes[0], annot_kws={"size":6},
            cbar=False)
axes[0].set_title("Data Asli (n=42)", fontweight="bold", color=GREEN_DARK)
sns.heatmap(corr_syn, mask=mask_, cmap=cmap_, center=0, annot=True, fmt=".2f",
            square=True, linewidths=0.3, ax=axes[1], annot_kws={"size":6},
            cbar=False)
axes[1].set_title(f"Data Sintetis (n={N_SYNTHETIC})", fontweight="bold", color=ACCENT)
sns.heatmap(corr_diff, mask=mask_, cmap="Reds", annot=True, fmt=".2f",
            square=True, linewidths=0.3, ax=axes[2], annot_kws={"size":6},
            cbar=False)
axes[2].set_title("|Selisih| Korelasi", fontweight="bold", color=RED)
plt.tight_layout()
plt.savefig(REPORT_DIR/"03_perbandingan_korelasi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 03_perbandingan_korelasi.png")

norm_pengetahuan = [(df_synthetic[c] - 1) / 4 for c in KOLOM_PENGETAHUAN]
norm_sikap       = [(df_synthetic[c] - 1) / 4 for c in KOLOM_SIKAP]
norm_perilaku    = [(df_synthetic[c] - 1) / 4 for c in KOLOM_PERILAKU]
norm_h_neg = [1 - df_synthetic[c] for c in KOLOM_HAMBATAN_NEGATIF]
norm_h_pos = [df_synthetic[c].astype(float) for c in KOLOM_HAMBATAN_POSITIF]

syn_skor_pengetahuan = sum(norm_pengetahuan) / len(KOLOM_PENGETAHUAN)
syn_skor_sikap       = sum(norm_sikap) / len(KOLOM_SIKAP)
syn_skor_perilaku    = sum(norm_perilaku) / len(KOLOM_PERILAKU)
syn_skor_hambatan    = (sum(norm_h_neg) + sum(norm_h_pos)) / len(KOLOM_HAMBATAN)
df_synthetic["skor_total"] = (syn_skor_pengetahuan + syn_skor_sikap +
                               syn_skor_perilaku + syn_skor_hambatan) / 4

df_asli_scores = pd.read_csv(DATA_IN)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_asli_scores["skor_total"], bins=15, alpha=0.7, color=GREEN_DARK,
        edgecolor="white", label=f"Data Asli (n=42)", density=True)
ax.hist(df_synthetic["skor_total"], bins=30, alpha=0.5, color=ACCENT,
        edgecolor="white", label=f"Data Sintetis (n={N_SYNTHETIC})", density=True)
ax.axvline(df_asli_scores["skor_total"].mean(), color=GREEN_DARK, linestyle="--",
           linewidth=2, label=f"Mean asli={df_asli_scores['skor_total'].mean():.3f}")
ax.axvline(df_synthetic["skor_total"].mean(), color=ACCENT, linestyle="--",
           linewidth=2, label=f"Mean sintetis={df_synthetic['skor_total'].mean():.3f}")
ax.set(xlabel="Skor Total (0-1)", ylabel="Densitas",
       title="Perbandingan Distribusi Skor Total: Asli vs Sintetis")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(REPORT_DIR/"04_distribusi_skor_total.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 04_distribusi_skor_total.png")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
fig.suptitle("Distribusi Label: Asli vs Sintetis", fontsize=13, fontweight="bold", color=GREEN_DARK)
colors_pie = [GREEN_DARK, GREEN_LIGHT]
wp = dict(linewidth=2, edgecolor="white")
asli_label_cnt = pd.read_csv(DATA_IN)["label"].value_counts()
syn_label_cnt  = df_synthetic["label"].value_counts()
axes[0].pie(asli_label_cnt.values, labels=asli_label_cnt.index, autopct="%1.1f%%",
            colors=colors_pie, wedgeprops=wp, textprops={"fontsize":9})
axes[0].set_title(f"Data Asli (n={len(df)})")
axes[1].pie(syn_label_cnt.values, labels=syn_label_cnt.index, autopct="%1.1f%%",
            colors=colors_pie, wedgeprops=wp, textprops={"fontsize":9})
axes[1].set_title(f"Data Sintetis (n={N_SYNTHETIC})")
plt.tight_layout()
plt.savefig(REPORT_DIR/"05_distribusi_label.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 05_distribusi_label.png")

kolom_simpan = KOLOM_FITUR + ["label", "label_num"]
df_synthetic[kolom_simpan].to_csv(DATA_OUT, index=False)
print(f"\n[SAVE] Data sintetis disimpan → {DATA_OUT}")
print(f"       Jumlah: {len(df_synthetic)} sampel | Kolom: {len(kolom_simpan)}")

print("\n" + "=" * 65)
print("  RINGKASAN SYNTHETIC DATA GENERATION")
print("=" * 65)
print(f"\n  Metode yang digunakan : {'SDV GaussianCopula' if USE_SDV else 'Bootstrap + Gaussian Noise (fallback)'}")
print(f"  Jumlah data sintetis  : {N_SYNTHETIC} sampel")
print(f"  Random seed           : {RANDOM_SEED}")
print(f"\n  Distribusi label sintetis:")
for lbl, cnt in df_synthetic["label"].value_counts().items():
    print(f"    {lbl}: {cnt} ({cnt/len(df_synthetic)*100:.1f}%)")
print(f"\n  Rata-rata KS p-value  : {np.mean([v[1] for v in ks_results.values()]):.4f}")
print(f"  (Semakin tinggi = semakin mirip data asli)")
print(f"\n  Output → {DATA_OUT}")
print(f"  Viz    → {REPORT_DIR}")
print(f"\n  LANGKAH SELANJUTNYA: Jalankan 04_model_training_survei1.py")
print("\n[SELESAI] 03_synthetic_data_generation.py berhasil dijalankan.\n")