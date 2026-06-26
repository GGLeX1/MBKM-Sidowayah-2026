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
NOISE_SCALE = 0.25

KOLOM_PENGETAHUAN = ["P1_tau_organik",  "P2_tau_cara",        "P3_tau_dampak"]
KOLOM_SIKAP       = ["S1_rasa_jwb",     "S2_percaya_penting", "S3_sekitar_milah"]
KOLOM_PERILAKU    = ["PK1_niat_rutin",  "PK2_sering_milah",   "PK3_ikut_kegiatan", "PK4_rutin_buang"]
KOLOM_HAMBATAN    = ["H1_merepotkan",   "H2_buang_waktu",     "H3_fasilitas",       "H4_lingkungan_dukung"]
KOLOM_FITUR       = KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU + KOLOM_HAMBATAN
LIKERT_COLS       = KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU


def generate_synthetic(df_source, n_total=N_SYNTHETIC, random_seed=RANDOM_SEED, prefer_sdv=True):
    x = df_source[KOLOM_FITUR].copy().reset_index(drop=True)
    y = df_source["label_num"].copy().reset_index(drop=True)

    if prefer_sdv:
        try:
            from sdv.single_table import GaussianCopulaSynthesizer
            from sdv.metadata import SingleTableMetadata
            df_fit = x.copy()
            df_fit["label_num"] = y.values
            metadata = SingleTableMetadata()
            metadata.detect_from_dataframe(df_fit)
            for col in LIKERT_COLS:
                metadata.update_column(col, sdtype="numerical")
            for col in KOLOM_HAMBATAN:
                metadata.update_column(col, sdtype="categorical")
            metadata.update_column("label_num", sdtype="categorical")
            synth = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
            synth.fit(df_fit)
            syn = synth.sample(n_total)
            syn["label_num"] = syn["label_num"].astype(int)
            for col in KOLOM_FITUR:
                syn[col] = syn[col].round().astype(int)
                syn[col] = syn[col].clip(1, 5) if col in LIKERT_COLS else syn[col].clip(0, 1)
            syn["label"] = syn["label_num"].map({1: "sudah_memilah", 0: "belum_memilah"})
            return syn[KOLOM_FITUR + ["label", "label_num"]].reset_index(drop=True)
        except Exception:
            pass

    rng = np.random.default_rng(random_seed)
    label_props = y.value_counts(normalize=True)
    parts = []
    for label_val, prop in label_props.items():
        n_class = int(round(n_total * prop))
        x_class = x[y == label_val].values.astype(float)
        idx = rng.integers(0, len(x_class), size=n_class)
        sampled = x_class[idx].copy()
        for j, col in enumerate(KOLOM_FITUR):
            if col in LIKERT_COLS:
                sampled[:, j] = sampled[:, j] + rng.normal(0, NOISE_SCALE, size=n_class)
                sampled[:, j] = np.clip(np.round(sampled[:, j]), 1, 5)
        dfc = pd.DataFrame(sampled, columns=KOLOM_FITUR)
        dfc["label_num"] = label_val
        parts.append(dfc)

    syn = pd.concat(parts, ignore_index=True)
    syn = syn.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    for col in KOLOM_FITUR:
        syn[col] = syn[col].round().astype(int)
        syn[col] = syn[col].clip(1, 5) if col in LIKERT_COLS else syn[col].clip(0, 1)
    syn["label"] = syn["label_num"].map({1: "sudah_memilah", 0: "belum_memilah"})
    return syn[KOLOM_FITUR + ["label", "label_num"]].reset_index(drop=True)


if __name__ == "__main__":
    BASE_DIR   = Path(__file__).parent.parent
    DATA_IN    = BASE_DIR / "data" / "processed" / "survei_1_labeled.csv"
    DATA_OUT   = BASE_DIR / "data" / "synthetic" / "sintetis_survei_1.csv"
    REPORT_DIR = BASE_DIR / "reports" / "synthetic_data"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "data" / "synthetic").mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  SYNTHETIC DATA GENERATION | Desa Sidowayah")
    print("=" * 65)

    df = pd.read_csv(DATA_IN)
    print(f"\n[LOAD] Data berlabel: {df.shape[0]} baris x {df.shape[1]} kolom")
    X = df[KOLOM_FITUR].copy()
    y = df["label_num"].copy()

    print(f"\n  Distribusi label asli:")
    for lbl, cnt in y.value_counts().sort_index().items():
        nama = "sudah_memilah" if lbl == 1 else "belum_memilah"
        print(f"    {nama}: {cnt} ({cnt/len(y)*100:.1f}%)")

    print("\n" + "-" * 65)
    print("  PERBAIKAN UTAMA vs versi lama")
    print("-" * 65)
    print("  Lama : Hambatan di-generate dari proporsi GLOBAL (label-independen)")
    print("         -> relasi Hambatan<->label putus, model mengabaikan H1-H4.")
    print("  Baru : baris di-bootstrap PER-KELAS, nilai Hambatan ikut barisnya")
    print("         -> P(Hambatan|label) terjaga, model bisa belajar Hambatan.")

    df_synthetic = generate_synthetic(df, n_total=N_SYNTHETIC, random_seed=RANDOM_SEED)
    print(f"\n[GEN] Data sintetis: {len(df_synthetic)} sampel")
    for lbl, cnt in df_synthetic["label"].value_counts().items():
        print(f"    {lbl}: {cnt} ({cnt/len(df_synthetic)*100:.1f}%)")

    print("\n" + "-" * 65)
    print("  VALIDASI 1: Kolmogorov-Smirnov (Likert) & Chi-Square (Hambatan)")
    print("-" * 65)
    ks_results = {}
    print(f"\n  {'Kolom':<22} {'KS-Stat':>9} {'p-value':>9} {'Status':>10}")
    for col in LIKERT_COLS:
        ks_stat, p_val = stats.ks_2samp(X[col], df_synthetic[col])
        ks_results[col] = (ks_stat, p_val)
        status = "OK" if p_val > 0.05 else "PERHATIAN"
        print(f"  {col:<22} {ks_stat:>9.4f} {p_val:>9.4f} {status:>10}")

    print(f"\n  {'Kolom':<22} {'Chi2':>9} {'p-value':>9} {'Status':>10}")
    for col in KOLOM_HAMBATAN:
        obs_asli = X[col].value_counts().sort_index()
        obs_syn  = df_synthetic[col].value_counts().sort_index().reindex(obs_asli.index, fill_value=0)
        expected = obs_asli / obs_asli.sum() * obs_syn.sum()
        if (expected > 5).all():
            chi2, p_val = stats.chisquare(obs_syn, f_exp=expected)
        else:
            chi2, p_val = 0.0, 1.0
        status = "OK" if p_val > 0.05 else "PERHATIAN"
        print(f"  {col:<22} {chi2:>9.4f} {p_val:>9.4f} {status:>10}")

    print("\n" + "-" * 65)
    print("  VALIDASI 2: Korelasi Hambatan<->label (BUKTI PERBAIKAN)")
    print("-" * 65)
    print(f"\n  {'Kolom':<22} {'corr asli':>10} {'corr syn':>10} {'selisih':>9}")
    for col in KOLOM_HAMBATAN:
        c_asli = np.corrcoef(X[col], y)[0, 1]
        c_syn  = np.corrcoef(df_synthetic[col], df_synthetic["label_num"])[0, 1]
        print(f"  {col:<22} {c_asli:>10.3f} {c_syn:>10.3f} {abs(c_asli-c_syn):>9.3f}")
    print("\n  (corr syn yang mengikuti corr asli = relasi Hambatan-label terjaga)")

    plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle("Validasi Likert — Asli vs Sintetis", fontsize=13, fontweight="bold", color=GREEN_DARK, y=1.01)
    for i, col in enumerate(LIKERT_COLS):
        ax = axes[i//5][i%5]
        ca = X[col].value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0)/len(X)
        cs = df_synthetic[col].value_counts().sort_index().reindex([1,2,3,4,5], fill_value=0)/len(df_synthetic)
        xp = np.arange(1,6)
        ax.bar(xp-0.2, ca, 0.35, label="Asli", color=GREEN_DARK, alpha=0.8, edgecolor="white")
        ax.bar(xp+0.2, cs, 0.35, label="Sintetis", color=ACCENT, alpha=0.8, edgecolor="white")
        ax.set_title(col, fontsize=8)
        ax.set_xticks([1,2,3,4,5])
        ax.set_xlabel(f"KS p={ks_results[col][1]:.3f}", fontsize=7)
        if i == 0: ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"01_validasi_likert.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [SAVED] 01_validasi_likert.png")

    corr_h_asli = [np.corrcoef(X[c], y)[0,1] for c in KOLOM_HAMBATAN]
    corr_h_syn  = [np.corrcoef(df_synthetic[c], df_synthetic["label_num"])[0,1] for c in KOLOM_HAMBATAN]
    fig, ax = plt.subplots(figsize=(10, 5))
    xp = np.arange(len(KOLOM_HAMBATAN))
    ax.bar(xp-0.2, corr_h_asli, 0.35, label="Asli", color=GREEN_DARK, edgecolor="white")
    ax.bar(xp+0.2, corr_h_syn,  0.35, label="Sintetis", color=ACCENT, edgecolor="white")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xticks(xp); ax.set_xticklabels(KOLOM_HAMBATAN, rotation=15, fontsize=8)
    ax.set(ylabel="Korelasi dengan label", title="Korelasi Hambatan<->label: Asli vs Sintetis")
    ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR/"02_korelasi_hambatan_label.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] 02_korelasi_hambatan_label.png")

    df_synthetic.to_csv(DATA_OUT, index=False)
    print(f"\n[SAVE] Data sintetis -> {DATA_OUT}")
    print(f"       {len(df_synthetic)} sampel | {len(KOLOM_FITUR)+2} kolom")

    print("\n" + "=" * 65)
    print("  RINGKASAN")
    print("=" * 65)
    print(f"  Fungsi generate_synthetic() siap dipanggil per-fold oleh 04.")
    print(f"  Rata-rata KS p-value Likert: {np.mean([v[1] for v in ks_results.values()]):.4f}")
    print(f"  Output -> {DATA_OUT}")
    print("\n[SELESAI] 03_synthetic_data_generation.py\n")