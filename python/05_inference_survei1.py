import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

BASE_DIR    = Path(__file__).parent.parent
MODEL_IN    = BASE_DIR / "models" / "model_survei1.pkl"
DATA_IN     = BASE_DIR / "data" / "processed" / "survei_1_labeled.csv"
DATA_OUT    = BASE_DIR / "data" / "output" / "survei_1_inference.csv"
REPORT_DIR  = BASE_DIR / "reports" / "inference_survei1"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data" / "output").mkdir(parents=True, exist_ok=True)

GREEN_DARK  = "#1a5e37"
GREEN_MED   = "#2d7a4e"
GREEN_LIGHT = "#a8d5b8"
ACCENT      = "#f0a500"
RED         = "#c0392b"

BORDERLINE_LOW  = 0.40
BORDERLINE_HIGH = 0.60

print("=" * 65)
print("  INFERENCE SURVEI 1 + SHAP | Desa Sidowayah")
print("  (penerapan model ke 42 data asli — label & penjelasan per orang)")
print("=" * 65)

artifact   = joblib.load(MODEL_IN)
model      = artifact["model"]
features   = artifact["features"]
model_name = artifact["model_name"]
cv_f1      = artifact.get("cv_f1_honest", None)
print(f"\n[LOAD] Model: {model_name} | {len(features)} fitur")
if cv_f1 is not None:
    print(f"       F1 jujur (CV out-of-fold dari 04) : {cv_f1:.4f}  <- angka performa resmi")

df = pd.read_csv(DATA_IN)
print(f"[LOAD] Data asli: {df.shape[0]} responden")

x_real = df[features].copy()

print("\n" + "-" * 65)
print("  STEP 1: Inference (prediksi label + probabilitas)")
print("-" * 65)

pred       = model.predict(x_real)
proba      = model.predict_proba(x_real)[:, 1]
df["label_model"]  = np.where(pred == 1, "sudah_memilah", "belum_memilah")
df["prob_sudah"]   = proba
df["sesuai_scoring"] = (pred == df["label_num"]).map({True: "ya", False: "tidak"})
df["borderline"]   = np.where((proba >= BORDERLINE_LOW) & (proba <= BORDERLINE_HIGH), "ya", "tidak")

n_sudah = int((pred == 1).sum())
n_belum = int((pred == 0).sum())
print(f"\n  Hasil klasifikasi model:")
print(f"    sudah_memilah : {n_sudah} orang ({n_sudah/len(df)*100:.1f}%)")
print(f"    belum_memilah : {n_belum} orang ({n_belum/len(df)*100:.1f}%)")
print(f"\n  Rentang probabilitas : {proba.min():.3f} – {proba.max():.3f}")
n_border = int((df["borderline"] == "ya").sum())
print(f"  Borderline ({BORDERLINE_LOW}-{BORDERLINE_HIGH}) : {n_border} orang")

print("\n" + "-" * 65)
print("  STEP 2: Catatan Konsistensi vs Label Scoring (BUKAN uji performa)")
print("-" * 65)
print("\n  Angka performa resmi model adalah F1 JUJUR dari CV out-of-fold di 04")
if cv_f1 is not None:
    print(f"  (= {cv_f1:.4f}). Kecocokan di bawah ini hanya catatan konsistensi:")
print("  label scoring & sintetis pelatihan sama-sama berasal dari 42 responden,")
print("  jadi kecocokan tinggi BUKAN generalisasi, hanya konsistensi dgn scoring.")

n_match = int((pred == df["label_num"]).sum())
print(f"\n  Kecocokan dengan label scoring : {n_match}/{len(df)} responden ({n_match/len(df)*100:.1f}%)")

print("\n" + "-" * 65)
print("  STEP 3: Tabel Hasil per Responden")
print("-" * 65)
print(f"\n  {'ID':<5} {'Umur':>4} {'Skor':>6} {'Scoring':>14} {'Model':>14} {'Prob':>6} {'Cocok':>6}")
print(f"  {'-'*62}")
for _, r in df.iterrows():
    print(f"  {r['ID']:<5} {int(r['Umur']):>4} {r['skor_total']:>6.3f} "
          f"{r['label']:>14} {r['label_model']:>14} {r['prob_sudah']:>6.3f} {r['sesuai_scoring']:>6}")

print("\n" + "-" * 65)
print("  STEP 4: SHAP — Penjelasan Global (XAI)")
print("-" * 65)

USE_SHAP = False
try:
    import shap
    USE_SHAP = True
    print(f"\n  [METHOD] SHAP tersedia (v{shap.__version__}) — TreeExplainer")
except ImportError:
    print("\n  [METHOD] SHAP tidak tersedia — lewati bagian XAI")
    print("           (Install SHAP: pip install shap)")

if USE_SHAP:
    explainer = shap.TreeExplainer(model)
    exp_raw   = explainer(x_real)
    shap_vals = exp_raw.values[:, :, 1] if exp_raw.values.ndim == 3 else exp_raw.values
    base_val  = exp_raw.base_values[:, 1] if np.ndim(exp_raw.base_values) == 2 else exp_raw.base_values
    exp = shap.Explanation(values=shap_vals, base_values=base_val,
                           data=x_real.values, feature_names=features)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    shap_imp = pd.DataFrame({"fitur": features, "mean_abs_shap": mean_abs})
    shap_imp = shap_imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    print(f"\n  Mean |SHAP| per fitur (kontribusi global ke kelas 'sudah_memilah'):")
    print(f"  {'Fitur':<25} {'mean|SHAP|':>12}")
    print(f"  {'-'*38}")
    for _, row in shap_imp.iterrows():
        bar = "█" * int(row["mean_abs_shap"] / shap_imp["mean_abs_shap"].max() * 25) if shap_imp["mean_abs_shap"].max() > 0 else ""
        print(f"  {row['fitur']:<25} {row['mean_abs_shap']:>12.4f}  {bar}")
    shap_imp.to_csv(REPORT_DIR / "shap_importance.csv", index=False)

    plt.figure()
    shap.plots.beeswarm(exp, max_display=14, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Beeswarm — Kontribusi Fitur (kelas: sudah_memilah)",
                 fontsize=12, fontweight="bold", color=GREEN_DARK)
    fig.set_size_inches(10, 7)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "02_shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  [SAVED] 02_shap_beeswarm.png")

    plt.figure()
    shap.plots.bar(exp, max_display=14, show=False)
    fig = plt.gcf()
    fig.suptitle("SHAP Global — Mean |SHAP| per Fitur",
                 fontsize=12, fontweight="bold", color=GREEN_DARK)
    fig.set_size_inches(9, 7)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "03_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  [SAVED] 03_shap_bar.png")

    print("\n" + "-" * 65)
    print("  STEP 5: SHAP — Penjelasan Lokal (per responden contoh)")
    print("-" * 65)

    idx_sudah  = int(np.argmax(proba))
    idx_belum  = int(np.argmin(proba))
    idx_border = int(np.argmin(np.abs(proba - 0.5)))
    contoh = [("paling yakin sudah", idx_sudah),
              ("paling yakin belum", idx_belum),
              ("paling borderline",  idx_border)]
    seen = set()
    for nama_kasus, idx in contoh:
        if idx in seen:
            continue
        seen.add(idx)
        rid = df.iloc[idx]["ID"]
        plt.figure()
        shap.plots.waterfall(exp[idx], max_display=14, show=False)
        fig = plt.gcf()
        fig.suptitle(f"SHAP Lokal — {rid} ({nama_kasus}, prob={proba[idx]:.3f})",
                     fontsize=11, fontweight="bold", color=GREEN_DARK)
        fig.set_size_inches(9, 6)
        plt.tight_layout()
        fname = f"04_shap_waterfall_{rid}.png"
        plt.savefig(REPORT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [SAVED] {fname}  ({nama_kasus}: {rid})")

print("\n" + "-" * 65)
print("  MEMBUAT VISUALISASI INFERENCE")
print("-" * 65)

plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(proba, bins=20, color=GREEN_MED, edgecolor="white", alpha=0.85)
ax.axvspan(BORDERLINE_LOW, BORDERLINE_HIGH, color=ACCENT, alpha=0.2, label="zona borderline")
ax.axvline(0.5, color=RED, linestyle="--", linewidth=1.5, label="batas 0.5")
ax.set(xlabel="Probabilitas 'sudah_memilah'", ylabel="Jumlah Responden",
       title="Distribusi Probabilitas Prediksi — 42 Data Asli")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(REPORT_DIR / "01_distribusi_probabilitas.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 01_distribusi_probabilitas.png")

df_sorted = df.sort_values("prob_sudah").reset_index(drop=True)
fig, ax = plt.subplots(figsize=(14, 6))
colors_bar = [GREEN_DARK if p >= 0.5 else GREEN_LIGHT for p in df_sorted["prob_sudah"]]
ax.bar(range(len(df_sorted)), df_sorted["prob_sudah"], color=colors_bar, edgecolor="white")
ax.axhline(0.5, color=RED, linestyle="--", linewidth=1.3, label="batas 0.5")
ax.axhspan(BORDERLINE_LOW, BORDERLINE_HIGH, color=ACCENT, alpha=0.15, label="zona borderline")
ax.set_xticks(range(len(df_sorted)))
ax.set_xticklabels(df_sorted["ID"], rotation=60, fontsize=6.5)
ax.set(xlabel="Responden (urut probabilitas)", ylabel="Probabilitas 'sudah_memilah'",
       title="Probabilitas Prediksi per Responden")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
patch_sudah = mpatches.Patch(color=GREEN_DARK, label="prediksi: sudah")
patch_belum = mpatches.Patch(color=GREEN_LIGHT, label="prediksi: belum")
ax.legend(handles=[patch_sudah, patch_belum], fontsize=9, loc="upper left")
plt.tight_layout()
plt.savefig(REPORT_DIR / "05_prob_per_responden.png", dpi=150, bbox_inches="tight")
plt.close()
print("  [SAVED] 05_prob_per_responden.png")

kolom_out = (["ID", "Umur", "Pendidikan Terakhir", "skor_total",
              "label", "label_num", "label_model", "prob_sudah",
              "sesuai_scoring", "borderline"])
df_out = df[[c for c in kolom_out if c in df.columns]].rename(columns={"label": "label_scoring"})
df_out.to_csv(DATA_OUT, index=False)
print(f"\n[SAVE] Hasil inference → {DATA_OUT}")

print("\n" + "=" * 65)
print("  RINGKASAN INFERENCE SURVEI 1")
print("=" * 65)
print(f"\n  Model               : {model_name}")
if cv_f1 is not None:
    print(f"  F1 jujur (CV, dari 04): {cv_f1:.4f}  <- angka performa resmi")
print(f"  Responden           : {len(df)}")
print(f"  Prediksi sudah/belum : {n_sudah} / {n_belum}")
print(f"  Borderline          : {n_border}")
print(f"  Cocok dgn scoring    : {n_match}/{len(df)} (catatan: konsistensi, bukan generalisasi)")
if USE_SHAP:
    print(f"  Fitur SHAP teratas   : {shap_imp.iloc[0]['fitur']} ({shap_imp.iloc[0]['mean_abs_shap']:.4f})")
print(f"\n  Output → {DATA_OUT}")
print(f"  Viz    → {REPORT_DIR}")
print("\n[SELESAI] 05_inference_survei1.py berhasil dijalankan.\n")