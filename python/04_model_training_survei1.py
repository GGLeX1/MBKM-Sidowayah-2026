import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import importlib.util
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

BASE_DIR   = Path(__file__).parent.parent
DATA_SYN   = BASE_DIR / "data" / "synthetic" / "sintetis_survei_1.csv"
DATA_REAL  = BASE_DIR / "data" / "processed"  / "survei_1_labeled.csv"
GEN_FILE   = BASE_DIR / "Python" / "03_synthetic_data_generation.py"
MODEL_DIR  = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "model_training"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("gen03", str(GEN_FILE))
gen03 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen03)
generate_synthetic = gen03.generate_synthetic

GREEN_DARK  = "#1a5e37"
GREEN_MED   = "#2d7a4e"
GREEN_LIGHT = "#a8d5b8"
ACCENT      = "#f0a500"
RED         = "#c0392b"

RANDOM_SEED     = 42
CV_MODE         = "kfold"
N_FOLDS         = 7
N_SYN_PER_FOLD  = 800
TARGET_F1       = 0.75

KOLOM_PENGETAHUAN = ["P1_tau_organik",  "P2_tau_cara",        "P3_tau_dampak"]
KOLOM_SIKAP       = ["S1_rasa_jwb",     "S2_percaya_penting", "S3_sekitar_milah"]
KOLOM_PERILAKU    = ["PK1_niat_rutin",  "PK2_sering_milah",   "PK3_ikut_kegiatan", "PK4_rutin_buang"]
KOLOM_HAMBATAN    = ["H1_merepotkan",   "H2_buang_waktu",     "H3_fasilitas",       "H4_lingkungan_dukung"]
KOLOM_FITUR       = KOLOM_PENGETAHUAN + KOLOM_SIKAP + KOLOM_PERILAKU + KOLOM_HAMBATAN

print("=" * 65)
print("  MODEL TRAINING SURVEI 1 (revisi) | Desa Sidowayah")
print("=" * 65)

USE_XGB = False
try:
    from xgboost import XGBClassifier
    USE_XGB = True
    boost_name = "XGBoost"
    print("\n  [METHOD] XGBoost tersedia")
except ImportError:
    boost_name = "Gradient Boosting"
    print("\n  [METHOD] XGBoost tidak ada -> fallback GradientBoosting")

def make_models():
    rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1)
    if USE_XGB:
        bo = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                           colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_SEED, n_jobs=-1)
    else:
        bo = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RANDOM_SEED)
    return {"Random Forest": rf, boost_name: bo}

df_syn  = pd.read_csv(DATA_SYN)
df_real = pd.read_csv(DATA_REAL)
x_syn, y_syn = df_syn[KOLOM_FITUR], df_syn["label_num"]
x_real, y_real = df_real[KOLOM_FITUR], df_real["label_num"]
print(f"\n[LOAD] Sintetis penuh : {len(df_syn)} | Data asli (CV) : {len(df_real)}")

print("\n" + "-" * 65)
print("  STEP 1: CV OPTIMISTIK (dalam data sintetis — bocor, sebagai pembanding)")
print("-" * 65)
skf_syn = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
opt_f1 = {}
for nama, model in make_models().items():
    s = cross_val_score(model, x_syn, y_syn, cv=skf_syn, scoring="f1", n_jobs=-1)
    opt_f1[nama] = s.mean()
    print(f"  {nama:<20}: F1 = {s.mean():.4f} (menyesatkan — train/test berbagi leluhur)")

print("\n" + "-" * 65)
print(f"  STEP 2: CV JUJUR ({CV_MODE}, regenerate sintetis per fold)")
print("-" * 65)
print("  Tiap fold: generate sintetis HANYA dari fold-latih, uji di fold-tahan asli.")
print("  Setiap responden ditebak model yang tak pernah memakainya.\n")

if CV_MODE == "loo":
    splitter = LeaveOneOut()
    splits = list(splitter.split(x_real))
else:
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(x_real, y_real))

oof_true = {n: [] for n in make_models()}
oof_pred = {n: [] for n in make_models()}
for fold, (tr_idx, te_idx) in enumerate(splits):
    src = df_real.iloc[tr_idx][KOLOM_FITUR + ["label_num"]]
    syn_fold = generate_synthetic(src, n_total=N_SYN_PER_FOLD, random_seed=RANDOM_SEED + fold)
    xs, ys = syn_fold[KOLOM_FITUR], syn_fold["label_num"]
    xt = x_real.iloc[te_idx]
    yt = y_real.iloc[te_idx]
    for nama, model in make_models().items():
        model.fit(xs, ys)
        oof_pred[nama].extend(model.predict(xt).tolist())
        oof_true[nama].extend(yt.tolist())

honest = {}
print(f"  {'Model':<20} {'F1':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8}")
print(f"  {'-'*54}")
for nama in oof_true:
    yt = np.array(oof_true[nama]); yp = np.array(oof_pred[nama])
    honest[nama] = {"f1": f1_score(yt, yp), "acc": accuracy_score(yt, yp),
                    "prec": precision_score(yt, yp), "rec": recall_score(yt, yp),
                    "cm": confusion_matrix(yt, yp)}
    h = honest[nama]
    print(f"  {nama:<20} {h['f1']:>8.4f} {h['acc']:>8.4f} {h['prec']:>8.4f} {h['rec']:>8.4f}")

print("\n" + "-" * 65)
print("  STEP 3: Pilih Model (berdasar F1 JUJUR) & Latih Final")
print("-" * 65)
best_name = max(honest, key=lambda k: honest[k]["f1"])
best_f1   = honest[best_name]["f1"]
status    = "TERCAPAI" if best_f1 >= TARGET_F1 else "BELUM TERCAPAI"
print(f"\n  Model terpilih   : {best_name}")
print(f"  F1 jujur (CV)    : {best_f1:.4f}  (target > {TARGET_F1}: {status})")
print(f"  F1 optimistik    : {opt_f1[best_name]:.4f}  (selisih = kebocoran terukur)")

best_model = make_models()[best_name]
best_model.fit(x_syn, y_syn)
print(f"\n  Model final dilatih di sintetis-penuh (n={len(x_syn)}, dari 42 asli).")

print("\n" + "-" * 65)
print("  STEP 4: Feature Importance Intrinsik (Gini)")
print("-" * 65)
imp_df = pd.DataFrame({"fitur": KOLOM_FITUR, "importance": best_model.feature_importances_})
imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
print(f"\n  {'Fitur':<25} {'Importance':>12}")
for _, r in imp_df.iterrows():
    bar = "█" * int(r["importance"]/imp_df["importance"].max()*25)
    print(f"  {r['fitur']:<25} {r['importance']:>12.4f}  {bar}")

plt.rcParams.update({"font.family":"sans-serif","axes.spines.top":False,"axes.spines.right":False})

fig, ax = plt.subplots(figsize=(10, 5))
xp = np.arange(len(honest)); names = list(honest)
ax.bar(xp-0.2, [opt_f1[n] for n in names], 0.35, label="F1 optimistik (bocor)", color=GREEN_LIGHT, edgecolor=GREEN_MED)
ax.bar(xp+0.2, [honest[n]["f1"] for n in names], 0.35, label="F1 jujur (CV per fold)", color=GREEN_DARK, edgecolor="white")
for i, n in enumerate(names):
    ax.text(i-0.2, opt_f1[n]+0.01, f"{opt_f1[n]:.3f}", ha="center", fontsize=8)
    ax.text(i+0.2, honest[n]["f1"]+0.01, f"{honest[n]['f1']:.3f}", ha="center", fontsize=8, fontweight="bold")
ax.axhline(TARGET_F1, color="gray", linestyle="--", label=f"Target {TARGET_F1}")
ax.set_xticks(xp); ax.set_xticklabels(names); ax.set_ylim(0, 1.1)
ax.set(ylabel="F1", title="F1 Optimistik vs Jujur — selisih = kebocoran")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(fontsize=9)
plt.tight_layout(); plt.savefig(REPORT_DIR/"01_f1_jujur_vs_optimistik.png", dpi=150, bbox_inches="tight"); plt.close()
print("\n  [SAVED] 01_f1_jujur_vs_optimistik.png")

fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(honest[best_name]["cm"], annot=True, fmt="d", cmap="Greens", cbar=False, ax=ax,
            xticklabels=["belum","sudah"], yticklabels=["belum","sudah"], annot_kws={"size":14,"fontweight":"bold"})
ax.set(xlabel="Prediksi", ylabel="Aktual", title=f"Confusion Matrix JUJUR (out-of-fold)\n{best_name} | F1={best_f1:.3f}")
ax.title.set(fontsize=11, fontweight="bold", color=GREEN_DARK)
plt.tight_layout(); plt.savefig(REPORT_DIR/"02_confusion_jujur.png", dpi=150, bbox_inches="tight"); plt.close()
print("  [SAVED] 02_confusion_jujur.png")

warna = {"P":GREEN_MED,"S":GREEN_DARK,"PK":ACCENT,"H":RED}
def pref(f): return "PK" if f.startswith("PK") else ("H" if f.startswith("H") else ("P" if f.startswith("P") else "S"))
cols = [warna[pref(f)] for f in imp_df["fitur"]]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(imp_df["fitur"][::-1], imp_df["importance"][::-1], color=cols[::-1], edgecolor="white")
for i, v in enumerate(imp_df["importance"][::-1]):
    ax.text(v+0.002, i, f"{v:.3f}", va="center", fontsize=8)
ax.set(xlabel="Importance", title=f"Feature Importance — {best_name} (revisi)")
ax.title.set(fontsize=13, fontweight="bold", color=GREEN_DARK)
ax.legend(handles=[mpatches.Patch(color=GREEN_MED,label="Pengetahuan"),mpatches.Patch(color=GREEN_DARK,label="Sikap"),
                   mpatches.Patch(color=ACCENT,label="Perilaku"),mpatches.Patch(color=RED,label="Hambatan")], fontsize=8, loc="lower right")
plt.tight_layout(); plt.savefig(REPORT_DIR/"03_feature_importance.png", dpi=150, bbox_inches="tight"); plt.close()
print("  [SAVED] 03_feature_importance.png")

model_path = MODEL_DIR / "model_survei1.pkl"
joblib.dump({"model": best_model, "model_name": best_name, "features": KOLOM_FITUR,
             "cv_f1_honest": best_f1, "cv_f1_optimistic": opt_f1[best_name],
             "cv_mode": CV_MODE, "random_seed": RANDOM_SEED}, model_path)
imp_df.to_csv(REPORT_DIR/"feature_importance.csv", index=False)
print(f"\n[SAVE] Model -> {model_path}")

print("\n" + "=" * 65)
print("  RINGKASAN")
print("=" * 65)
print(f"  Model terpilih    : {best_name}")
print(f"  F1 JUJUR (CV)     : {best_f1:.4f}  (inilah angka yang dilaporkan)")
print(f"  F1 optimistik     : {opt_f1[best_name]:.4f}  (bocor)")
print(f"  Fitur teratas     : {imp_df.iloc[0]['fitur']} ({imp_df.iloc[0]['importance']:.4f})")
hamb_imp = imp_df[imp_df['fitur'].str.startswith('H')]['importance'].sum()
print(f"  Total importance Hambatan: {hamb_imp:.4f}  (sebelum revisi ~0)")
print(f"\n  LANGKAH SELANJUTNYA: 05_inference_survei1.py")
print("\n[SELESAI] 04 (revisi)\n")