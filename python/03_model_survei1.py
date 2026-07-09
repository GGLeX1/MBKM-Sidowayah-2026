from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import common as c
 
folder = Path(__file__).parent.parent
keluaran = folder / "output"
 
df = pd.read_csv(keluaran / "survei_1_labeled.csv")
fitur = c.fitur_survei1
X, y = df[fitur], df["label_num"]
 
sudah = int(y.sum())
print("Model Survei 1 (Random Forest, leave-one-out)")
print(f"Fitur : {len(fitur)} item (pengetahuan, sikap, hambatan)")
print(f"Kelas : sudah={sudah}, belum={len(y) - sudah}")
 
model = RandomForestClassifier(n_estimators=300, min_samples_leaf=1, class_weight="balanced", random_state=42, n_jobs=-1)
 
tebakan = cross_val_predict(model, X, y, cv=LeaveOneOut(), n_jobs=-1)
peluang = cross_val_predict(model, X, y, cv=LeaveOneOut(), method="predict_proba", n_jobs=-1)[:, 1]
 
akurasi = accuracy_score(y, tebakan)
f1 = f1_score(y, tebakan, zero_division=0)
km = confusion_matrix(y, tebakan)
 
print(f"\nHasil cross-validation (jujur, out-of-fold):")
print(f"  Akurasi  : {akurasi:.3f}")
print(f"  F1       : {f1:.3f}")
print(f"  Matriks  : [benar belum={km[0, 0]}, salah={km[0, 1]}] [salah={km[1, 0]}, benar sudah={km[1, 1]}]")
if sudah < 8:
    print(f"  (kelas 'sudah' cuma {sudah} — angka di atas tidak stabil, baca sebagai catatan saja)")
 
model.fit(X, y)
penting = pd.DataFrame({"fitur": fitur, "gini": model.feature_importances_}).sort_values("gini", ascending=False)
penting.to_csv(keluaran / "gini_survei1.csv", index=False)
 
def kelompok(nama):
    if nama in c.pengetahuan:
        return "pengetahuan"
    if nama in c.sikap:
        return "sikap"
    return "hambatan"
 
per_kelompok = penting.assign(kel=penting["fitur"].map(kelompok)).groupby("kel")["gini"].sum().sort_values(ascending=False)
print("\nGini importance per kategori:")
for kel, nilai in per_kelompok.items():
    print(f"  {kel:<12}: {nilai:.3f} ({nilai * 100:.0f}%)")
 
df_out = df[["id", "label", "label_num"]].copy()
df_out["prediksi_num"] = tebakan
df_out["peluang_sudah"] = peluang.round(3)
df_out.to_csv(keluaran / "survei_1_inference.csv", index=False)
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
warna_kel = {"pengetahuan": c.SEDANG, "sikap": c.GELAP, "hambatan": c.MERAH}
fig, ax = plt.subplots(figsize=(9, 6))
p = penting.iloc[::-1]
ax.barh(p["fitur"], p["gini"], color=[warna_kel[kelompok(f)] for f in p["fitur"]], edgecolor="white")
ax.set_xlabel("Gini importance")
ax.set_title("Faktor prediktor 'sudah memilah' — Survei 1", color=c.GELAP, fontweight="bold")
fig.tight_layout()
fig.savefig(keluaran / "03_gini_survei1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
print(f"\nGambar    -> output/03_gini_survei1.png")
print(f"Inference -> output/survei_1_inference.csv")