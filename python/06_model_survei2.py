from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import common as c
 
folder = Path(__file__).parent.parent
keluaran = folder / "output"
 
df = pd.read_csv(keluaran / "survei_2_labeled.csv")
fitur = c.fitur_survei2
X, y = df[fitur], df["label_num"]
 
sudah = int(y.sum())
print("Model Survei 2 (Random Forest, leave-one-out)")
print(f"Fitur : {len(fitur)} item (pengetahuan, sikap, hambatan, reward)")
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
 
dasar = max(y.mean(), 1 - y.mean())
print(f"  Baseline tebak-mayoritas: {dasar:.3f}")
model_bermakna = akurasi > dasar + 0.05
if not model_bermakna:
    print("  Model TIDAK mengungguli tebak-mayoritas: fitur survei praktis tak memprediksi")
    print("  siapa yang memilah. Baca Gini di bawah dengan sangat hati-hati.")
 
model.fit(X, y)
penting = pd.DataFrame({"fitur": fitur, "gini": model.feature_importances_}).sort_values("gini", ascending=False)
penting.to_csv(keluaran / "gini_survei2.csv", index=False)
 
def kelompok(nama):
    if nama in c.pengetahuan:
        return "pengetahuan"
    if nama in c.sikap:
        return "sikap"
    if nama in c.hambatan:
        return "hambatan"
    return "reward"
 
per_kelompok = penting.assign(kel=penting["fitur"].map(kelompok)).groupby("kel")["gini"].sum().sort_values(ascending=False)
print("\nGini importance per kategori:")
for kel, nilai in per_kelompok.items():
    tanda = "  <- reward" if kel == "reward" else ""
    print(f"  {kel:<12}: {nilai:.3f} ({nilai * 100:.0f}%){tanda}")
 
reward_total = per_kelompok.get("reward", 0)
print(f"\nPeran reward menurut Gini: {reward_total * 100:.0f}% dari total.")
print("Hati-hati menafsir angka ini. Gini condong ke fitur yang nilainya lebih beragam.")
print("Reward (skala 1-5) punya lebih banyak nilai berbeda daripada hambatan (0/1), jadi")
print("bobotnya bisa terangkat oleh sebab teknis itu, bukan tentu karena mendorong perilaku.")
print("Bukti reward yang lebih bisa dipegang tetap dari kenaikan pemilah deskriptif (skrip 07).")
 
df_out = df[["id", "label", "label_num"]].copy()
df_out["prediksi_num"] = tebakan
df_out["peluang_sudah"] = peluang.round(3)
df_out.to_csv(keluaran / "survei_2_inference.csv", index=False)
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
warna_kel = {"pengetahuan": c.SEDANG, "sikap": c.GELAP, "hambatan": c.MERAH, "reward": c.AKSEN}
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
 
p = penting.iloc[::-1]
ax[0].barh(p["fitur"], p["gini"], color=[warna_kel[kelompok(f)] for f in p["fitur"]], edgecolor="white")
ax[0].set_xlabel("Gini importance")
ax[0].set_title("Faktor prediktor 'sudah memilah' — Survei 2", color=c.GELAP, fontweight="bold")
 
kel_urut = per_kelompok.iloc[::-1]
ax[1].barh(kel_urut.index, kel_urut.values, color=[warna_kel[k] for k in kel_urut.index], edgecolor="white")
for i, v in enumerate(kel_urut.values):
    ax[1].text(v + 0.005, i, f"{v * 100:.0f}%", va="center", fontsize=9)
ax[1].set_xlabel("Total Gini per kategori")
ax[1].set_title("Reward vs faktor lain", color=c.GELAP, fontweight="bold")
 
fig.tight_layout()
fig.savefig(keluaran / "06_gini_survei2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
print(f"\nGambar    -> output/06_gini_survei2.png")
print(f"Inference -> output/survei_2_inference.csv")