import re
from pathlib import Path
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import common as c
 
folder = Path(__file__).parent.parent
keluaran = folder / "output"
 
 
def norm_nama(x):
    x = str(x).strip().lower()
    return re.sub(r"^(bu|ibu|pak|bapak|mbak|mas|sdr|sdri)\.?\s+", "", x).strip()
 
 
lapangan = pd.read_csv(keluaran / "keberlangsungan_metrics.csv")
survei2 = pd.read_csv(keluaran / "survei_2_labeled.csv")
raw2 = pd.read_csv(folder / "data" / "survei_2_raw.csv")
raw2.columns = raw2.columns.str.strip()
 
peta_nama = pd.DataFrame({
    "id": [f"R{str(i + 1).zfill(2)}" for i in range(len(raw2))],
    "nama_norm": raw2["Nama Lengkap"].map(norm_nama),
})
 
gabung = (peta_nama.merge(survei2[["id", "label", "label_num"]], on="id")
          .merge(lapangan[["nama_norm", "total_kg", "hari_aktif", "status"]], on="nama_norm")
          .drop_duplicates("id"))
n = len(gabung)
 
print("Validasi label survei vs perilaku lapangan")
print(f"Responden survei tertaut ke data lapangan: {n}/42 "
      f"({n / 42 * 100:.0f}%; sisanya beda ejaan atau tak ada padanan)")
 
gabung["memilah_lapangan"] = (gabung["total_kg"] > 0).astype(int)
 
sudah_survei = gabung["label_num"] == 1
belum_survei = gabung["label_num"] == 0
memilah = gabung["memilah_lapangan"] == 1
 
cocok_sudah = int((sudah_survei & memilah).sum())
klaim_lebih = int((sudah_survei & ~memilah).sum())
cocok_belum = int((belum_survei & ~memilah).sum())
diam_diam = int((belum_survei & memilah).sum())
 
print("\nSurvei 'sudah memilah' vs 'pernah memilah' di lapangan (total kg > 0):")
print(f"                          lapangan: memilah   lapangan: tidak")
print(f"  survei: sudah                {cocok_sudah:>10}        {klaim_lebih:>8}")
print(f"  survei: belum                {diam_diam:>10}        {cocok_belum:>8}")
 
setuju = (cocok_sudah + cocok_belum) / n * 100
kappa = cohen_kappa_score(gabung["label_num"], gabung["memilah_lapangan"])
print(f"\n  Kecocokan  : {setuju:.0f}% ({cocok_sudah + cocok_belum}/{n})")
print(f"  Cohen kappa: {kappa:.3f}")
print(f"  Catatan: '{diam_diam} orang' survei bilang belum tapi di lapangan menyetor -> banyak")
print(f"  yang sudah memilah nyata walau laporan-dirinya belum mengaku rutin.")
 
print("\nSurvei 'sudah memilah' vs status lapangan (3 kelas):")
tab = pd.crosstab(gabung["label"], gabung["status"])
print(tab.to_string())
 
print("\nApakah yang mengaku 'sudah' benar-benar menyetor lebih banyak?")
kg_sudah = gabung.loc[sudah_survei, "total_kg"]
kg_belum = gabung.loc[belum_survei, "total_kg"]
akt_sudah = gabung.loc[sudah_survei, "hari_aktif"]
akt_belum = gabung.loc[belum_survei, "hari_aktif"]
print(f"  total kg    : sudah median {kg_sudah.median():.2f} vs belum median {kg_belum.median():.2f}")
print(f"  hari aktif  : sudah median {akt_sudah.median():.1f} vs belum median {akt_belum.median():.1f}")
if len(kg_sudah) > 0 and len(kg_belum) > 0:
    p_kg = stats.mannwhitneyu(kg_sudah, kg_belum, alternative="greater").pvalue
    p_akt = stats.mannwhitneyu(akt_sudah, akt_belum, alternative="greater").pvalue
    print(f"  Mann-Whitney (sudah > belum): kg p={p_kg:.3f}, hari aktif p={p_akt:.3f}")
    selaras = p_kg < 0.05 or p_akt < 0.05
    print(f"  -> {'Label survei SELARAS dgn perilaku nyata.' if selaras else 'Label survei BELUM terbukti selaras dgn perilaku nyata (laporan-diri != lapangan).'}")
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
 
mat = [[cocok_belum, diam_diam], [klaim_lebih, cocok_sudah]]
im = ax[0].imshow(mat, cmap="Greens")
for (i, j), v in [((i, j), mat[i][j]) for i in range(2) for j in range(2)]:
    ax[0].text(j, i, str(v), ha="center", va="center", fontsize=18, fontweight="bold")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["lapangan:\ntidak", "lapangan:\nmemilah"])
ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["survei:\nbelum", "survei:\nsudah"])
ax[0].set_title(f"Survei vs lapangan (n={n}, kappa={kappa:.2f})", color=c.GELAP, fontweight="bold")
 
data_box = [kg_belum.values, kg_sudah.values]
bp = ax[1].boxplot(data_box, patch_artist=True, widths=0.5)
ax[1].set_xticks([1, 2])
ax[1].set_xticklabels(["survei: belum", "survei: sudah"])
for patch, col in zip(bp["boxes"], [c.ABU, c.GELAP]):
    patch.set_facecolor(col)
for i, d in enumerate(data_box):
    ax[1].scatter([i + 1] * len(d), d, color=c.MERAH, alpha=0.5, s=20, zorder=3)
ax[1].set_ylabel("Total kg terkumpul (lapangan)")
ax[1].set_title("Setoran nyata per label survei", color=c.GELAP, fontweight="bold")
 
fig.tight_layout()
fig.savefig(keluaran / "09_validasi.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
gabung.to_csv(keluaran / "validasi_terlink.csv", index=False)
print(f"\nGambar -> output/09_validasi.png")
print(f"Data   -> output/validasi_terlink.csv")