import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import common as c
 
folder = Path(__file__).parent.parent
keluaran = folder / "output"
keluaran.mkdir(exist_ok=True)
 
HARI = ["hari 2", "hari 4", "hari 6", "hari 8", "hari 10", "hari 12", "hari 14", "hari 16"]
HARI_ANGKA = [2, 4, 6, 8, 10, 12, 14, 16]
HARI_SETOR = HARI[1:]
 
warna_status = {"konsisten": c.GELAP, "sudah_belum_konsisten": c.AKSEN, "tidak_memilah": c.ABU}
 
 
def norm_nama(x):
    x = str(x).strip().lower()
    return re.sub(r"^(bu|ibu|pak|bapak|mbak|mas|sdr|sdri)\.?\s+", "", x).strip()
 
 
def ke_angka(kolom):
    return kolom.astype(str).str.strip().str.replace(",", ".", regex=False).replace({"nan": "0", "": "0", "-": "0"}).astype(float)
 
 
df = pd.read_csv(folder / "data" / "keberlangsungan_raw.csv")
df.columns = df.columns.str.strip()
df["nama_norm"] = df["nama warga"].map(norm_nama)
df.insert(0, "id_keb", [f"K{str(i + 1).zfill(2)}" for i in range(len(df))])
for h in HARI:
    df[h] = ke_angka(df[h])
 
df["total_kg"] = df[HARI].sum(axis=1)
df["hari_aktif"] = (df[HARI_SETOR] > 0).sum(axis=1)
df["rata_per_hari"] = df["total_kg"] / len(HARI)
 
 
def status(row):
    if row["total_kg"] == 0:
        return "tidak_memilah"
    if all(row[h] > 0 for h in HARI_SETOR):
        return "konsisten"
    return "sudah_belum_konsisten"
 
 
df["status"] = df.apply(status, axis=1)
 
print("Keberlangsungan (data lapangan)")
print(f"Warga dipantau     : {len(df)}")
print(f"Hari pantauan      : {len(HARI)} (hari 2 baseline nol, penyetoran dihitung dari hari 4)")
print(f"Total kg terkumpul : {df['total_kg'].sum():.1f} kg")
print(f"Rata-rata per warga: {df['total_kg'].mean():.2f} kg")
 
print("\nStatus keberlangsungan:")
for s in ["konsisten", "sudah_belum_konsisten", "tidak_memilah"]:
    n = int((df["status"] == s).sum())
    print(f"  {s:<22}: {n} ({n / len(df) * 100:.0f}%)")
 
print("\nPeringkat — paling banyak memilah (total kg):")
for i, r in df.sort_values("total_kg", ascending=False).head(10).iterrows():
    print(f"  {r['nama warga']:<22} {r['total_kg']:>6.2f} kg   ({r['hari_aktif']}/{len(HARI_SETOR)} hari aktif)")
 
print("\nPeringkat — paling rajin memilah (jumlah hari aktif):")
for i, r in df.sort_values(["hari_aktif", "total_kg"], ascending=False).head(10).iterrows():
    print(f"  {r['nama warga']:<22} {r['hari_aktif']}/{len(HARI_SETOR)} hari   ({r['total_kg']:.2f} kg)")
 
tren = [df[h].sum() for h in HARI]
print(f"\nTren total per hari pantauan: {tren[0]:.0f} -> {tren[-1]:.0f} kg "
      f"({'naik' if tren[-1] > tren[1] else 'turun'} sepanjang program)")
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(17, 5))
 
top = df.sort_values("total_kg", ascending=False).head(12).iloc[::-1]
ax[0].barh(top["nama warga"], top["total_kg"], color=[warna_status[s] for s in top["status"]], edgecolor="white")
ax[0].set_xlabel("Total kg terkumpul")
ax[0].set_title("12 warga paling banyak memilah", color=c.GELAP, fontweight="bold")
 
hit = df["status"].value_counts().reindex(["konsisten", "sudah_belum_konsisten", "tidak_memilah"]).fillna(0)
ax[1].bar(range(len(hit)), hit.values, color=[warna_status[s] for s in hit.index], edgecolor="white")
for i, v in enumerate(hit.values):
    ax[1].text(i, v + 0.3, str(int(v)), ha="center", fontweight="bold")
ax[1].set_xticks(range(len(hit)))
ax[1].set_xticklabels(["konsisten", "sudah, belum\nkonsisten", "tidak\nmemilah"], fontsize=9)
ax[1].set_ylabel("Jumlah warga")
ax[1].set_title("Status keberlangsungan", color=c.GELAP, fontweight="bold")
 
ax2b = ax[2].twinx()
ax[2].bar(range(len(HARI)), tren, color=c.MUDA, edgecolor="white")
ax2b.plot(range(len(HARI)), [(df[h] > 0).sum() for h in HARI], color=c.MERAH, marker="o", linewidth=2)
ax[2].set_xticks(range(len(HARI)))
ax[2].set_xticklabels([f"h{a}" for a in HARI_ANGKA], fontsize=8)
ax[2].set_ylabel("Total kg per hari", color=c.SEDANG)
ax2b.set_ylabel("Jumlah warga aktif", color=c.MERAH)
ax[2].set_title("Tren sepanjang program", color=c.GELAP, fontweight="bold")
 
fig.tight_layout()
fig.savefig(keluaran / "08_keberlangsungan.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
simpan = ["id_keb", "nama warga", "nama_norm", "total_kg", "hari_aktif", "rata_per_hari", "status"]
df[simpan].sort_values("total_kg", ascending=False).to_csv(keluaran / "keberlangsungan_metrics.csv", index=False)
print(f"\nGambar -> output/08_keberlangsungan.png")
print(f"Data   -> output/keberlangsungan_metrics.csv")