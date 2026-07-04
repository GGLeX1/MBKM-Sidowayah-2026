from pathlib import Path
import matplotlib.pyplot as plt
import common as c
 
folder = Path(__file__).parent
keluaran = folder / "output"
 
df = c.muat_survei(folder / "data" / "survei_1_raw.csv")
df = c.beri_label(df)
df["tingkat"] = c.tingkat_perilaku(df)
 
aturan = " dan ".join(f"{k} >= {c.AMBANG_AKSI}" for k in c.ITEM_AKSI)
sudah = int(df["label_num"].sum())
belum = len(df) - sudah
 
print("Pelabelan Survei 1")
print(f"Aturan 'sudah memilah': {aturan}")
print(f"  sudah_memilah : {sudah} ({sudah / len(df) * 100:.0f}%)")
print(f"  belum_memilah : {belum} ({belum / len(df) * 100:.0f}%)")
 
print("\nTiga tingkat (pengetahuan x perilaku):")
urut = ["belum_paham", "paham_belum_memilah", "sudah_memilah"]
for t in urut:
    n = int((df["tingkat"] == t).sum())
    print(f"  {t:<22}: {n} ({n / len(df) * 100:.0f}%)")
 
if sudah < 8:
    print(f"\nCatatan: kelas 'sudah' cuma {sudah} orang — model prediksi survei ini akan tidak stabil.")
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(8, 5))
warna = {"belum_paham": c.ABU, "paham_belum_memilah": c.AKSEN, "sudah_memilah": c.GELAP}
jumlah = [int((df["tingkat"] == t).sum()) for t in urut]
ax.bar(range(len(urut)), jumlah, color=[warna[t] for t in urut], edgecolor="white")
for i, n in enumerate(jumlah):
    ax.text(i, n + 0.3, str(n), ha="center", fontweight="bold")
ax.set_xticks(range(len(urut)))
ax.set_xticklabels(["belum paham", "paham,\nbelum memilah", "sudah\nmemilah"])
ax.set_ylabel("Jumlah warga")
ax.set_title("Survei 1 — dari paham menuju melakukan", color=c.GELAP, fontweight="bold")
fig.tight_layout()
fig.savefig(keluaran / "02_tingkat_survei1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
kolom = ["id", "Umur", "Pendidikan Terakhir"] + c.fitur_survei1 + ["PK1_niat", "PK2_sering", "PK3_kegiatan", "PK4_rutin", "label", "label_num", "tingkat"]
df[[k for k in kolom if k in df.columns]].to_csv(keluaran / "survei_1_labeled.csv", index=False)
print(f"\nGambar -> output/02_tingkat_survei1.png")
print(f"Data   -> output/survei_1_labeled.csv")