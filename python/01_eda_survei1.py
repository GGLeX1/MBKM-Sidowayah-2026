from pathlib import Path
import matplotlib.pyplot as plt
import common as c
 
folder = Path(__file__).parent
keluaran = folder / "output"
keluaran.mkdir(exist_ok=True)
 
df = c.muat_survei(folder / "data" / "survei_1_raw_(2).csv")
likert = c.pengetahuan + c.sikap + ["PK1_niat", "PK2_sering", "PK3_kegiatan", "PK4_rutin"]
 
print("EDA Survei 1")
print(f"Responden       : {len(df)}")
print(f"Nilai kosong    : {df[likert + c.hambatan].isnull().sum().sum()}")
 
print("\nRata-rata per kategori (skala 1-5):")
for nama, kolom in [("Pengetahuan", c.pengetahuan), ("Sikap", c.sikap)]:
    print(f"  {nama:<12}: {df[kolom].mean().mean():.2f}")
print(f"  Perilaku    : {df[['PK1_niat', 'PK2_sering', 'PK3_kegiatan', 'PK4_rutin']].mean().mean():.2f}")
 
print("\nItem aksi (dasar pelabelan nanti):")
for k in c.ITEM_AKSI:
    tinggi = (df[k] >= c.AMBANG_AKSI).sum()
    print(f"  {k:<12}: rata-rata {df[k].mean():.2f}, yang menjawab >= {c.AMBANG_AKSI} ada {tinggi} orang")
 
print("\nHambatan (proporsi menjawab 'Ya'):")
for h in c.hambatan:
    print(f"  {h:<12}: {df[h].mean() * 100:.0f}%")
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
 
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
 
rata = df[likert].mean().sort_values()
warna = [c.AKSEN if nama in c.ITEM_AKSI else c.SEDANG for nama in rata.index]
ax[0].barh(rata.index, rata.values, color=warna, edgecolor="white")
ax[0].axvline(c.AMBANG_AKSI, color=c.MERAH, linestyle="--", linewidth=1, label=f"ambang aksi = {c.AMBANG_AKSI}")
ax[0].set_xlabel("Rata-rata jawaban (1-5)")
ax[0].set_title("Rata-rata per item — kuning = item aksi", color=c.GELAP, fontweight="bold")
ax[0].legend(fontsize=8)
 
korr = df[likert].corr()
im = ax[1].imshow(korr, cmap="Greens", vmin=0, vmax=1)
ax[1].set_xticks(range(len(likert)))
ax[1].set_xticklabels(likert, rotation=90, fontsize=7)
ax[1].set_yticks(range(len(likert)))
ax[1].set_yticklabels(likert, fontsize=7)
ax[1].set_title("Korelasi antar item", color=c.GELAP, fontweight="bold")
fig.colorbar(im, ax=ax[1], fraction=0.046)
 
fig.tight_layout()
fig.savefig(keluaran / "01_eda_survei1.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
df.to_csv(keluaran / "survei_1_clean.csv", index=False)
print(f"\nGambar    -> output/01_eda_survei1.png")
print(f"Data siap -> output/survei_1_clean.csv")