from pathlib import Path
import matplotlib.pyplot as plt
import common as c
 
folder = Path(__file__).parent.parent
keluaran = folder / "output"
 
df = c.muat_survei(folder / "data" / "survei_2_raw.csv", pakai_reward=True)
likert = c.pengetahuan + c.sikap + ["PK1_niat", "PK2_sering", "PK3_kegiatan", "PK4_rutin"]
 
print("EDA Survei 2")
print(f"Responden    : {len(df)}")
print(f"Nilai kosong : {df[likert + c.hambatan + c.kolom_reward].isnull().sum().sum()}")
 
print("\nRata-rata per kategori (skala 1-5):")
print(f"  Pengetahuan : {df[c.pengetahuan].mean().mean():.2f}")
print(f"  Sikap       : {df[c.sikap].mean().mean():.2f}")
print(f"  Perilaku    : {df[['PK1_niat', 'PK2_sering', 'PK3_kegiatan', 'PK4_rutin']].mean().mean():.2f}")
 
print("\nItem aksi:")
for k in c.ITEM_AKSI:
    tinggi = (df[k] >= c.AMBANG_AKSI).sum()
    print(f"  {k:<12}: rata-rata {df[k].mean():.2f}, menjawab >= {c.AMBANG_AKSI} ada {tinggi} orang")
 
print("\nReward (skala 1-5):")
for r in c.kolom_reward:
    tinggi = (df[r] >= 4).mean() * 100
    print(f"  {r:<14}: rata-rata {df[r].mean():.2f}, yang menjawab tinggi (>=4) {tinggi:.0f}%")
print("  -> kalau sebagian besar menjawab tinggi, reward kurang bervariasi antar warga.")
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
 
rata = df[likert].mean().sort_values()
warna = [c.AKSEN if nama in c.ITEM_AKSI else c.SEDANG for nama in rata.index]
ax[0].barh(rata.index, rata.values, color=warna, edgecolor="white")
ax[0].axvline(c.AMBANG_AKSI, color=c.MERAH, linestyle="--", linewidth=1)
ax[0].set_xlabel("Rata-rata jawaban (1-5)")
ax[0].set_title("Rata-rata per item — kuning = item aksi", color=c.GELAP, fontweight="bold")
 
for r in c.kolom_reward:
    sebaran = df[r].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    ax[1].plot([1, 2, 3, 4, 5], sebaran.values, marker="o", label=r)
ax[1].set_xlabel("Skor reward (1-5)")
ax[1].set_ylabel("Jumlah warga")
ax[1].set_title("Sebaran jawaban reward", color=c.GELAP, fontweight="bold")
ax[1].legend(fontsize=8)
 
fig.tight_layout()
fig.savefig(keluaran / "04_eda_survei2.png", dpi=150, bbox_inches="tight")
plt.close(fig)
 
df.to_csv(keluaran / "survei_2_clean.csv", index=False)
print(f"\nGambar    -> output/04_eda_survei2.png")
print(f"Data siap -> output/survei_2_clean.csv")