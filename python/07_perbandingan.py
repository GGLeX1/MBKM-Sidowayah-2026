from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import common as c
 
folder = Path(__file__).parent.parent
keluaran = folder / "output"
 
s1 = pd.read_csv(keluaran / "survei_1_labeled.csv")
s2 = pd.read_csv(keluaran / "survei_2_labeled.csv")
 
pasang = s1[["id", "label_num", "tingkat"]].merge(
    s2[["id", "label_num", "tingkat"] + c.kolom_reward], on="id", suffixes=("_s1", "_s2"))
 
p1 = s1["label_num"].mean() * 100
p2 = s2["label_num"].mean() * 100
 
print("Perbandingan Survei 1 vs Survei 2")
print(f"Warga yang benar-benar memilah:")
print(f"  Survei 1 : {int(s1['label_num'].sum())}/42 ({p1:.0f}%)")
print(f"  Survei 2 : {int(s2['label_num'].sum())}/42 ({p2:.0f}%)")
print(f"  Kenaikan : {p2 - p1:+.0f} poin persen (dari {p1:.0f}% ke {p2:.0f}%)")
 
pindah_naik = int(((pasang["label_num_s1"] == 0) & (pasang["label_num_s2"] == 1)).sum())
pindah_turun = int(((pasang["label_num_s1"] == 1) & (pasang["label_num_s2"] == 0)).sum())
tetap = len(pasang) - pindah_naik - pindah_turun
print(f"\nPerpindahan per orang (S1 -> S2):")
print(f"  belum -> sudah : {pindah_naik}")
print(f"  sudah -> belum : {pindah_turun}")
print(f"  tetap          : {tetap}")
 
print("\nTiga tingkat, S1 -> S2:")
urut = ["belum_paham", "paham_belum_memilah", "sudah_memilah"]
for t in urut:
    a = int((s1["tingkat"] == t).sum())
    b = int((s2["tingkat"] == t).sum())
    print(f"  {t:<22}: {a} -> {b}")
 
reward_rata = pasang[c.kolom_reward].mean(axis=1)
r_sudah = reward_rata[pasang["label_num_s2"] == 1].mean()
r_belum = reward_rata[pasang["label_num_s2"] == 0].mean()
print(f"\nSikap reward di Survei 2 (rata-rata 1-5):")
print(f"  yang sudah memilah : {r_sudah:.2f}")
print(f"  yang belum memilah : {r_belum:.2f}")
print("  Selisih kecil dan reward umumnya tinggi untuk semua -> reward sulit membedakan siapa yang memilah.")
 
print(f"\nRingkas: pemilah nyata naik dari {p1:.0f}% ke {p2:.0f}% berbarengan dengan masuknya reward.")
print("Ini menunjukkan perubahan yang selaras dengan program, bukan bukti reward sebagai penyebab")
print("(tak ada kelompok tanpa-reward, dan ini masih perilaku laporan-diri).")
 
plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(16, 5))
 
ax[0].bar(["Survei 1", "Survei 2"], [p1, p2], color=[c.MUDA, c.GELAP], edgecolor="white", width=0.6)
for i, v in enumerate([p1, p2]):
    ax[0].text(i, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
ax[0].set_ylim(0, 100)
ax[0].set_ylabel("Persen warga")
ax[0].set_title("Yang benar-benar memilah", color=c.GELAP, fontweight="bold")
 
warna = {"belum_paham": c.ABU, "paham_belum_memilah": c.AKSEN, "sudah_memilah": c.GELAP}
x = range(len(urut))
a1 = [int((s1["tingkat"] == t).sum()) for t in urut]
a2 = [int((s2["tingkat"] == t).sum()) for t in urut]
ax[1].bar([i - 0.2 for i in x], a1, 0.35, label="Survei 1", color=c.MUDA, edgecolor="white")
ax[1].bar([i + 0.2 for i in x], a2, 0.35, label="Survei 2", color=c.GELAP, edgecolor="white")
ax[1].set_xticks(list(x))
ax[1].set_xticklabels(["belum\npaham", "paham,\nbelum memilah", "sudah\nmemilah"], fontsize=8)
ax[1].set_ylabel("Jumlah warga")
ax[1].set_title("Pergeseran tiga tingkat", color=c.GELAP, fontweight="bold")
ax[1].legend(fontsize=8)
 
ax[2].bar(["sudah\nmemilah", "belum\nmemilah"], [r_sudah, r_belum], color=[c.GELAP, c.ABU], edgecolor="white", width=0.6)
for i, v in enumerate([r_sudah, r_belum]):
    ax[2].text(i, v + 0.05, f"{v:.2f}", ha="center", fontweight="bold")
ax[2].set_ylim(0, 5)
ax[2].set_ylabel("Rata-rata sikap reward (1-5)")
ax[2].set_title("Sikap reward per kelompok (Survei 2)", color=c.GELAP, fontweight="bold")
 
fig.tight_layout()
fig.savefig(keluaran / "07_perbandingan.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nGambar -> output/07_perbandingan.png")