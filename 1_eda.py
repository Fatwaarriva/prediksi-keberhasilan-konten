# ============================================================
#   MODUL AHP LENGKAP + UJI KONSISTENSI
#                   + PERBANDINGAN THRESHOLD (Q2 / Q3 / Mean)
#
#   Script BERDIRI SENDIRI — load langsung dari file mentah
#   Input : data scraping instagram.xlsx + data_transaksi.xlsx
#   Output: ahp_report.txt | threshold_comparison.xlsx
#           threshold_comparison.png
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# [0] KONFIGURASI — sesuaikan nama file jika berbeda
# ────────────────────────────────────────────────────────────
NAMA_FILE_INSTAGRAM = 'data/data scraping instagram.xlsx'
NAMA_FILE_TRANSAKSI = 'data/data_transaksi.xlsx'

BOBOT_TRANSAKSI = 0.75
BOBOT_INTERAKSI = 0.25

# ────────────────────────────────────────────────────────────
# [1] LOAD DATA
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODUL AHP LENGKAP + PERBANDINGAN THRESHOLD")
print("=" * 60)

try:
    df_ig    = pd.read_excel(NAMA_FILE_INSTAGRAM, engine='openpyxl')
    df_trans = pd.read_excel(NAMA_FILE_TRANSAKSI, engine='openpyxl')
    print(f"✅ Instagram : {len(df_ig)} baris | Kolom: {list(df_ig.columns)}")
    print(f"✅ Transaksi : {len(df_trans)} baris | Kolom: {list(df_trans.columns)}")
except Exception as e:
    print(f"❌ Gagal membaca file: {e}")
    exit()

# ────────────────────────────────────────────────────────────
# [2] AUTO-DETECT KOLOM
# ────────────────────────────────────────────────────────────
def cari_kolom(df, kata_kunci_list):
    for kata in kata_kunci_list:
        hasil = [c for c in df.columns if kata.lower() in c.lower()]
        if hasil:
            return hasil[0]
    return None

col_tanggal  = cari_kolom(df_ig,    ['date','tanggal','tgl','time','waktu'])
col_likes    = cari_kolom(df_ig,    ['like','suka'])
col_comments = cari_kolom(df_ig,    ['comment','komen','komentar'])
col_type     = cari_kolom(df_ig,    ['type','tipe','jenis','kind','format'])
col_bulan    = cari_kolom(df_trans, ['bulan','month','tanggal','date','periode'])
col_order    = cari_kolom(df_trans, ['order','pesanan','penjualan','sales','transaksi','jumlah'])
col_tahun    = cari_kolom(df_trans, ['tahun','year'])

print(f"\n🔍 Kolom terdeteksi:")
print(f"   Tanggal      → {col_tanggal}")
print(f"   Likes        → {col_likes}")
print(f"   Comments     → {col_comments}")
print(f"   Tipe Konten  → {col_type}")
print(f"   Bulan        → {col_bulan}")
print(f"   Jumlah Order → {col_order}")
print(f"   Tahun        → {col_tahun}")

for nama, val in [('Tanggal', col_tanggal), ('Likes', col_likes),
                  ('Comments', col_comments), ('Bulan', col_bulan),
                  ('Jumlah Order', col_order)]:
    if not val:
        print(f"\n❌ Kolom '{nama}' tidak ditemukan! Cek nama kolom di file Excel.")
        exit()

# ────────────────────────────────────────────────────────────
# [3] PREPROCESSING
# ────────────────────────────────────────────────────────────
print("\n🔧 Preprocessing...")

# Instagram
df_ig['dt']        = pd.to_datetime(df_ig[col_tanggal], errors='coerce')
df_ig              = df_ig.dropna(subset=['dt']).drop_duplicates()
df_ig['Bulan_Num'] = df_ig['dt'].dt.month
df_ig['Tahun']     = df_ig['dt'].dt.year
df_ig['likes']     = pd.to_numeric(df_ig[col_likes],    errors='coerce').fillna(0)
df_ig['comments']  = pd.to_numeric(df_ig[col_comments], errors='coerce').fillna(0)
df_ig['total_interaction'] = df_ig['likes'] + df_ig['comments']

# Transaksi
bulan_map = {
    'januari':1,'februari':2,'maret':3,'april':4,
    'mei':5,'juni':6,'juli':7,'agustus':8,
    'september':9,'oktober':10,'november':11,'desember':12
}
df_trans['Bulan_Num'] = df_trans[col_bulan].astype(str).str.lower().str.strip().map(bulan_map)

if col_tahun:
    df_trans['Tahun'] = pd.to_numeric(df_trans[col_tahun], errors='coerce')
else:
    tahun_modus = df_ig['Tahun'].mode()[0]
    df_trans['Tahun'] = tahun_modus
    print(f"   ⚠️  Kolom Tahun tidak ada di transaksi, menggunakan tahun modus: {tahun_modus}")

df_trans[col_order] = pd.to_numeric(df_trans[col_order], errors='coerce')

df_trans_agg = (df_trans
                .groupby(['Bulan_Num','Tahun'])[col_order]
                .sum()
                .reset_index()
                .rename(columns={col_order: 'Jumlah_Order'}))

# Merge
df = pd.merge(df_ig, df_trans_agg, on=['Bulan_Num','Tahun'], how='left')
df['Jumlah_Order'] = df['Jumlah_Order'].fillna(0)

# Normalisasi Min-Max
scaler = MinMaxScaler()
df[['Interaksi_Norm','Order_Norm']] = scaler.fit_transform(
    df[['total_interaction','Jumlah_Order']]
)

# Total Score AHP
df['Total_Score'] = (df['Order_Norm'] * BOBOT_TRANSAKSI) + \
                    (df['Interaksi_Norm'] * BOBOT_INTERAKSI)

print(f"✅ Data siap: {len(df)} baris")

# ════════════════════════════════════════════════════════════
#  BAGIAN 1 — PERHITUNGAN AHP LENGKAP
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  BAGIAN 1 — PERHITUNGAN AHP (Analytical Hierarchy Process)")
print("═" * 60)

# ── 1.1 Matriks Perbandingan Berpasangan ──
KRITERIA   = ['Transaksi', 'Engagement']
N_KRITERIA = len(KRITERIA)

matriks_ahp = np.array([
    [1,      3   ],   # Transaksi "Agak Lebih Penting" (Skala 3) vs Engagement
    [1/3,    1   ]
])

print("\n📋 Matriks Perbandingan Berpasangan:")
print(f"   {'Kriteria':<14} {'Transaksi':>12} {'Engagement':>12}")
print("   " + "-" * 40)
for i, baris in enumerate(KRITERIA):
    print(f"   {baris:<14} {matriks_ahp[i, 0]:>12.4f} {matriks_ahp[i, 1]:>12.4f}")

# ── 1.2 Normalisasi dan Eigen Vector ──
jumlah_kolom    = matriks_ahp.sum(axis=0)
matriks_norm    = matriks_ahp / jumlah_kolom
bobot_prioritas = matriks_norm.mean(axis=1)

print("\n📊 Jumlah Per Kolom:")
for j, kol in enumerate(KRITERIA):
    print(f"   Kolom {kol:<12}: {jumlah_kolom[j]:.4f}")

print("\n📊 Matriks Ternormalisasi + Bobot Prioritas:")
print(f"   {'Kriteria':<14} {'Kol. Transaksi':>15} {'Kol. Engagement':>17} {'Bobot (Wi)':>12}")
print("   " + "-" * 60)
for i, baris in enumerate(KRITERIA):
    print(f"   {baris:<14} {matriks_norm[i,0]:>15.4f} {matriks_norm[i,1]:>17.4f} {bobot_prioritas[i]:>11.4f}")

print(f"\n   ✅ Bobot Transaksi  (W₁) = {bobot_prioritas[0]:.4f} ({bobot_prioritas[0]*100:.1f}%)")
print(f"   ✅ Bobot Engagement (W₂) = {bobot_prioritas[1]:.4f} ({bobot_prioritas[1]*100:.1f}%)")

# ── 1.3 Uji Konsistensi ──
print("\n" + "─" * 60)
print("  UJI KONSISTENSI (Consistency Ratio — CR)")
print("─" * 60)

weighted_sum     = matriks_ahp @ bobot_prioritas
lambda_per_baris = weighted_sum / bobot_prioritas
lambda_max       = lambda_per_baris.mean()

CI = (lambda_max - N_KRITERIA) / (N_KRITERIA - 1) if N_KRITERIA > 1 else 0

RI_TABLE = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12,
            6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
RI = RI_TABLE.get(N_KRITERIA, 1.49)
CR = (CI / RI) if RI != 0 else 0.0

print(f"\n   Jumlah Kriteria (n)     : {N_KRITERIA}")
print(f"\n   Weighted Sum Vector:")
for i, k in enumerate(KRITERIA):
    print(f"     {k:<14}: {weighted_sum[i]:.4f} / {bobot_prioritas[i]:.4f} = {lambda_per_baris[i]:.4f}")

print(f"\n   λ_max (rata-rata rasio)  : {lambda_max:.6f}")
print(f"   CI  (Consistency Index)  : {CI:.6f}")
print(f"   RI  (Random Index, n={N_KRITERIA}) : {RI:.2f}")
print(f"   CR  (Consistency Ratio)  : {CR:.6f}")

if CR <= 0.10:
    print(f"\n   ✅ CR = {CR:.4f} ≤ 0.10 → Matriks KONSISTEN dan VALID")
else:
    print(f"\n   ⚠️  CR = {CR:.4f} > 0.10 → Perlu revisi perbandingan berpasangan!")

print(f"\n   ℹ️  Catatan: Untuk n=2, RI=0.00 sehingga CR selalu 0 secara matematis.")
print(f"      Konsistensi dijamin karena hanya ada satu perbandingan unik (a₁₂ = {matriks_ahp[0,1]:.0f}).")

# ════════════════════════════════════════════════════════════
#  BAGIAN 2 — PERBANDINGAN TIGA THRESHOLD
# ════════════════════════════════════════════════════════════
print("\n" + "═" * 60)
print("  BAGIAN 2 — PERBANDINGAN THRESHOLD (Q2 / Q3 / Mean)")
print("═" * 60)

total_score    = df['Total_Score']
threshold_Q2   = total_score.quantile(0.50)
threshold_Q3   = total_score.quantile(0.75)
threshold_Mean = total_score.mean()

thresholds = {
    'Q2 (Median)': threshold_Q2,
    'Q3'         : threshold_Q3,
    'Mean'       : threshold_Mean,
}

print(f"\n   Total Score — Statistik Dasar:")
print(f"     Min    : {total_score.min():.4f}")
print(f"     Q1     : {total_score.quantile(0.25):.4f}")
print(f"     Median : {threshold_Q2:.4f}")
print(f"     Mean   : {threshold_Mean:.4f}")
print(f"     Q3     : {threshold_Q3:.4f}")
print(f"     Max    : {total_score.max():.4f}")
print(f"     Std    : {total_score.std():.4f}")

print(f"\n{'─'*72}")
print(f"  {'Threshold':<16} {'Nilai':>8} {'Sukses':>10} {'Gagal':>10} {'Rasio S:G':>11} {'Imbalance':>11}")
print(f"{'─'*72}")

hasil_threshold = []
for nama, nilai in thresholds.items():
    label  = (df['Total_Score'] >= nilai).astype(int)
    ns     = label.sum()
    ng     = len(df) - ns
    rasio  = ns / ng if ng > 0 else float('inf')
    imbal  = abs(ns - ng) / len(df) * 100
    status = "✅ Seimbang" if imbal < 15 else ("⚠️  Agak Timpang" if imbal < 35 else "❌ Timpang")

    print(f"  {nama:<16} {nilai:>8.4f} {ns:>7} ({ns/len(df)*100:4.1f}%) "
          f"{ng:>7} ({ng/len(df)*100:4.1f}%)  {rasio:>8.2f}   {imbal:>6.1f}%  {status}")

    hasil_threshold.append({
        'Threshold'     : nama,
        'Nilai'         : round(nilai, 4),
        'N_Sukses'      : int(ns),
        'Pct_Sukses'    : round(ns/len(df)*100, 1),
        'N_Gagal'       : int(ng),
        'Pct_Gagal'     : round(ng/len(df)*100, 1),
        'Rasio_S_G'     : round(rasio, 3),
        'Imbalance_Pct' : round(imbal, 1),
    })

print(f"{'─'*72}")
print(f"\n   ℹ️  Threshold Q3 dipilih dalam penelitian ini berdasarkan evaluasi")
print(f"      model Random Forest (akurasi + F1-Score tertinggi).")

# ════════════════════════════════════════════════════════════
#  BAGIAN 3 — SIMPAN OUTPUT
# ════════════════════════════════════════════════════════════
print("\n📁 Menyimpan hasil...")

# Excel
df_export = pd.DataFrame([{
    'Threshold'      : h['Threshold'],
    'Nilai'          : h['Nilai'],
    'N_Sukses'       : h['N_Sukses'],
    'Pct_Sukses (%)' : h['Pct_Sukses'],
    'N_Gagal'        : h['N_Gagal'],
    'Pct_Gagal (%)'  : h['Pct_Gagal'],
    'Rasio S:G'      : h['Rasio_S_G'],
    'Imbalance (%)'  : h['Imbalance_Pct'],
} for h in hasil_threshold])
df_export.to_excel('threshold_comparison.xlsx', index=False)
print("✅ threshold_comparison.xlsx tersimpan")

# Teks
with open('ahp_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n  LAPORAN AHP + UJI KONSISTENSI\n" + "=" * 60 + "\n\n")
    f.write("MATRIKS PERBANDINGAN BERPASANGAN\n")
    for i, b in enumerate(KRITERIA):
        for j, k in enumerate(KRITERIA):
            f.write(f"  {b} vs {k}: {matriks_ahp[i,j]:.4f}\n")
    f.write("\nBOBOT PRIORITAS (EIGEN VECTOR)\n")
    for i, k in enumerate(KRITERIA):
        f.write(f"  {k}: {bobot_prioritas[i]:.4f} ({bobot_prioritas[i]*100:.1f}%)\n")
    f.write(f"\nUJI KONSISTENSI\n  lambda_max={lambda_max:.6f} | CI={CI:.6f} | RI={RI:.2f} | CR={CR:.6f}\n")
    f.write(f"  Status: {'KONSISTEN' if CR <= 0.10 else 'TIDAK KONSISTEN'}\n\n")
    f.write("PERBANDINGAN THRESHOLD\n")
    for h in hasil_threshold:
        f.write(f"  {h['Threshold']:<16}: nilai={h['Nilai']:.4f} | "
                f"sukses={h['N_Sukses']} ({h['Pct_Sukses']}%) | "
                f"gagal={h['N_Gagal']} ({h['Pct_Gagal']}%) | "
                f"imbalance={h['Imbalance_Pct']}%\n")
print("✅ ahp_report.txt tersimpan")

# Grafik
print("📊 Membuat grafik...")
PALETTE = ['#3A86FF', '#8338EC', '#FF9F1C']
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle('Perbandingan Threshold: Q2 (Median) vs Q3 vs Mean\n'
             'Distribusi Total Score AHP (75% Transaksi + 25% Engagement)',
             fontsize=13, fontweight='bold', y=0.98)

ax_hist = fig.add_subplot(gs[0, :])
ax_hist.hist(df['Total_Score'], bins=40, color='#2EC4B6', edgecolor='white',
             linewidth=0.5, alpha=0.75, label='Total Score')
ax_hist.axvline(threshold_Q2,   color=PALETTE[0], linestyle='--',  linewidth=2,
                label=f'Q2/Median = {threshold_Q2:.3f}')
ax_hist.axvline(threshold_Q3,   color=PALETTE[1], linestyle='-.',  linewidth=2,
                label=f'Q3       = {threshold_Q3:.3f}')
ax_hist.axvline(threshold_Mean, color=PALETTE[2], linestyle=':',   linewidth=2.5,
                label=f'Mean     = {threshold_Mean:.3f}')
ax_hist.set_title('Distribusi Total Score dengan Tiga Kandidat Threshold',
                  fontsize=11, fontweight='bold')
ax_hist.set_xlabel('Total Score')
ax_hist.set_ylabel('Jumlah Postingan')
ax_hist.legend(fontsize=9, loc='upper right')

nama_labels = ['Q2\n(Median)', 'Q3', 'Mean']
for idx, (h, nama) in enumerate(zip(hasil_threshold, nama_labels)):
    ax = fig.add_subplot(gs[1, idx])
    bars = ax.bar(['Gagal (0)', 'Sukses (1)'],
                  [h['N_Gagal'], h['N_Sukses']],
                  color=['#E84855', '#2EC4B6'],
                  edgecolor='white', linewidth=1.2, width=0.55)
    for bar in bars:
        h_ = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h_ + 0.5,
                str(int(h_)), ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_title(f'Threshold: {h["Threshold"]}\nNilai = {h["Nilai"]:.4f} | '
                 f'Imbalance = {h["Imbalance_Pct"]:.1f}%',
                 fontsize=9.5, fontweight='bold')
    ax.set_ylabel('Jumlah Postingan')
    ax.set_ylim(0, max(h['N_Gagal'], h['N_Sukses']) * 1.2)
    ax.text(0.5, 0.92, f"Rasio S:G = {h['Rasio_S_G']:.2f}",
            ha='center', transform=ax.transAxes, fontsize=9, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0',
                      edgecolor='#cccccc', alpha=0.8))

plt.savefig('threshold_comparison.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
print("✅ threshold_comparison.png tersimpan")

print("\n" + "=" * 60)
print("  🎉 Selesai! File yang dihasilkan:")
print("     • ahp_report.txt            — laporan teks AHP lengkap")
print("     • threshold_comparison.xlsx — tabel perbandingan threshold")
print("     • threshold_comparison.png  — grafik perbandingan threshold")
print("=" * 60)