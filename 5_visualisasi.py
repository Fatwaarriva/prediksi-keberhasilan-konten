# ============================================================
#   VISUALISASI LAPORAN — RANDOM FOREST
#   Input : data_siap_training_v6.csv + model_rf_final.pkl
#   Output: Grafik tampil LANGSUNG di Colab + disimpan PNG
#
#   Grafik 1: Feature Importance
#   Grafik 2: Confusion Matrix
#   Grafik 3: Peluang Sukses per Waktu Posting
#   Grafik 4: Efek Tanggal Gajian
#   Grafik 5: Pola Musiman per Bulan
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from IPython.display import display
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ────────────────────────────────────────────────────────────
# [0] KONFIGURASI
# ────────────────────────────────────────────────────────────
NAMA_FILE_DATA  = 'data_siap_training_v6.csv'
NAMA_FILE_MODEL = 'model_rf_final.pkl'

# ────────────────────────────────────────────────────────────
# [1] LOAD DATA & MODEL
# ────────────────────────────────────────────────────────────
print("=" * 55)
print("  VISUALISASI LAPORAN — RANDOM FOREST")
print("=" * 55)

try:
    df    = pd.read_csv(NAMA_FILE_DATA)
    model = joblib.load(NAMA_FILE_MODEL)
    print(f"✅ Data  : {len(df)} baris | Kolom: {list(df.columns)}")
    print(f"✅ Model : {NAMA_FILE_MODEL} berhasil dimuat")
except FileNotFoundError as e:
    print(f"❌ File tidak ditemukan: {e}")
    print("   Pastikan sudah menjalankan:")
    print("   1. preprocessing_v6.py")
    print("   2. random_forest_final.py")
    raise

# Split — HARUS SAMA PERSIS dengan training
X = df.drop(columns=['is_success'])
y = df['is_success']

_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
y_pred = model.predict(X_test)
cm     = confusion_matrix(y_test, y_pred)
acc    = (cm[0][0] + cm[1][1]) / cm.sum()

sns.set(style="whitegrid")
PALETTE   = ['#2EC4B6', '#E84855', '#FF9F1C', '#3A86FF', '#8338EC']
nama_bulan = {
    1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr',
    5:'Mei', 6:'Jun', 7:'Jul', 8:'Agu',
    9:'Sep', 10:'Okt', 11:'Nov', 12:'Des'
}
time_map = {
    0: 'Pagi\n(05-11)',
    1: 'Siang\n(11-15)',
    2: 'Sore\n(15-18)',
    3: 'Malam\n(18-05)'
}

# Helper: tambah nilai di atas bar
def label_bars(ax):
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}',
                        (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

# ════════════════════════════════════════════════════════════
#  GRAFIK 1 — Feature Importance
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("  📊 GRAFIK 1 — Feature Importance")
print("─" * 55)

importances = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

colors = [PALETTE[0] if i < 3 else '#BBBBBB'
          for i in range(len(importances))]

fig1, ax1 = plt.subplots(figsize=(10, 6))
bars = ax1.barh(importances.index[::-1],
                importances.values[::-1],
                color=colors[::-1],
                edgecolor='white', linewidth=0.8)

for bar, val in zip(bars, importances.values[::-1]):
    ax1.text(bar.get_width() + 0.001,
             bar.get_y() + bar.get_height() / 2,
             f'{val:.4f}', va='center', fontsize=9)

ax1.set_title('Faktor Paling Berpengaruh Terhadap Kesuksesan Konten\n'
              '(Label AHP: 75% Transaksi + 25% Engagement)',
              fontsize=13, fontweight='bold')
ax1.set_xlabel('Tingkat Kepentingan (Importance Score)', fontsize=11)
ax1.set_ylabel('Fitur', fontsize=11)
plt.tight_layout()
plt.savefig('grafik_1_feature_importance.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Grafik 1 tampil + disimpan → grafik_1_feature_importance.png")

# ════════════════════════════════════════════════════════════
#  GRAFIK 2 — Confusion Matrix
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("  📊 GRAFIK 2 — Confusion Matrix")
print("─" * 55)

fig2, ax2 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Prediksi Gagal', 'Prediksi Sukses'],
            yticklabels=['Asli Gagal', 'Asli Sukses'],
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 14, "weight": "bold"})
ax2.set_title(f'Confusion Matrix\nAkurasi: {acc*100:.2f}%',
              fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('grafik_2_confusion_matrix.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Grafik 2 tampil + disimpan → grafik_2_confusion_matrix.png")

# ════════════════════════════════════════════════════════════
#  GRAFIK 3 — Peluang Sukses per Waktu Posting
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("  📊 GRAFIK 3 — Peluang Sukses per Waktu Posting")
print("─" * 55)

if 'time_category' not in df.columns:
    print("⚠️  Kolom 'time_category' tidak ada — Grafik 3 dilewati")
else:
    df['Waktu']   = df['time_category'].map(time_map)
    order_waktu   = ['Pagi\n(05-11)', 'Siang\n(11-15)',
                     'Sore\n(15-18)', 'Malam\n(18-05)']

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    sns.barplot(x='Waktu', y='is_success', data=df,
                order=order_waktu, palette='coolwarm',
                errorbar=None, ax=ax3)
    label_bars(ax3)
    ax3.set_title('Peluang Sukses Berdasarkan Waktu Posting\n'
                  '(Sukses = Tinggi Penjualan + Engagement)',
                  fontsize=13, fontweight='bold')
    ax3.set_ylabel('Rata-rata Peluang Sukses (0–1)', fontsize=11)
    ax3.set_xlabel('Waktu Posting', fontsize=11)
    ax3.set_ylim(0, df.groupby('Waktu')['is_success'].mean().max() * 1.3)
    plt.tight_layout()
    plt.savefig('grafik_3_waktu_terbaik.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("✅ Grafik 3 tampil + disimpan → grafik_3_waktu_terbaik.png")

# ════════════════════════════════════════════════════════════
#  GRAFIK 4 — Efek Tanggal Gajian
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("  📊 GRAFIK 4 — Efek Tanggal Gajian")
print("─" * 55)

if 'is_payday' not in df.columns:
    print("⚠️  Kolom 'is_payday' tidak ada — Grafik 4 dilewati")
else:
    fig4, ax4 = plt.subplots(figsize=(6, 5))
    sns.barplot(x='is_payday', y='is_success', data=df,
                palette='Set2', errorbar=None, ax=ax4)
    ax4.set_xticklabels(['Tanggal Biasa', 'Tanggal Gajian\n(tgl 25–5)'])
    label_bars(ax4)
    ax4.set_title('Apakah Tanggal Gajian Berpengaruh\n'
                  'terhadap Keberhasilan Konten?',
                  fontsize=13, fontweight='bold')
    ax4.set_ylabel('Rata-rata Peluang Sukses (0–1)', fontsize=11)
    ax4.set_xlabel('')
    ax4.set_ylim(0, df.groupby('is_payday')['is_success'].mean().max() * 1.35)
    plt.tight_layout()
    plt.savefig('grafik_4_efek_gajian.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("✅ Grafik 4 tampil + disimpan → grafik_4_efek_gajian.png")

# ════════════════════════════════════════════════════════════
#  GRAFIK 5 Tipe Konten vs Peluang Sukses
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("  📊 GRAFIK 5 Tipe Konten vs Peluang Sukses")
print("─" * 55)

df['Tipe_Konten'] = df['type_encoded'].map({1: 'Video/Reels', 0: 'Foto/Carousel'})

fig7, axes7 = plt.subplots(1, 2, figsize=(13, 6))
colors_pie = ['#2EC4B6', '#FF9F1C']

# Pie chart kiri — Proporsi tipe konten keseluruhan
tipe_count = df['Tipe_Konten'].value_counts()
wedges1, texts1, autotexts1 = axes7[0].pie(
    tipe_count.values,
    labels=tipe_count.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 3},
    textprops={'fontsize': 11},
    pctdistance=0.75,
    explode=(0.05, 0.05)
)
for at in autotexts1:
    at.set_fontweight('bold')
    at.set_fontsize(12)
axes7[0].set_title('Proporsi Tipe Konten\nKeseluruhan',
                   fontsize=12, fontweight='bold', pad=15)

# Tambah lingkaran tengah (donut effect)
centre_circle1 = plt.Circle((0,0), 0.55, fc='white')
axes7[0].add_patch(centre_circle1)
axes7[0].text(0, 0, f'Total\n{len(df):,}',
              ha='center', va='center',
              fontsize=11, fontweight='bold', color='#333333')

# Pie chart kanan — Proporsi tipe konten yang Sukses
df_sukses = df[df['is_success'] == 1]
tipe_sukses = df_sukses['Tipe_Konten'].value_counts()
wedges2, texts2, autotexts2 = axes7[1].pie(
    tipe_sukses.values,
    labels=tipe_sukses.index,
    autopct='%1.1f%%',
    colors=colors_pie,
    startangle=90,
    wedgeprops={'edgecolor': 'white', 'linewidth': 3},
    textprops={'fontsize': 11},
    pctdistance=0.75,
    explode=(0.05, 0.05)
)
for at in autotexts2:
    at.set_fontweight('bold')
    at.set_fontsize(12)
axes7[1].set_title('Proporsi Tipe Konten\npada Postingan Sukses',
                   fontsize=12, fontweight='bold', pad=15)

# Tambah lingkaran tengah (donut effect)
centre_circle2 = plt.Circle((0,0), 0.55, fc='white')
axes7[1].add_patch(centre_circle2)
axes7[1].text(0, 0, f'Sukses\n{len(df_sukses):,}',
              ha='center', va='center',
              fontsize=11, fontweight='bold', color='#333333')

fig7.suptitle('Perbandingan Tipe Konten: Keseluruhan vs Postingan Sukses',
              fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('grafik_7_tipe_konten.png', dpi=300,
            bbox_inches='tight', facecolor='white')
plt.show()
print("✅ Grafik 7 tampil + disimpan → grafik_7_tipe_konten.png")


# ════════════════════════════════════════════════════════════
#  GRAFIK 6 Tren Order per Bulan
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 55)
print("  📊 GRAFIK 6 Tren Order per Bulan")
print("─" * 55)

try:
    df_trans = pd.read_excel('data/data_transaksi.xlsx', engine='openpyxl')

    bulan_map_str = {
        'januari':1,'februari':2,'maret':3,'april':4,
        'mei':5,'juni':6,'juli':7,'agustus':8,
        'september':9,'oktober':10,'november':11,'desember':12
    }

    col_bulan = [c for c in df_trans.columns if any(k in c.lower() for k in ['bulan','month'])][0]
    col_order = [c for c in df_trans.columns if any(k in c.lower() for k in ['order','pesanan','jumlah','sales'])][0]

    df_trans['Bulan_Num'] = df_trans[col_bulan].astype(str).str.lower().str.strip().map(bulan_map_str)
    df_trans[col_order]   = pd.to_numeric(df_trans[col_order], errors='coerce')

    df_tren = df_trans.groupby('Bulan_Num')[col_order].sum().reset_index()
    df_tren['Nama_Bulan'] = df_tren['Bulan_Num'].map(nama_bulan)
    df_tren = df_tren.sort_values('Bulan_Num')

    fig8, ax8 = plt.subplots(figsize=(13, 6))

    # Gradient bar + line
    bar_colors8 = []
    max_val = df_tren[col_order].max()
    for val in df_tren[col_order]:
        intensity = val / max_val
        bar_colors8.append((0.18, intensity * 0.76 + 0.1, intensity * 0.7 + 0.15))

    bars8 = ax8.bar(df_tren['Nama_Bulan'], df_tren[col_order],
                    color='#2EC4B6', alpha=0.3,
                    edgecolor='#2EC4B6', linewidth=1.5, width=0.6)

    ax8.plot(df_tren['Nama_Bulan'], df_tren[col_order],
             marker='o', color='#2EC4B6', linewidth=2.5,
             markersize=9, markerfacecolor='white',
             markeredgewidth=2.5, zorder=5)

    ax8.fill_between(range(len(df_tren)), df_tren[col_order],
                     alpha=0.15, color='#2EC4B6')

    # Highlight bulan tertinggi
    idx_max = df_tren[col_order].idxmax()
    ax8.bar(df_tren.loc[idx_max, 'Nama_Bulan'],
            df_tren.loc[idx_max, col_order],
            color='#FF6584', alpha=0.8,
            edgecolor='#FF6584', linewidth=1.5, width=0.6,
            label='Bulan Tertinggi', zorder=4)

    for i, row in df_tren.iterrows():
        ax8.annotate(f"{int(row[col_order]):,}",
                     (row['Nama_Bulan'], row[col_order]),
                     textcoords="offset points", xytext=(0, 12),
                     ha='center', fontsize=9, fontweight='bold',
                     color='#333333')

    ax8.set_title('Tren Jumlah Order per Bulan\nData Transaksi Tifahampers',
                  fontsize=14, fontweight='bold', pad=15)
    ax8.set_ylabel('Jumlah Order', fontsize=11)
    ax8.set_xlabel('Bulan', fontsize=11)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    ax8.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax8.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('grafik_8_tren_order.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("✅ Grafik 8 tampil + disimpan → grafik_8_tren_order.png")

    # ════════════════════════════════════════════════════════
    #  GRAFIK 7 Order Tertinggi vs Terendah
    # ════════════════════════════════════════════════════════
    print("\n" + "─" * 55)
    print("  📊 GRAFIK TAMBAHAN 4 — Order Tertinggi vs Terendah")
    print("─" * 55)

    df_tren_sorted = df_tren.sort_values(col_order, ascending=False)
    top3   = df_tren_sorted.head(3)
    bot3   = df_tren_sorted.tail(3)
    df_cmp = pd.concat([top3, bot3])
    colors9 = ['#2EC4B6', '#36CFC9', '#5CDBD3'] + ['#FF6584', '#FF85A1', '#FFB3C1']

    fig9, ax9 = plt.subplots(figsize=(11, 6))
    bars9 = ax9.bar(df_cmp['Nama_Bulan'], df_cmp[col_order],
                    color=colors9, edgecolor='white',
                    linewidth=2, width=0.6)

    for bar, val in zip(bars9, df_cmp[col_order]):
        ax9.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(df_cmp[col_order]) * 0.01,
                 f'{int(val):,}',
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2EC4B6', label='3 Bulan Tertinggi'),
        Patch(facecolor='#FF6584', label='3 Bulan Terendah')
    ]
    ax9.legend(handles=legend_elements, fontsize=11,
               loc='upper right', framealpha=0.9)
    ax9.set_title('Perbandingan Bulan dengan Order Tertinggi vs Terendah\nData Transaksi Tifahampers',
                  fontsize=14, fontweight='bold', pad=15)
    ax9.set_ylabel('Jumlah Order', fontsize=11)
    ax9.set_xlabel('Bulan', fontsize=11)
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)
    ax9.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('grafik_9_order_tertinggi_terendah.png', dpi=300,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print("✅ Grafik 9 tampil + disimpan → grafik_9_order_tertinggi_terendah.png")

except Exception as e:
    print(f"⚠️  Grafik transaksi dilewati: {e}")
    print("   Pastikan file 'data/data_transaksi.xlsx' tersedia")

# ════════════════════════════════════════════════════════════
#  RINGKASAN AKHIR
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  🎉 Selesai! File grafik yang dihasilkan:")
print("     • grafik_1_feature_importance.png")
print("     • grafik_2_confusion_matrix.png")
print("     • grafik_3_waktu_terbaik.png")
print("     • grafik_4_efek_gajian.png")
print("     • grafik_5_tipe_konten.png")
print("     • grafik_6_tren_order.png")
print("     • grafik_7_order_tertinggi_terendah.png")
print("=" * 55)
print("\n💡 Semua grafik tampil di Colab + tersimpan sebagai PNG")
print("   Siap dimasukkan ke BAB 4 skripsi!")