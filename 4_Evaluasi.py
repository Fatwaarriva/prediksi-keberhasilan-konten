# ============================================================
#   MODEL RANDOM FOREST — EVALUASI
#   Input : data_siap_training_v6.csv
#           model_rf_final.pkl
#   Output: grafik_random_forest.png
#           hasil_evaluasi_rf.txt
#
#   Label : AHP 75% Transaksi + 25% Engagement
#   Fitur : time_category, is_weekend, is_payday, Bulan_Num,
#           caption_length, hashtag_count, is_question,
#           is_hard_selling, has_cta, type_encoded
# ============================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score
)

# ────────────────────────────────────────────────────────────
# [0] KONFIGURASI
# ────────────────────────────────────────────────────────────
NAMA_FILE_DATA  = 'data_siap_training_v6.csv'
NAMA_FILE_MODEL = 'model_rf_final.pkl'
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

# ────────────────────────────────────────────────────────────
# [1] LOAD MODEL
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("  MODEL RANDOM FOREST — EVALUASI")
print("  Label AHP: 75% Transaksi + 25% Engagement")
print("=" * 60)

try:
    best_model = joblib.load(NAMA_FILE_MODEL)
    print(f"\n✅ Model berhasil dimuat: '{NAMA_FILE_MODEL}'")
except FileNotFoundError:
    print(f"\n❌ File '{NAMA_FILE_MODEL}' tidak ditemukan!")
    print("   Pastikan sudah menjalankan training_rf.py terlebih dahulu.")
    exit()

# ────────────────────────────────────────────────────────────
# [2] LOAD DATA & RECREATE X_TEST, Y_TEST
# ────────────────────────────────────────────────────────────
print("\n🔧 Memuat data dan mempersiapkan data uji...")

try:
    df = pd.read_csv(NAMA_FILE_DATA)
    print(f"   ✅ Data berhasil dimuat: {len(df)} baris")
except FileNotFoundError:
    print(f"\n❌ File '{NAMA_FILE_DATA}' tidak ditemukan!")
    exit()

FITUR = [
    'time_category',
    'is_weekend',
    'is_payday',
    'Bulan_Num',
    'caption_length',
    'hashtag_count',
    'is_question',
    'is_hard_selling',
    'has_cta',
    'type_encoded'
]

fitur_tersedia = [f for f in FITUR if f in df.columns]

X = df[fitur_tersedia]
y = df['is_success']

# Split ulang dengan random_state & test_size yang SAMA agar X_test identik
_, X_test, _, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"   ✅ Fitur digunakan : {fitur_tersedia}")
print(f"   ✅ Data uji        : {len(X_test)} baris")

# ────────────────────────────────────────────────────────────
# [3] EVALUASI MODEL DI DATA UJI
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  HASIL EVALUASI MODEL DI DATA UJI")
print("=" * 60)

y_pred  = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

acc       = accuracy_score(y_test, y_pred)
auc       = roc_auc_score(y_test, y_proba)
f1        = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
cm        = confusion_matrix(y_test, y_pred)

print(f"\n   🌟 Akurasi          : {acc * 100:.2f}%")
print(f"   🌟 AUC-ROC          : {auc:.4f}")
print(f"   🌟 F1-Score         : {f1:.4f}")
print(f"   🌟 Precision        : {precision:.4f}")
print(f"   🌟 Recall           : {recall:.4f}")

print(f"\n   Confusion Matrix:")
print(f"   {'':20} Prediksi Gagal   Prediksi Sukses")
print(f"   Aktual Gagal   :  {cm[0][0]:^15} {cm[0][1]:^15}")
print(f"   Aktual Sukses  :  {cm[1][0]:^15} {cm[1][1]:^15}")

print(f"\n   Laporan Klasifikasi Detail:")
print(classification_report(y_test, y_pred,
                             target_names=['Gagal (0)', 'Sukses (1)']))

if auc >= 0.9:
    interp_auc = "Sangat Baik (Excellent)"
elif auc >= 0.8:
    interp_auc = "Baik (Good)"
elif auc >= 0.7:
    interp_auc = "Cukup Baik (Fair)"
else:
    interp_auc = "Perlu Perbaikan (Poor)"

print(f"   ℹ️  Interpretasi AUC : {interp_auc}")

# ────────────────────────────────────────────────────────────
# [4] FEATURE IMPORTANCE
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FEATURE IMPORTANCE")
print("=" * 60)

feature_imp = pd.Series(
    best_model.feature_importances_,
    index=fitur_tersedia
).sort_values(ascending=False)

print("\n   Ranking Fitur Berdasarkan Kepentingan:")
for i, (fitur, nilai) in enumerate(feature_imp.items(), 1):
    bar = "█" * int(nilai * 50)
    print(f"   {i:2}. {fitur:<20} {nilai:.4f}  {bar}")

# ────────────────────────────────────────────────────────────
# [5] VISUALISASI (4 Grafik untuk Skripsi)
# ────────────────────────────────────────────────────────────
print("\n📊 Membuat grafik evaluasi...")

sns.set(style="whitegrid")
PALETTE = ['#2EC4B6', '#E84855', '#3A86FF', '#FF9F1C']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle(
    'Evaluasi Model Random Forest\n'
    'Label AHP: 75% Transaksi + 25% Engagement',
    fontsize=14, fontweight='bold', y=0.98
)

# ── Grafik 1: Confusion Matrix ──
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
    xticklabels=['Gagal (0)', 'Sukses (1)'],
    yticklabels=['Gagal (0)', 'Sukses (1)'],
    linewidths=0.5, linecolor='white', annot_kws={"size": 14}
)
axes[0, 0].set_title(f'Confusion Matrix\nAkurasi: {acc*100:.2f}%', fontweight='bold')
axes[0, 0].set_xlabel('Prediksi', fontsize=11)
axes[0, 0].set_ylabel('Aktual', fontsize=11)

# ── Grafik 2: Feature Importance ──
colors = [PALETTE[0] if i < 3 else '#AAAAAA' for i in range(len(feature_imp))]
feature_imp.plot(kind='barh', ax=axes[0, 1], color=colors,
                 edgecolor='white', linewidth=0.8)
axes[0, 1].invert_yaxis()
axes[0, 1].set_title('Feature Importance\n(Top fitur penentu keberhasilan)',
                      fontweight='bold')
axes[0, 1].set_xlabel('Importance Score')
for i, v in enumerate(feature_imp.values):
    axes[0, 1].text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

# ── Grafik 3: ROC Curve ──
fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1, 0].plot(fpr, tpr, color=PALETTE[2], lw=2.5,
                label=f'ROC Curve (AUC = {auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
axes[1, 0].fill_between(fpr, tpr, alpha=0.1, color=PALETTE[2])
axes[1, 0].set_title(f'ROC Curve\nAUC = {auc:.4f} ({interp_auc})', fontweight='bold')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend(loc='lower right', fontsize=10)
axes[1, 0].set_xlim([0, 1])
axes[1, 0].set_ylim([0, 1.02])

# ── Grafik 4: Ringkasan Metrik ──
metrik_nama  = ['Accuracy', 'AUC-ROC', 'F1-Score', 'Precision', 'Recall']
metrik_nilai = [acc, auc, f1, precision, recall]
bars = axes[1, 1].bar(
    metrik_nama, metrik_nilai,
    color=[PALETTE[0], PALETTE[2], PALETTE[3], '#8338EC', '#06D6A0'],
    edgecolor='white', linewidth=1.2, width=0.6
)
for bar, val in zip(bars, metrik_nilai):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f'{val:.4f}',
        ha='center', va='bottom',
        fontweight='bold', fontsize=10
    )
axes[1, 1].set_title('Ringkasan Metrik Evaluasi', fontweight='bold')
axes[1, 1].set_ylabel('Nilai')
axes[1, 1].set_ylim(0, 1.15)
axes[1, 1].axhline(y=0.7, color='red', linestyle='--',
                   linewidth=1.2, alpha=0.5, label='Batas minimum (0.70)')
axes[1, 1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('grafik_random_forest.png', dpi=300,
            bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()
print("✅ Grafik disimpan → 'grafik_random_forest.png'")

# ────────────────────────────────────────────────────────────
# [6] SIMPAN LAPORAN TEKS
# ────────────────────────────────────────────────────────────
with open('hasil_evaluasi_rf.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("  LAPORAN EVALUASI RANDOM FOREST\n")
    f.write("  Label AHP: 75% Transaksi + 25% Engagement\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Hasil Evaluasi Data Uji:\n")
    f.write(f"  Akurasi   : {acc*100:.2f}%\n")
    f.write(f"  AUC-ROC   : {auc:.4f}\n")
    f.write(f"  F1-Score  : {f1:.4f}\n")
    f.write(f"  Precision : {precision:.4f}\n")
    f.write(f"  Recall    : {recall:.4f}\n")
    f.write(f"\nConfusion Matrix:\n{cm}\n")
    f.write(f"\nFeature Importance:\n{feature_imp.to_string()}\n")
    f.write(f"\nLaporan Klasifikasi:\n")
    f.write(classification_report(y_test, y_pred,
                                  target_names=['Gagal (0)', 'Sukses (1)']))

print("✅ Laporan disimpan → 'hasil_evaluasi_rf.txt'")

# ────────────────────────────────────────────────────────────
# [7] RINGKASAN AKHIR
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RINGKASAN AKHIR")
print("=" * 60)
print(f"  Total data uji      : {len(X_test)}")
print(f"  Jumlah fitur        : {len(fitur_tersedia)}")
print(f"  Akurasi             : {acc*100:.2f}%")
print(f"  AUC-ROC             : {auc:.4f} ({interp_auc})")
print(f"  F1-Score            : {f1:.4f}")
print("=" * 60)
print("\n🎉 Evaluasi selesai!")
print("\n💡 OUTPUT YANG DIHASILKAN:")
print("   1. Cek grafik  → 'grafik_random_forest.png'")
print("   2. Cek laporan → 'hasil_evaluasi_rf.txt'")
print("   3. Catat akurasi dan AUC untuk BAB 4 skripsi")