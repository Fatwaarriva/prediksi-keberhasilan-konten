# ============================================================
#   MODEL RANDOM FOREST — TRAINING
#   Input : data_siap_training_v6.csv
#   Output: model_rf_final.pkl
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

# ────────────────────────────────────────────────────────────
# [0] KONFIGURASI
# ────────────────────────────────────────────────────────────
NAMA_FILE_DATA  = 'data_siap_training_v6.csv'
NAMA_FILE_MODEL = 'model_rf_final.pkl'
RANDOM_STATE    = 42
TEST_SIZE       = 0.2

# ────────────────────────────────────────────────────────────
# [1] LOAD DATA
# ────────────────────────────────────────────────────────────
print("=" * 60)
print("   MODEL RANDOM FOREST")
print("=" * 60)

try:
    df = pd.read_csv(NAMA_FILE_DATA)
    print(f"\n✅ Data berhasil dimuat: {len(df)} baris")
    print(f"   Kolom: {list(df.columns)}")
except FileNotFoundError:
    print(f"\n❌ File '{NAMA_FILE_DATA}' tidak ditemukan!")
    print("   Pastikan sudah menjalankan preprocessing_v6.py terlebih dahulu.")
    exit()

# ────────────────────────────────────────────────────────────
# [2] PERSIAPAN FITUR DAN LABEL
# ────────────────────────────────────────────────────────────
print("\n🔧 Mempersiapkan fitur dan label...")

FITUR = [
    'time_category',
    'is_weekend',
    'is_payday',
    'caption_length',
    'hashtag_count',
    'is_question',
    'is_hard_selling',
    'has_cta',
    'type_encoded'
]

fitur_tersedia = [f for f in FITUR if f in df.columns]
fitur_kurang   = [f for f in FITUR if f not in df.columns]

if fitur_kurang:
    print(f"   ⚠️  Fitur tidak ditemukan: {fitur_kurang}")
    print(f"   ℹ️  Menggunakan fitur yang tersedia saja.")

if 'is_success' not in df.columns:
    print("   ❌ Kolom 'is_success' tidak ditemukan!")
    exit()

X = df[fitur_tersedia]
y = df['is_success']

print(f"   ✅ Fitur digunakan  : {fitur_tersedia}")
print(f"   ✅ Total fitur      : {len(fitur_tersedia)}")
print(f"   ✅ Total data       : {len(X)}")
print(f"   ✅ Label Sukses (1) : {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"   ✅ Label Gagal  (0) : {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

imbalance = abs(y.sum() - (y==0).sum()) / len(y) * 100
if imbalance > 30:
    print(f"\n   ⚠️  Imbalance cukup tinggi ({imbalance:.1f}%).")
    print(f"      Model akan menggunakan class_weight='balanced' untuk kompensasi.")

# ────────────────────────────────────────────────────────────
# [3] SPLIT DATA TRAIN & TEST (80:20)
# ────────────────────────────────────────────────────────────
print("\n📂 Membagi data Train & Test (80:20)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"   ✅ Data latih : {len(X_train)} baris")
print(f"   ✅ Data uji   : {len(X_test)} baris")
print(f"   ✅ Stratify   : aktif (distribusi label terjaga)")

# ────────────────────────────────────────────────────────────
# [4] GRID SEARCH + CROSS VALIDATION
# ────────────────────────────────────────────────────────────
print("\n🔍 Grid Search + 5-Fold Stratified Cross Validation...")
print("   (Mohon tunggu 2-3 menit...)\n")

param_grid = {
    'n_estimators'     : [100, 200, 300],
    'max_depth'        : [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight'     : ['balanced', None]
}

rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

grid_search = GridSearchCV(
    estimator  = rf_base,
    param_grid = param_grid,
    cv         = cv,
    n_jobs     = -1,
    verbose    = 1,
    scoring    = 'f1'
)

grid_search.fit(X_train, y_train)

best_model  = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_f1  = grid_search.best_score_

print(f"\n   ✅ Parameter terbaik  : {best_params}")
print(f"   ✅ F1-Score CV (rata) : {best_cv_f1 * 100:.2f}%")

# ────────────────────────────────────────────────────────────
# [5] SIMPAN MODEL
# ────────────────────────────────────────────────────────────
import os

joblib.dump(best_model, NAMA_FILE_MODEL)
size = os.path.getsize(NAMA_FILE_MODEL)

print(f"\n💾 Model disimpan → '{NAMA_FILE_MODEL}'")
print(f"📦 Ukuran model   → {size} bytes")

# ────────────────────────────────────────────────────────────
# [6] RINGKASAN AKHIR
# ────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING SELESAI!")
print("=" * 60)
print(f"  Total data latih    : {len(X_train)}")
print(f"  Total data uji      : {len(X_test)}")
print(f"  Jumlah fitur        : {len(fitur_tersedia)}")
print(f"  F1 CV terbaik       : {best_cv_f1*100:.2f}%")
print(f"  Parameter terbaik   : {best_params}")
print("=" * 60)
print("\n🎉 Training selesai!")
print("\n💡 LANGKAH SELANJUTNYA:")
print("   Jalankan → evaluasi_rf.py")