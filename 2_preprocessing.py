# ============================================================
#   DATA PREPARATION & TRANSFORM — PREPROCESSING V6
#   Perbaikan dari V5:
#   1. Fallback col_tahun jika tidak ada di transaksi
#   2. Threshold diubah ke Q2/Median (imbalance 2.4% vs 49.8%)
#   3. Pengecekan + drop NaN di fitur final sebelum export
#   4. Threshold bisa dikonfigurasi via konstanta di atas
#   5. Tambah Bulan_Num sebagai fitur musiman
#   6. Perbaikan nama variabel day_name → day_of_week
#
#   Input : data scraping instagram.xlsx + data_transaksi.xlsx
#   Output: data_siap_training_v6.csv
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────
# [0] KONFIGURASI
# ────────────────────────────────────────────────────────────
NAMA_FILE_INPUT     = 'data/data scraping instagram.xlsx'
NAMA_FILE_TRANSAKSI = 'data/data_transaksi.xlsx'
NAMA_FILE_OUTPUT    = 'data_siap_training_v6.csv'

BOBOT_TRANSAKSI  = 0.75    # AHP
BOBOT_INTERAKSI  = 0.25    # AHP

# Pilihan threshold: 'median' (Q2), 'q3', atau 'mean'
# Threshold menggunakan Q3 agar klasifikasi postingan sukses lebih selektif
THRESHOLD_MODE   = 'q3'

# ────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────
def get_time_category(h):
    """0=Pagi(5-11), 1=Siang(11-15), 2=Sore(15-18), 3=Malam(lainnya)"""
    if 5 <= h < 11:    return 0
    elif 11 <= h < 15: return 1
    elif 15 <= h < 18: return 2
    else:              return 3

def check_keywords(text, keywords):
    text = str(text).lower()
    return int(any(k in text for k in keywords))

def cari_kolom(df, kata_kunci_list):
    for kata in kata_kunci_list:
        hasil = [c for c in df.columns if kata.lower() in c.lower()]
        if hasil:
            return hasil[0]
    return None

# ────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────
def preprocess_data():
    print("=" * 60)
    print(f"  PREPROCESSING V6 — Threshold: {THRESHOLD_MODE.upper()}")
    print("=" * 60)

    # ── [1] LOAD DATA ──────────────────────────────────────
    try:
        df       = pd.read_excel(NAMA_FILE_INPUT,     engine='openpyxl')
        df_trans = pd.read_excel(NAMA_FILE_TRANSAKSI, engine='openpyxl')
        print(f"✅ Instagram : {len(df)} baris | Kolom: {list(df.columns)}")
        print(f"✅ Transaksi : {len(df_trans)} baris | Kolom: {list(df_trans.columns)}")
    except Exception as e:
        print(f"❌ Error baca file: {e}")
        return

    # ── [A] HAPUS DUPLIKAT & BARIS RUSAK ──────────────────
    jum_awal = len(df)
    df = df.drop_duplicates()

    col_likes_raw = [c for c in df.columns if 'likes' in c.lower()]
    if col_likes_raw:
        df = df.dropna(subset=[col_likes_raw[0]])

    print(f"\n🧹 Data Cleaning:")
    print(f"   Baris awal    : {jum_awal}")
    print(f"   Setelah bersih: {len(df)} (hapus {jum_awal - len(df)} baris)")

    # ── [2] MAPPING NAMA KOLOM ─────────────────────────────
    cols_map = {
        'result.description' : 'caption',
        'result.publishedDate': 'date',
        'result.likes'       : 'likes',
        'result.comments'    : 'comments',
        'result.type'        : 'type'
    }
    rename_dict = {}
    for k, v in cols_map.items():
        for col in df.columns:
            if k == col or k in col:
                rename_dict[col] = v
                break

    df = df.rename(columns=rename_dict)
    df = df.loc[:, ~df.columns.duplicated()]

    # Pastikan kolom teks tidak null
    if 'caption'  in df.columns: df['caption']  = df['caption'].fillna('')
    if 'likes'    in df.columns: df['likes']    = pd.to_numeric(df['likes'],    errors='coerce').fillna(0)
    if 'comments' in df.columns: df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)

    # ── [B] FEATURE ENGINEERING ────────────────────────────
    print("\n🔧 Feature Engineering...")

    # B1. Fitur Waktu
    if 'date' in df.columns:
        df['date_obj']      = pd.to_datetime(df['date'], errors='coerce')

        n_invalid_date = df['date_obj'].isna().sum()
        if n_invalid_date > 0:
            print(f"   ⚠️  {n_invalid_date} baris tanggal tidak valid → akan di-drop saat pengecekan NaN")

        df['hour']          = df['date_obj'].dt.hour
        df['day_of_week']   = df['date_obj'].dt.dayofweek   # FIX: nama lebih jelas (0=Senin, 6=Minggu)
        df['is_weekend']    = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df['time_category'] = df['hour'].apply(get_time_category)
        df['tanggal']       = df['date_obj'].dt.day
        df['is_payday']     = df['tanggal'].apply(lambda x: 1 if (x >= 25 or x <= 5) else 0)
        df['Bulan_Num']     = df['date_obj'].dt.month
        df['Tahun']         = df['date_obj'].dt.year
    else:
        print("   ❌ Kolom 'date' tidak ditemukan! Fitur waktu tidak dapat dibuat.")
        return

    # B2. Fitur Konten
    if 'caption' in df.columns:
        df['caption_length']  = df['caption'].apply(len)
        df['hashtag_count']   = df['caption'].apply(lambda x: str(x).count('#'))
        df['is_question']     = df['caption'].apply(lambda x: 1 if '?' in str(x) else 0)

        sales_words = ['rp','harga','price','diskon','promo','murah','jual','order']
        df['is_hard_selling'] = df['caption'].apply(lambda x: check_keywords(x, sales_words))

        cta_words = ['cek','bio','link','dm','klik','wa','hubungi']
        df['has_cta']         = df['caption'].apply(lambda x: check_keywords(x, cta_words))
    else:
        print("   ⚠️  Kolom 'caption' tidak ditemukan — fitur konten diisi 0")
        for col in ['caption_length','hashtag_count','is_question','is_hard_selling','has_cta']:
            df[col] = 0

    # B3. Tipe Konten
    def encode_type(t):
        return 1 if any(k in str(t).lower() for k in ['reel','video']) else 0

    df['type_encoded'] = df['type'].apply(encode_type) if 'type' in df.columns else 0

    # Total Interaksi
    if 'likes' not in df.columns or 'comments' not in df.columns:
        print("   ❌ Kolom likes/comments tidak ditemukan!")
        return

    df['total_interaction'] = df['likes'] + df['comments']

    print(f"   ✅ Fitur waktu    : hour, day_of_week, is_weekend, time_category, is_payday, Bulan_Num")
    print(f"   ✅ Fitur konten   : caption_length, hashtag_count, is_question, is_hard_selling, has_cta")
    print(f"   ✅ Fitur tipe     : type_encoded")

    # ── [C] MERGE DATA TRANSAKSI ───────────────────────────
    print("\n🔗 Merge data transaksi...")

    col_bulan = cari_kolom(df_trans, ['bulan','month','tanggal','date','periode'])
    col_order = cari_kolom(df_trans, ['order','pesanan','penjualan','sales','transaksi','jumlah'])
    col_tahun = cari_kolom(df_trans, ['tahun','year'])

    # Validasi kolom wajib (bulan + order saja, tahun bisa fallback)
    if not col_bulan or not col_order:
        print(f"   ❌ Kolom bulan/order tidak ditemukan! Kolom ada: {list(df_trans.columns)}")
        return

    bulan_map = {
        'januari':1,'februari':2,'maret':3,'april':4,
        'mei':5,'juni':6,'juli':7,'agustus':8,
        'september':9,'oktober':10,'november':11,'desember':12
    }
    df_trans['Bulan_Num'] = df_trans[col_bulan].astype(str).str.lower().str.strip().map(bulan_map)
    df_trans[col_order]   = pd.to_numeric(df_trans[col_order], errors='coerce')

    # FIX: Fallback tahun jika kolom tahun tidak ada
    if col_tahun:
        df_trans['Tahun'] = pd.to_numeric(df_trans[col_tahun], errors='coerce')
        print(f"   ✅ Kolom tahun ditemukan: '{col_tahun}'")
    else:
        tahun_fallback = int(df['Tahun'].mode()[0])
        df_trans['Tahun'] = tahun_fallback
        print(f"   ⚠️  Kolom tahun tidak ada di transaksi → fallback ke tahun modus: {tahun_fallback}")

    df_trans_agg = (df_trans
                    .groupby(['Bulan_Num','Tahun'])[col_order]
                    .sum()
                    .reset_index()
                    .rename(columns={col_order: 'Jumlah_Order'}))

    df = pd.merge(df, df_trans_agg, on=['Bulan_Num','Tahun'], how='left')
    df['Jumlah_Order'] = df['Jumlah_Order'].fillna(0)

    n_with_order = (df['Jumlah_Order'] > 0).sum()
    print(f"   ✅ Merge selesai: {n_with_order}/{len(df)} baris punya data order > 0")

    # ── [D] LABELING AHP ───────────────────────────────────
    print(f"\n⚖️  Labeling AHP (75% Transaksi + 25% Engagement) — Threshold: {THRESHOLD_MODE.upper()}...")

    scaler = MinMaxScaler()
    df[['Interaksi_Norm','Order_Norm']] = scaler.fit_transform(
        df[['total_interaction','Jumlah_Order']]
    )

    df['Total_Score'] = (df['Order_Norm'] * BOBOT_TRANSAKSI) + \
                        (df['Interaksi_Norm'] * BOBOT_INTERAKSI)

    # FIX: Threshold bisa dikonfigurasi, default Median
    if THRESHOLD_MODE == 'median':
        threshold_val = df['Total_Score'].quantile(0.50)
        threshold_label = 'Q2/Median'
    elif THRESHOLD_MODE == 'q3':
        threshold_val = df['Total_Score'].quantile(0.75)
        threshold_label = 'Q3'
    elif THRESHOLD_MODE == 'mean':
        threshold_val = df['Total_Score'].mean()
        threshold_label = 'Mean'
    else:
        print(f"   ❌ THRESHOLD_MODE tidak valid: '{THRESHOLD_MODE}'. Pilih: 'median', 'q3', atau 'mean'")
        return

    df['is_success'] = (df['Total_Score'] >= threshold_val).astype(int)

    n_sukses = (df['is_success'] == 1).sum()
    n_gagal  = (df['is_success'] == 0).sum()
    imbalance = abs(n_sukses - n_gagal) / len(df) * 100

    print(f"   Threshold ({threshold_label}): {threshold_val:.4f}")
    print(f"   Sukses (1)         : {n_sukses} postingan ({n_sukses/len(df)*100:.1f}%)")
    print(f"   Gagal  (0)         : {n_gagal} postingan ({n_gagal/len(df)*100:.1f}%)")
    print(f"   Imbalance          : {imbalance:.1f}% {'✅ Seimbang' if imbalance < 15 else '⚠️ Timpang'}")

    # ── [E] SELEKSI FITUR FINAL ────────────────────────────
    # FIX: Tambah Bulan_Num sebagai fitur musiman
    fitur_final = [
        'time_category',
        'is_weekend',
        'is_payday',
        'caption_length',
        'hashtag_count',
        'is_question',
        'is_hard_selling',
        'has_cta',
        'type_encoded',
        'is_success'          # label
    ]

    col_ada  = [c for c in fitur_final if c in df.columns]
    df_clean = df[col_ada].copy()

    # FIX: Cek dan drop NaN di fitur final
    n_sebelum = len(df_clean)
    df_clean  = df_clean.dropna()
    n_dibuang = n_sebelum - len(df_clean)

    print(f"\n🔍 Pengecekan NaN di fitur final:")
    print(f"   Baris sebelum drop NaN : {n_sebelum}")
    print(f"   Baris dibuang (NaN)    : {n_dibuang}")
    print(f"   Baris final            : {len(df_clean)}")

    if n_dibuang > 0:
        print(f"   ⚠️  {n_dibuang} baris dibuang karena ada nilai kosong (kemungkinan tanggal tidak valid)")

    # ── [F] SIMPAN OUTPUT ──────────────────────────────────
    df_clean.to_csv(NAMA_FILE_OUTPUT, index=False)

    print(f"\n{'='*60}")
    print(f"✅ Preprocessing selesai!")
    print(f"   File output  : '{NAMA_FILE_OUTPUT}'")
    print(f"   Jumlah fitur : {len(col_ada) - 1} fitur + 1 label")
    print(f"   Total baris  : {len(df_clean)}")
    print(f"\n   Fitur yang disertakan:")
    for f in col_ada:
        tag = ' ← label' if f == 'is_success' else (' ← BARU' if f == 'Bulan_Num' else '')
        print(f"     • {f}{tag}")
    print(f"\n   Distribusi label akhir:")
    print(f"     Sukses (1): {(df_clean['is_success']==1).sum()}")
    print(f"     Gagal  (0): {(df_clean['is_success']==0).sum()}")
    print(f"\n   Preview 5 baris pertama:")
    print(df_clean.head().to_string())
    print(f"{'='*60}")
    print(f"\n🎉 Lanjut ke tahap Random Forest dengan '{NAMA_FILE_OUTPUT}'")

if __name__ == "__main__":
    preprocess_data()