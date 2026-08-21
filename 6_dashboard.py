import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
 
# ══════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SocialInsight AI — Tifahampers",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# CSS KUSTOM
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
 
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0A0F !important;
    color: #E8E8F0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stHeader"] { background: #0A0A0F !important; border-bottom: 1px solid #1E1E2E; }
[data-testid="stSidebar"] {
    background: #0D0D18 !important;
    border-right: 1px solid #1E1E2E !important;
    transform: none !important;
    transition: transform 0.3s ease !important;
}

[data-testid="stSidebar"] * { color: #C8C8D8 !important; }
 
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stHeader"] { visibility: visible !important; background: #0A0A0F !important; border-bottom: 1px solid #1E1E2E; }
.block-container { padding: 2rem 3rem !important; max-width: 1400px; }
 
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
.mono { font-family: 'IBM Plex Mono', monospace !important; }
 
.metric-card {
    background: #111120;
    border: 1px solid #1E1E35;
    border-radius: 16px;
    padding: 1.5rem 1.8rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #6C63FF44; }
.metric-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #6C63FF, #FF6584);
}
.metric-label {
    font-size: 0.68rem; letter-spacing: 0.15em; text-transform: uppercase;
    color: #666680; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.5rem;
}
.metric-value { font-size: 1.8rem; font-weight: 700; letter-spacing: -1px; color: #F0F0FF; line-height: 1; }
.metric-delta { font-size: 0.72rem; color: #2ECC71; margin-top: 0.4rem; font-family: 'IBM Plex Mono', monospace; }
.metric-icon { position: absolute; top: 1.2rem; right: 1.4rem; font-size: 1.1rem; opacity: 0.4; }
 
.section-tag {
    font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase;
    color: #6C63FF; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.6rem; font-weight: 700; color: #F0F0FF;
    letter-spacing: -0.5px; margin-bottom: 1.5rem;
}
.content-card {
    background: #111120;
    border: 1px solid #1E1E35;
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    overflow: visible !important;
}
.result-card-success {
    background: linear-gradient(135deg, #0A2010 0%, #0F1A20 100%);
    border: 1px solid #2ECC7144;
    border-radius: 16px; padding: 2rem;
}
.result-card-fail {
    background: linear-gradient(135deg, #200A10 0%, #1A0F10 100%);
    border: 1px solid #FF668844;
    border-radius: 16px; padding: 2rem;
}
.growth-label { font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: #6C63FF; font-family: 'IBM Plex Mono', monospace; }
.growth-title { font-size: 1.8rem; font-weight: 700; letter-spacing: -1px; margin: 0.4rem 0 1rem 0; }
.growth-desc { font-size: 0.85rem; color: #AAAACC; line-height: 1.6; }
.growth-desc strong { color: #F0F0FF; }
 
.mini-stat {
    background: #0D0D1A; border: 1px solid #1E1E35; border-radius: 10px; padding: 1rem 1.2rem;
    display: inline-block; min-width: 130px; margin: 0.5rem 0.4rem 0 0;
}
.mini-stat-label { font-size: 0.62rem; color: #555570; letter-spacing: 0.12em; text-transform: uppercase; font-family: 'IBM Plex Mono', monospace; }
.mini-stat-value { font-size: 1.1rem; font-weight: 700; color: #F0F0FF; margin-top: 2px; }
 
.drill-number {
    width: 30px; height: 30px; background: #1E1E35; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; color: #888899; flex-shrink: 0;
}
.drill-title { font-size: 0.95rem; font-weight: 600; color: #F0F0FF; text-transform: uppercase; letter-spacing: 0.05em; }
.drill-meta { font-size: 0.72rem; color: #555570; font-family: 'IBM Plex Mono', monospace; margin-top: 2px; }
.pill {
    background: #1E1E35; color: #AAAACC; font-size: 0.65rem;
    font-family: 'IBM Plex Mono', monospace; padding: 3px 10px; border-radius: 20px; letter-spacing: 0.08em;
}
.pill-video { background: #1A1A40; color: #8888FF; }
.pill-photo { background: #1A2530; color: #66AACC; }
 
.perf-bar-wrap { display: flex; align-items: center; gap: 0.8rem; }
.perf-bar { height: 5px; border-radius: 3px; background: #1E1E35; flex: 1; overflow: hidden; }
.perf-bar-fill { height: 100%; border-radius: 3px; background: linear-gradient(90deg, #6C63FF, #2ECC71); }
.perf-score { font-size: 0.85rem; font-weight: 700; color: #F0F0FF; font-family: 'IBM Plex Mono', monospace; min-width: 28px; }
 
.status-badge {
    background: #0F2A1A; border: 1px solid #1A4030;
    color: #2ECC71; font-size: 0.7rem; font-family: 'IBM Plex Mono', monospace;
    padding: 4px 12px; border-radius: 20px; letter-spacing: 0.1em;
    text-transform: uppercase;
}
.sync-info { font-size: 0.72rem; font-family: 'IBM Plex Mono', monospace; color: #555570; text-align: right; margin-top: 4px; }
 
[data-testid="stRadio"] label div p {
    color: #AAAACC !important;
    font-size: 1.1rem !important;
    line-height: 1.6 !important;
}
[data-testid="stTextArea"] textarea::placeholder {
    color: #555570 !important;
}
 
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] select,
[data-testid="stDateInput"] input,
[data-testid="stTimeInput"] input {
    background: #0D0D1A !important; border: 1px solid #2A2A40 !important;
    color: #E8E8F0 !important; border-radius: 8px !important; font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus { border-color: #6C63FF !important; box-shadow: 0 0 0 2px #6C63FF22 !important; }
 
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6C63FF, #5A54E0) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
    padding: 0.7rem 2rem !important; transition: opacity 0.2s !important;
}
[data-testid="stButton"] button:hover { opacity: 0.85 !important; }
 
hr { border-color: #1E1E35 !important; margin: 1.5rem 0 !important; }
[data-testid="stDataFrame"] { border: 1px solid #1E1E35 !important; border-radius: 12px !important; overflow: hidden; }
[data-testid="stExpander"] { background: #111120 !important; border: 1px solid #1E1E35 !important; border-radius: 12px !important; }
 
/* Transaction Insight Cards */
.txn-card-success {
    background: linear-gradient(135deg, #071A0D 0%, #0B1510 100%);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s;
}
.txn-card-success:hover { transform: translateY(-2px); }
.txn-card-fail {
    background: linear-gradient(135deg, #1A0708 0%, #140B0B 100%);
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s;
}
.txn-card-fail:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<script>
    function openSidebar() {
        const btn = window.parent.document.querySelector('[data-testid="stSidebarCollapsedControl"] button');
        if (btn) {
            btn.click();
        } else {
            setTimeout(openSidebar, 500);
        }
    }
    setTimeout(openSidebar, 800);
</script>
""", unsafe_allow_html=True) 

# ── matplotlib global dark theme ──────────────────────────
plt.rcParams.update({
    'figure.facecolor':  '#111120',
    'axes.facecolor':    '#111120',
    'axes.edgecolor':    '#2A2A40',
    'axes.labelcolor':   '#AAAACC',
    'xtick.color':       '#666680',
    'ytick.color':       '#666680',
    'text.color':        '#E8E8F0',
    'grid.color':        '#1E1E35',
    'grid.linestyle':    '--',
    'grid.alpha':        0.5,
})
 
# ══════════════════════════════════════════════════════════
# KONSTANTA
# ══════════════════════════════════════════════════════════
BOBOT_TRANSAKSI = 0.75
BOBOT_INTERAKSI = 0.25
PALETTE = ['#6C63FF', '#FF6584', '#FFB347', '#2ECC71', '#36D7B7']
NAMA_BULAN = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
              7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}
 
# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
def cari_kolom(df, kata_kunci_list):
    for kata in kata_kunci_list:
        hasil = [c for c in df.columns if kata.lower() in c.lower()]
        if hasil:
            return hasil[0]
    return None
 
def get_time_category(h):
    if 5 <= h < 11:    return 0
    elif 11 <= h < 15: return 1
    elif 15 <= h < 18: return 2
    else:              return 3
 
def check_keywords(text, keywords):
    text = str(text).lower()
    return int(any(k in text for k in keywords))
 
def preprocess_input(tgl, jam, caption, tipe):
    dt_obj = datetime.combine(tgl, jam)
    text   = str(caption)
    sales_words = ['rp','harga','price','diskon','promo','murah','jual','order']
    cta_words   = ['cek','bio','link','dm','klik','wa','hubungi']
    return pd.DataFrame({
        'Bulan_Num':       [dt_obj.month],
        'time_category':   [get_time_category(dt_obj.hour)],
        'is_weekend':      [1 if dt_obj.weekday() >= 5 else 0],
        'is_payday':       [1 if (dt_obj.day >= 25 or dt_obj.day <= 5) else 0],
        'caption_length':  [len(text)],
        'hashtag_count':   [text.count('#')],
        'is_question':     [1 if '?' in text else 0],
        'is_hard_selling': [check_keywords(text, sales_words)],
        'has_cta':         [check_keywords(text, cta_words)],
        'type_encoded':    [1 if tipe == 'Video / Reels' else 0]
    })
 
# ══════════════════════════════════════════════════════════
# LOAD DATA & MODEL
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        return joblib.load('model_rf_final.pkl')
    except:
        return None
 
@st.cache_data
def load_and_prepare():
    try:
        df_ig    = pd.read_excel('data/FINAL.xlsx',          engine='openpyxl')
        df_trans = pd.read_excel('data/data_transaksi.xlsx', engine='openpyxl')
    except Exception as e:
        return None, str(e)
 
    col_tgl      = cari_kolom(df_ig,    ['date','tanggal','tgl','time','waktu'])
    col_likes    = cari_kolom(df_ig,    ['like','suka'])
    col_comments = cari_kolom(df_ig,    ['comment','komen','komentar'])
    col_type     = cari_kolom(df_ig,    ['type','tipe','jenis','kind','format'])
    col_caption  = cari_kolom(df_ig,    ['caption','description','desc','teks','konten'])
    col_bulan    = cari_kolom(df_trans, ['bulan','month','tanggal','date','periode'])
    col_order    = cari_kolom(df_trans, ['order','pesanan','penjualan','sales','jumlah'])
 
    if not all([col_tgl, col_likes, col_comments, col_type, col_bulan, col_order]):
        return None, "Kolom wajib tidak ditemukan."
 
    df = df_ig.copy()
    df['dt']        = pd.to_datetime(df[col_tgl], errors='coerce')
    df              = df.dropna(subset=['dt']).drop_duplicates()
    df['Bulan_Num'] = df['dt'].dt.month
    df['Tahun']     = df['dt'].dt.year
    df['Hari']      = df['dt'].dt.day_name()
    df['hour']      = df['dt'].dt.hour
    df['time_cat']  = df['hour'].apply(get_time_category)
    df['likes']     = pd.to_numeric(df[col_likes],    errors='coerce').fillna(0)
    df['comments']  = pd.to_numeric(df[col_comments], errors='coerce').fillna(0)
    df['total_interaction'] = df['likes'] + df['comments']
    df['type']      = df[col_type].astype(str)
    df['caption']   = df[col_caption].fillna('') if col_caption else ''
 
    col_tahun_trans = cari_kolom(df_trans, ['tahun', 'year'])
    bulan_map = {
        'januari':1,'februari':2,'maret':3,'april':4,
        'mei':5,'juni':6,'juli':7,'agustus':8,
        'september':9,'oktober':10,'november':11,'desember':12
    }
    dt2              = df_trans.copy()
    dt2['Bulan_Num'] = dt2[col_bulan].str.lower().str.strip().map(bulan_map)
    dt2['Tahun']     = pd.to_numeric(dt2[col_tahun_trans], errors='coerce')
    dt2[col_order]   = pd.to_numeric(dt2[col_order], errors='coerce')
    agg = (dt2.groupby(['Bulan_Num','Tahun'])[col_order]
              .sum().reset_index()
              .rename(columns={col_order: 'Jumlah_Order'}))
 
    df = pd.merge(df, agg, on=['Bulan_Num','Tahun'], how='left')
    df['Jumlah_Order'] = df['Jumlah_Order'].fillna(0)
 
    scaler = MinMaxScaler()
    df[['Interaksi_Norm','Order_Norm']] = scaler.fit_transform(
        df[['total_interaction','Jumlah_Order']]
    )
    df['Total_Score'] = (df['Order_Norm'] * BOBOT_TRANSAKSI) + \
                        (df['Interaksi_Norm'] * BOBOT_INTERAKSI)
    q3               = df['Total_Score'].quantile(0.75)
    df['is_success'] = (df['Total_Score'] >= q3).astype(int)
    df['Label']      = df['is_success'].map({1:'✅ Sukses', 0:'❌ Gagal'})
    df['Bulan_Nama'] = df['Bulan_Num'].map(NAMA_BULAN)
 
    return df, None
 
model   = load_model()
df, err = load_and_prepare()
 
# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    try:
        from PIL import Image
        import base64
        import io
        img = Image.open("logo.jpg")
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"""
        <div style='text-align:center;padding:0.5rem 0 0.3rem 0;'>
            <img src='data:image/jpeg;base64,{img_b64}' width='95' 
            style='border-radius:50%;border:2px solid #1E1E35;'>
        </div>
        """, unsafe_allow_html=True)
    except:
        pass

    st.markdown("""
    <div style='padding: 0.3rem 0 0.5rem 0; text-align:center;'>
        <div style='font-size:0.6rem;letter-spacing:0.2em;color:#555570;
                    font-family:"IBM Plex Mono",monospace;text-transform:uppercase;
                    margin-bottom:4px'>● NETWORK STATUS: OPERATIONAL</div>
        <div style='font-size:1.5rem;font-weight:700;letter-spacing:-0.5px;color:#F0F0FF'>
            SOCIAL<span style='color:#6C63FF;font-style:italic'>INSIGHT</span>.AI
        </div>
        <div style='font-size:0.80rem;color:#555570;letter-spacing:0.1em;
                    text-transform:uppercase;margin-top:2px'>
            Tifahampers & Florist
        </div>
    </div>
    <hr style='border-color:#1E1E35;margin:1rem 0;'>
    """, unsafe_allow_html=True)
 
    menu = st.radio("", [
        "🏠  Executive Summary",
        "🔮  Predictive Planning",
        "🔍  Diagnostic Analysis",
        "📋  Historical Drill-Down"
    ], label_visibility="collapsed")
    
    st.markdown("""
    <hr style='border-color:#1E1E35;margin:1rem 0;'>
    <div style='font-size:0.65rem;color:#444460;font-family:"IBM Plex Mono",monospace;letter-spacing:0.08em;text-transform:uppercase;'>
        CONTACT INFO<br>
        <span style='color:#8888AA'>IG: @tifahampersflorist</span><br>
        <span style='color:#8888AA'>WA: 0821-1299-7010</span><br>
        <span style='color:#8888AA'>Lokasi: Ruko Cendana Hive</span><br><br>
        <span style='color:#6C63FF'>● ACTIVE SINCE 2018</span>
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════════════════
def render_header(tag, title):
    now = datetime.now()
    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;justify-content:space-between;margin-bottom:2rem;'>
        <div>
            <div class='section-tag'>{tag}</div>
            <div class='section-title'>{title}</div>
        </div>
        <div style='text-align:right;'>
            <div class='status-badge'>● OPERATIONAL</div>
            <div class='sync-info'>LAST SYNC {now.strftime("%d %b %Y")} // {now.strftime("%H:%M")}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
def metric_card(label, value, delta="", icon=""):
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-icon'>{icon}</div>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        {'<div class="metric-delta">▲ ' + delta + '</div>' if delta else ''}
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════
# [A] EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════
if "Executive Summary" in menu:
    render_header("SYSTEM ANALYTICS", "EXECUTIVE SUMMARY")

    if df is None:
        st.error(f"❌ Gagal memuat data: {err}")
        st.stop()

    cols = st.columns(6)
    with cols[0]: metric_card("TOTAL POST",        f"{len(df):,}",                         "+4.2% CR", "📄")
    with cols[1]: metric_card("AGGREGATED LIKES",  f"{int(df['likes'].sum()):,}",           "+12.4% CR","👍")
    with cols[2]: metric_card("INTERACTION VOL",   f"{int(df['comments'].sum()):,}",        "+12.4% CR","💬")
    with cols[3]: metric_card("AVG INTERACTION",   f"{df['total_interaction'].mean():.0f}", "",         "📊")
    with cols[4]: metric_card("SUCCESS RATE",      f"{df['is_success'].mean()*100:.1f}%",   "",         "✅")
    with cols[5]: metric_card("AVG ORDER/MO",      f"{df['Jumlah_Order'].mean():.0f}",      "",         "🛒")

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    import plotly.graph_objects as go

    col1, col2 = st.columns(2)

    # ── Grafik 1: Tren Interaksi ──
    with col1:
        df_bln = df.groupby('Bulan_Num')['total_interaction'].mean().reset_index()
        df_bln['Bulan'] = df_bln['Bulan_Num'].map(NAMA_BULAN)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_bln['Bulan'],
            y=df_bln['total_interaction'],
            mode='lines+markers',
            line=dict(color='#6C63FF', width=2.5),
            marker=dict(size=7, color='#6C63FF',
                        line=dict(color='#0A0A0F', width=2)),
            fill='tozeroy',
            fillcolor='rgba(108,99,255,0.08)',
            hovertemplate='<b>%{x}</b><br>Avg Interaksi: <b>%{y:.0f}</b><br><i>Rata-rata interaksi postingan di bulan ini</i><extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=220,
            xaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Avg Interaksi', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#6C63FF', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>ENGAGEMENT ANALYTICS</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Tren Interaksi per Bulan</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 2: Tren Order ──
    with col2:
        df_ord = df.groupby('Bulan_Num')['Jumlah_Order'].mean().reset_index()
        df_ord['Bulan'] = df_ord['Bulan_Num'].map(NAMA_BULAN)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_ord['Bulan'],
            y=df_ord['Jumlah_Order'],
            marker_color='#2ECC71',
            marker_line_width=0,
            hovertemplate='<b>%{x}</b><br>Avg Order: <b>%{y:.0f}</b><br><i>Rata-rata order masuk di bulan ini</i><extra></extra>'
        ))
        fig2.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=220,
            xaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Jumlah Order', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#2ECC71', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>TRANSACTION METRICS</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Tren Jumlah Order per Bulan</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    # ── Grafik 3: Proporsi Tipe Konten ──
    with col3:
        tipe_cnt = df['type'].value_counts()

        fig3 = go.Figure()
        fig3.add_trace(go.Pie(
            labels=tipe_cnt.index,
            values=tipe_cnt.values,
            hole=0.6,
            marker=dict(colors=PALETTE[:len(tipe_cnt)],
                        line=dict(color='#0A0A0F', width=2)),
            hovertemplate='<b>%{label}</b><br>Jumlah: <b>%{value}</b><br>Proporsi: <b>%{percent}</b><br><i>Tipe konten yang diposting</i><extra></extra>',
            textfont=dict(color='#F0F0FF', size=10),
        ))
        fig3.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=250,
            legend=dict(font=dict(color='#AAAACC', size=9),
                        bgcolor='#111120', bordercolor='#1E1E35'),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#6C63FF', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>CONTENT SCHEMA</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Proporsi Tipe Konten</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 4: Sukses vs Gagal ──
    with col4:
        lbl = df['is_success'].value_counts().sort_index()

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            x=['Gagal', 'Sukses'],
            y=[lbl.get(0, 0), lbl.get(1, 0)],
            marker_color=[PALETTE[1], PALETTE[0]],
            marker_line_width=0,
            width=0.45,
            text=[lbl.get(0, 0), lbl.get(1, 0)],
            textposition='outside',
            textfont=dict(color='#F0F0FF', size=11),
            hovertemplate='<b>%{x}</b><br>Jumlah: <b>%{y}</b><br><i>Berdasarkan AHP Score threshold</i><extra></extra>'
        ))
        fig4.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=30, b=10), height=250,
            xaxis=dict(tickfont=dict(color='#AAAACC', size=10),
                       showline=False, gridcolor='#1E1E35'),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Jumlah Postingan', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#FF6584', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>PERFORMANCE INDEX</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Sukses vs Gagal (AHP Score)</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
 
# ══════════════════════════════════════════════════════════
# [B] PREDICTIVE PLANNING
# ══════════════════════════════════════════════════════════
elif "Predictive Planning" in menu:
    
    # ── Header lebih besar ──
    st.markdown("""
    <div style='margin-bottom:1.5rem;'>
        <div style='font-size:0.75rem;letter-spacing:0.2em;color:#6C63FF;
                    font-family:"IBM Plex Mono",monospace;text-transform:uppercase;
                    margin-bottom:0.3rem;'>PREDICTIVE PLANNING MODULE</div>
        <div style='font-size:2rem;font-weight:700;color:#F0F0FF;letter-spacing:-0.5px;'>
            CONTENT PREDICTOR
        </div>
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.markdown("""
        <div style='background:#1A0A0A;border:1px solid #FF668844;border-radius:12px;padding:1.5rem;'>
            <div style='color:#FF6584;font-weight:700;font-family:"IBM Plex Mono",monospace;font-size:0.85rem;'>⚠ MODEL NOT FOUND</div>
            <div style='color:#AAAACC;font-size:0.85rem;margin-top:0.5rem;'>Jalankan <code>3_training_rf.py</code> terlebih dahulu untuk generate model_rf_final.pkl</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
 
    # ── Siapkan data order per bulan untuk Transaction Insight ──
    if df is not None:
        df_order_bulan = (
            df.groupby(['Bulan_Num', 'Bulan_Nama'])['Jumlah_Order']
            .mean()
            .reset_index()
            .sort_values('Jumlah_Order', ascending=False)
        )
        avg_order_global = df_order_bulan['Jumlah_Order'].mean()
 
        avg_order_sukses = df[df['is_success'] == 1]['Jumlah_Order'].mean()
        avg_order_gagal  = df[df['is_success'] == 0]['Jumlah_Order'].mean()
    else:
        df_order_bulan   = None
        avg_order_global = 0
        avg_order_sukses = 0
        avg_order_gagal  = 0
 
    # ── Form Input ──────────────────────────────────────────
    # ── Form Input dalam kotak ──
    st.markdown(
        "<div style='background:#111120;border:1px solid #1E1E35;border-radius:16px;"
        "padding:0.7rem 1rem;margin-bottom:1rem;'>"
        "<div style='font-size:1rem;font-weight:700;letter-spacing:-0.5px;"
        "color:#F0F0FF;margin-bottom:0.6rem;'>"
        "INPUT PARAMETERS</div>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<p style='color:#FFFFFF;font-size:0.85rem;margin-bottom:0.2rem;'>Rencana Tanggal</p>", unsafe_allow_html=True)
        tgl_input = st.date_input("", datetime.now(), label_visibility="collapsed")

        st.markdown("<p style='color:#FFFFFF;font-size:0.85rem;margin-bottom:0.2rem;'>Rencana Jam (Temporal Index)</p>", unsafe_allow_html=True)
        jam_input = st.time_input("", datetime.now(), label_visibility="collapsed")

        st.markdown("<p style='color:#AAAACC;font-size:0.85rem;margin-bottom:0.2rem;'>Vector Type</p>", unsafe_allow_html=True)
        tipe_input = st.radio("", ('Foto / Carousel', 'Video / Reels'), horizontal=True, label_visibility="collapsed")

        st.markdown("<p style='color:#FFFFFF;font-size:0.85rem;margin-bottom:0.2rem;'>Cluster Tag</p>", unsafe_allow_html=True)
        kategori = st.selectbox("", ['Hampers','Buket Bunga','Snack Box','Hampers Pernikahan','Gift'], label_visibility="collapsed")

    with col2:
        st.markdown("<p style='color:#FFFFFF;font-size:0.85rem;margin-bottom:0.2rem;'>Caption Input (termasuk hashtag)</p>", unsafe_allow_html=True)
        caption_input = st.text_area("", height=190,
                                     placeholder="Contoh: Promo Hampers Lebaran! Diskon 20%... #hampers #murah",
                                     label_visibility="collapsed")

    predict_btn = st.button("⚡  RUN PREDICTION ENGINE", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
 
    # ── Hasil Prediksi ───────────────────────────────────────
    if predict_btn:
        X_baru  = preprocess_input(tgl_input, jam_input, caption_input, tipe_input)
        X_baru = X_baru[model.feature_names_in_]
    
        pred    = model.predict(X_baru)[0]
        prob    = model.predict_proba(X_baru)[0]
        peluang = prob[1] * 100
 
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
 
        # ── Baris 1: Success Matrix + Result Card ─────────────
        left, right = st.columns([1.1, 1])
 
        with left:
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            st.markdown("""
            <div class='section-tag'>SYSTEM ANALYTICS</div>
            <div style='font-size:1.1rem;font-weight:700;letter-spacing:0.05em;color:#F0F0FF;margin-bottom:1.2rem'>SUCCESS MATRIX</div>
            """, unsafe_allow_html=True)
 
            fitur_cols = list(model.feature_names_in_)   # ← otomatis ambil dari model
            label_map  = {
                'Bulan_Num':       'Bulan',               # ← tambahkan ini
                'time_category':   'Jam Posting',
                'is_weekend':      'Weekend',
                'is_payday':       'Tanggal Gajian',
                'caption_length':  'Panjang Caption',
                'hashtag_count':   'Jumlah Hashtag',
                'is_question':     'Ada Pertanyaan',
                'is_hard_selling': 'Kata Jualan',
                'has_cta':         'Ada CTA',
                'type_encoded':    'Tipe Konten'
            }
            top5 = ['time_category','hashtag_count','type_encoded','caption_length','is_hard_selling']
            bar_colors_sm = ['#6C63FF','#FF6584','#FFB347','#2ECC71','#36D7B7']
 
            fig5, ax5 = plt.subplots(figsize=(6, 3.2))
            vals  = [X_baru[c].iloc[0] for c in top5]
            norms = [min(v/max(max(vals),1), 1.0) for v in vals]
            lbls  = [label_map.get(c, c) for c in top5]
            for i, (lbl, n) in enumerate(zip(lbls, norms)):
                ax5.barh(i, n, color=bar_colors_sm[i % len(bar_colors_sm)], height=0.55, edgecolor='none')
            ax5.set_yticks(range(len(lbls)))
            ax5.set_yticklabels(lbls, fontsize=9)
            ax5.set_xlim(0, 1.05)
            ax5.xaxis.set_visible(False)
            sns.despine(left=False, bottom=True)
            fig5.patch.set_alpha(0)
            st.pyplot(fig5); plt.close()
            st.markdown("</div>", unsafe_allow_html=True)
 
        with right:
            if pred == 1:
                card_class   = "result-card-success"
                icon_html    = '<div style="background:#1A3A20;width:50px;height:50px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;margin-bottom:1rem;">📈</div>'
                growth_label = "MARKET PREDICTION"
                growth_title = "GROWTH POTENTIAL"
                growth_color = "#2ECC71"
                growth_body  = f'Advanced heuristic analysis indicates that <strong>VIDEO ASSETS</strong> deployed during <strong>18:00 – 20:00</strong> window generate exponential engagement loops.'
                opt_label    = "OPTIMAL VECTOR"
                opt_val      = "MOTION_HD"
                imp_label    = "EST IMPACT"
                imp_val      = f"+{peluang:.1f}%"
                btn_txt      = "GENERATE STRATEGIC REPORT"
            else:
                card_class   = "result-card-fail"
                icon_html    = '<div style="background:#3A1A20;width:50px;height:50px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:1.3rem;margin-bottom:1rem;">📉</div>'
                growth_label = "RISK ALERT"
                growth_title = "LOW CONVERSION"
                growth_color = "#FF6584"
                growth_body  = f'Content pattern does not match high-performing clusters. Estimated conversion gap: <strong>{100-peluang:.1f}%</strong> below target threshold.'
                opt_label    = "RECOMMENDED FIX"
                opt_val      = "REVISE_CONTENT"
                imp_label    = "RISK INDEX"
                imp_val      = f"–{100-peluang:.1f}%"
                btn_txt      = "GENERATE RECOVERY PLAN"
 
            st.markdown(f"""
            <div class='{card_class}'>
                {icon_html}
                <div class='growth-label'>{growth_label}</div>
                <div class='growth-title' style='color:{growth_color}'>{growth_title}</div>
                <div class='growth-desc'>{growth_body}</div>
                <div style='display:flex;gap:1rem;margin-top:1.5rem;'>
                    <div class='mini-stat'>
                        <div class='mini-stat-label'>{opt_label}</div>
                        <div class='mini-stat-value'>{opt_val}</div>
                    </div>
                    <div class='mini-stat'>
                        <div class='mini-stat-label'>{imp_label}</div>
                        <div class='mini-stat-value' style='color:{growth_color}'>{imp_val}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
 
        # ══════════════════════════════════════════════════
        # TRANSACTION INSIGHT — BAGIAN BARU
        # ══════════════════════════════════════════════════
        if df is not None and df_order_bulan is not None and len(df_order_bulan) > 0:
 
            if pred == 1:
                # ── SUKSES: tampilkan 3 bulan order TERTINGGI ──
                top3 = df_order_bulan.nlargest(3, 'Jumlah_Order').reset_index(drop=True)
                selisih_top = ((top3['Jumlah_Order'].mean() - avg_order_global) / avg_order_global * 100) if avg_order_global > 0 else 0
                peak_val    = int(top3['Jumlah_Order'].max())
                colors_card = ['#2ECC71', '#36D7B7', '#6C63FF']
                border_hex  = ['#2ECC7144', '#36D7B744', '#6C63FF44']
 
                st.markdown("""
                <div class='content-card'>
                    <div class='section-tag'>TRANSACTION INSIGHT — HISTORICAL SUCCESS</div>
                    <div style='font-size:1.05rem;font-weight:700;color:#F0F0FF;margin-bottom:0.3rem;'>
                        📈 Data Nyata: Bulan-Bulan dengan Order Tertinggi
                    </div>
                    <div style='font-size:0.77rem;color:#555570;font-family:"IBM Plex Mono",monospace;margin-bottom:1.4rem;letter-spacing:0.06em;'>
                        Konten yang diprediksi SUKSES berkorelasi dengan periode transaksi berikut
                    </div>
                """, unsafe_allow_html=True)
 
                cols_txn = st.columns(3)
                rank_labels = ['🥇 PUNCAK TERTINGGI', '🥈 PUNCAK KEDUA', '🥉 PUNCAK KETIGA']
                for idx, col in enumerate(cols_txn):
                    if idx < len(top3):
                        row_b     = top3.iloc[idx]
                        bulan_nm  = row_b['Bulan_Nama']
                        order_val = int(row_b['Jumlah_Order'])
                        pct_vs_avg = ((order_val - avg_order_global) / avg_order_global * 100) if avg_order_global > 0 else 0
                        sign      = "+" if pct_vs_avg >= 0 else ""
 
                        with col:
                            st.markdown(f"""
                            <div style='background:linear-gradient(135deg,#071A0D,#0B1510);
                                        border:1px solid {border_hex[idx]};
                                        border-radius:14px;padding:1.4rem 1.2rem;text-align:center;
                                        transition:transform 0.2s;'>
                                <div style='font-size:0.58rem;letter-spacing:0.2em;text-transform:uppercase;
                                            color:{colors_card[idx]};font-family:"IBM Plex Mono",monospace;
                                            margin-bottom:0.6rem;'>
                                    {rank_labels[idx]}
                                </div>
                                <div style='font-size:2rem;font-weight:700;color:#F0F0FF;letter-spacing:-1px;line-height:1;'>
                                    {bulan_nm}
                                </div>
                                <div style='font-size:2.2rem;font-weight:700;color:{colors_card[idx]};
                                            font-family:"IBM Plex Mono",monospace;margin-top:0.5rem;
                                            letter-spacing:-1px;line-height:1;'>
                                    {order_val:,}
                                </div>
                                <div style='font-size:0.65rem;color:#555570;font-family:"IBM Plex Mono",monospace;
                                            margin-top:0.25rem;letter-spacing:0.08em;'>
                                    ORDER / BULAN
                                </div>
                                <div style='margin-top:0.8rem;background:{colors_card[idx]}22;border:1px solid {colors_card[idx]}44;
                                            border-radius:8px;padding:0.4rem 0.6rem;
                                            font-size:0.7rem;color:{colors_card[idx]};
                                            font-family:"IBM Plex Mono",monospace;letter-spacing:0.06em;'>
                                    {sign}{pct_vs_avg:.1f}% vs rata-rata
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
 
                # Bar mini chart semua bulan
                st.markdown("<div style='margin-top:1.4rem;'>", unsafe_allow_html=True)
                df_chart = df_order_bulan.sort_values('Bulan_Num')
                fig_txn, ax_txn = plt.subplots(figsize=(10, 2.8))
                bar_clrs = ['#2ECC71' if row['Bulan_Num'] in top3['Bulan_Num'].values
                            else '#1E2830'
                            for _, row in df_chart.iterrows()]
                bars_txn = ax_txn.bar(range(len(df_chart)), df_chart['Jumlah_Order'],
                                      color=bar_clrs, edgecolor='none', width=0.65)
                for i, b in enumerate(bars_txn):
                    if df_chart.iloc[i]['Bulan_Num'] in top3['Bulan_Num'].values:
                        ax_txn.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                                    f"{int(b.get_height()):,}", ha='center', fontsize=8,
                                    fontweight='bold', color='#2ECC71')
                ax_txn.axhline(avg_order_global, color='#6C63FF', linewidth=1.5,
                               linestyle='--', alpha=0.7, label=f'Rata-rata: {int(avg_order_global):,}')
                ax_txn.set_xticks(range(len(df_chart)))
                ax_txn.set_xticklabels(df_chart['Bulan_Nama'], fontsize=8)
                ax_txn.set_ylabel('Jumlah Order', fontsize=8)
                ax_txn.legend(fontsize=8, framealpha=0)
                ax_txn.grid(True, axis='y'); sns.despine(); fig_txn.patch.set_alpha(0)
                st.pyplot(fig_txn); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
 
                # Footer insight
                st.markdown(f"""
                <div style='margin-top:0.8rem;background:#0D0D1A;border:1px solid #2ECC7133;
                            border-radius:10px;padding:0.9rem 1.2rem;
                            display:flex;align-items:center;gap:0.8rem;'>
                    <div style='font-size:1.2rem;'>📊</div>
                    <div style='font-size:0.78rem;color:#AAAACC;font-family:"IBM Plex Mono",monospace;line-height:1.6;'>
                        Bulan puncak (<strong style='color:#2ECC71'>{top3.iloc[0]['Bulan_Nama']}</strong>) mencatat 
                        <strong style='color:#F0F0FF'>{peak_val:,} order</strong> — 
                        rata-rata <strong style='color:#2ECC71'>+{selisih_top:.1f}%</strong> 
                        di atas rata-rata bulanan <strong style='color:#F0F0FF'>({int(avg_order_global):,} order/bln)</strong>.
                        Posting di periode serupa berpotensi mendorong konversi signifikan.
                    </div>
                </div>
                </div>
                """, unsafe_allow_html=True)
 
            else:
                # ── GAGAL: tampilkan 3 bulan order TERENDAH ──
                bot3        = df_order_bulan.nsmallest(3, 'Jumlah_Order').reset_index(drop=True)
                selisih_bot = ((avg_order_global - bot3['Jumlah_Order'].mean()) / avg_order_global * 100) if avg_order_global > 0 else 0
                lowest_val  = int(bot3['Jumlah_Order'].min())
                colors_card = ['#FF6584', '#FFB347', '#FF8C69']
                border_hex  = ['#FF658444', '#FFB34744', '#FF8C6944']
 
                st.markdown("""
                <div class='content-card'>
                    <div class='section-tag'>TRANSACTION INSIGHT — RISK REFERENCE</div>
                    <div style='font-size:1.05rem;font-weight:700;color:#F0F0FF;margin-bottom:0.3rem;'>
                        📉 Data Nyata: Bulan-Bulan dengan Omset Paling Sedikit
                    </div>
                    <div style='font-size:0.77rem;color:#555570;font-family:"IBM Plex Mono",monospace;margin-bottom:1.4rem;letter-spacing:0.06em;'>
                        Konten yang diprediksi GAGAL cenderung menghasilkan konversi rendah seperti periode berikut
                    </div>
                """, unsafe_allow_html=True)
 
                cols_txn = st.columns(3)
                risk_labels = ['⚠️ OMSET TERKECIL', '⚠️ OMSET RENDAH 2', '⚠️ OMSET RENDAH 3']
                for idx, col in enumerate(cols_txn):
                    if idx < len(bot3):
                        row_b     = bot3.iloc[idx]
                        bulan_nm  = row_b['Bulan_Nama']
                        order_val = int(row_b['Jumlah_Order'])
                        pct_vs_avg = ((order_val - avg_order_global) / avg_order_global * 100) if avg_order_global > 0 else 0
 
                        with col:
                            st.markdown(f"""
                            <div style='background:linear-gradient(135deg,#1A0708,#140B0B);
                                        border:1px solid {border_hex[idx]};
                                        border-radius:14px;padding:1.4rem 1.2rem;text-align:center;'>
                                <div style='font-size:0.58rem;letter-spacing:0.2em;text-transform:uppercase;
                                            color:{colors_card[idx]};font-family:"IBM Plex Mono",monospace;
                                            margin-bottom:0.6rem;'>
                                    {risk_labels[idx]}
                                </div>
                                <div style='font-size:2rem;font-weight:700;color:#F0F0FF;letter-spacing:-1px;line-height:1;'>
                                    {bulan_nm}
                                </div>
                                <div style='font-size:2.2rem;font-weight:700;color:{colors_card[idx]};
                                            font-family:"IBM Plex Mono",monospace;margin-top:0.5rem;
                                            letter-spacing:-1px;line-height:1;'>
                                    {order_val:,}
                                </div>
                                <div style='font-size:0.65rem;color:#555570;font-family:"IBM Plex Mono",monospace;
                                            margin-top:0.25rem;letter-spacing:0.08em;'>
                                    ORDER / BULAN
                                </div>
                                <div style='margin-top:0.8rem;background:{colors_card[idx]}22;border:1px solid {colors_card[idx]}44;
                                            border-radius:8px;padding:0.4rem 0.6rem;
                                            font-size:0.7rem;color:{colors_card[idx]};
                                            font-family:"IBM Plex Mono",monospace;letter-spacing:0.06em;'>
                                    {pct_vs_avg:.1f}% vs rata-rata
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
 
                # Bar mini chart semua bulan
                st.markdown("<div style='margin-top:1.4rem;'>", unsafe_allow_html=True)
                df_chart = df_order_bulan.sort_values('Bulan_Num')
                fig_txn, ax_txn = plt.subplots(figsize=(10, 2.8))
                bar_clrs = ['#FF6584' if row['Bulan_Num'] in bot3['Bulan_Num'].values
                            else '#1E2830'
                            for _, row in df_chart.iterrows()]
                bars_txn = ax_txn.bar(range(len(df_chart)), df_chart['Jumlah_Order'],
                                      color=bar_clrs, edgecolor='none', width=0.65)
                for i, b in enumerate(bars_txn):
                    if df_chart.iloc[i]['Bulan_Num'] in bot3['Bulan_Num'].values:
                        ax_txn.text(b.get_x()+b.get_width()/2, b.get_height()+0.3,
                                    f"{int(b.get_height()):,}", ha='center', fontsize=8,
                                    fontweight='bold', color='#FF6584')
                ax_txn.axhline(avg_order_global, color='#6C63FF', linewidth=1.5,
                               linestyle='--', alpha=0.7, label=f'Rata-rata: {int(avg_order_global):,}')
                ax_txn.set_xticks(range(len(df_chart)))
                ax_txn.set_xticklabels(df_chart['Bulan_Nama'], fontsize=8)
                ax_txn.set_ylabel('Jumlah Order', fontsize=8)
                ax_txn.legend(fontsize=8, framealpha=0)
                ax_txn.grid(True, axis='y'); sns.despine(); fig_txn.patch.set_alpha(0)
                st.pyplot(fig_txn); plt.close()
                st.markdown("</div>", unsafe_allow_html=True)
 
                # Footer insight
                st.markdown(f"""
                <div style='margin-top:0.8rem;background:#0D0D1A;border:1px solid #FF658433;
                            border-radius:10px;padding:0.9rem 1.2rem;
                            display:flex;align-items:center;gap:0.8rem;'>
                    <div style='font-size:1.2rem;'>⚠️</div>
                    <div style='font-size:0.78rem;color:#AAAACC;font-family:"IBM Plex Mono",monospace;line-height:1.6;'>
                        Bulan terendah (<strong style='color:#FF6584'>{bot3.iloc[0]['Bulan_Nama']}</strong>) hanya mencatat 
                        <strong style='color:#F0F0FF'>{lowest_val:,} order</strong> — 
                        rata-rata <strong style='color:#FF6584'>–{selisih_bot:.1f}%</strong> 
                        di bawah rata-rata bulanan <strong style='color:#F0F0FF'>({int(avg_order_global):,} order/bln)</strong>.
                        Perbaiki konten sesuai saran Recovery Protocol di bawah.
                    </div>
                </div>
                </div>
                """, unsafe_allow_html=True)
 
        # ── Saran perbaikan jika gagal ─────────────────────────
        if pred == 0:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            saran = []
            if len(caption_input) < 100:
                saran.append("📝 Caption terlalu pendek — tulis minimal 100 karakter.")
            if caption_input.count('#') < 5:
                saran.append("🔖 Hashtag kurang — tambahkan minimal 10–20 hashtag relevan.")
            if tipe_input == 'Foto / Carousel':
                saran.append("🎬 Coba format Reels/Video — engagement biasanya lebih tinggi.")
            if not any(w in caption_input.lower() for w in ['promo','diskon','harga','order']):
                saran.append("🛒 Tidak ada kata penjualan — tambahkan 'Promo', 'Diskon', atau 'Order'.")
            if not any(w in caption_input.lower() for w in ['dm','wa','link','bio','hubungi']):
                saran.append("📣 Tidak ada CTA — tambahkan 'DM kami' atau 'Klik link di bio'.")
            if not saran:
                saran.append("⏰ Coba ubah jam posting ke prime time (18.00–21.00).")
 
            # ← WAJIB ADA INI sebelum loop
            saran_html = ""
            for s in saran:
                saran_html += (
                    f"<div style='padding:0.6rem 0;border-bottom:1px solid #1E1E35;"
                    f"color:#AAAACC;font-size:0.95rem;'>→ {s}</div>"
                )

            st.markdown(
                "<div style='background:#111120;border:1px solid #1E1E35;"
                "border-radius:16px;padding:1.2rem 1.5rem;margin-top:1rem;'>"
                "<div class='section-tag'>RECOVERY PROTOCOL</div>"
                "<div style='font-size:1.1rem;font-weight:700;color:#F0F0FF;"
                "margin-bottom:0.8rem;'>Saran Perbaikan</div>"
                + saran_html +
                "</div>",
                unsafe_allow_html=True
            )
            
 
# ══════════════════════════════════════════════════════════
# [C] DIAGNOSTIC ANALYSIS
# ══════════════════════════════════════════════════════════
elif "Diagnostic Analysis" in menu:
    render_header("DIAGNOSTIC ENGINE", "DIAGNOSTIC ANALYSIS")

    if df is None:
        st.error(f"❌ Gagal memuat data: {err}")
        st.stop()

    import plotly.graph_objects as go
    import plotly.express as px

    time_map_label = {0:'Pagi (05-11)', 1:'Siang (11-15)',
                      2:'Sore (15-18)', 3:'Malam (18-05)'}
    df['Waktu'] = df['time_cat'].map(time_map_label)
    df['is_payday_lbl'] = df['dt'].dt.day.apply(
        lambda x: 'Tanggal Gajian' if (x >= 25 or x <= 5) else 'Tanggal Biasa'
    )

    col1, col2 = st.columns(2)

    # ── Grafik 1: Interaksi per Tipe Konten ──
    with col1:
        df_box = df[['type','total_interaction']].copy()
        fig = go.Figure()
        for i, tipe in enumerate(df_box['type'].unique()):
            data_tipe = df_box[df_box['type'] == tipe]['total_interaction']
            fig.add_trace(go.Box(
                y=data_tipe,
                name=tipe,
                marker_color=PALETTE[i % len(PALETTE)],
                line_color=PALETTE[i % len(PALETTE)],
                hovertemplate='<b>%{x}</b><br>Nilai: <b>%{y:.0f}</b><br><i>Distribusi interaksi tipe ini</i><extra></extra>'
            ))
        fig.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color='#AAAACC', size=9), showline=False),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Total Interaksi', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#6C63FF', font_size=12),
            legend=dict(font=dict(color='#AAAACC', size=9), bgcolor='#111120'),
            showlegend=False
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>CONTENT SCHEMA</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Interaksi per Tipe Konten</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 2: Peluang Sukses per Waktu ──
    with col2:
        df_waktu = df.groupby('Waktu')['is_success'].mean().reset_index()
        order_waktu = ['Pagi (05-11)', 'Siang (11-15)', 'Sore (15-18)', 'Malam (18-05)']
        df_waktu['Waktu'] = pd.Categorical(df_waktu['Waktu'], categories=order_waktu, ordered=True)
        df_waktu = df_waktu.sort_values('Waktu')

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=df_waktu['Waktu'],
            y=df_waktu['is_success'],
            marker_color=PALETTE[:len(df_waktu)],
            marker_line_width=0,
            hovertemplate='<b>%{x}</b><br>Peluang Sukses: <b>%{y:.2f}</b><br><i>Rata-rata konten sukses di waktu ini</i><extra></extra>'
        ))
        fig2.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color='#AAAACC', size=9), showline=False),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Avg Peluang Sukses', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#6C63FF', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>TEMPORAL INDEX</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Peluang Sukses per Waktu Posting</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    # ── Grafik 3: Efek Gajian ──
    with col3:
        df_payday = df.groupby('is_payday_lbl')['is_success'].mean().reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=df_payday['is_payday_lbl'],
            y=df_payday['is_success'],
            marker_color=[PALETTE[0], PALETTE[2]],
            marker_line_width=0,
            width=0.45,
            hovertemplate='<b>%{x}</b><br>Peluang Sukses: <b>%{y:.2f}</b><br><i>Pengaruh tanggal gajian terhadap performa konten</i><extra></extra>'
        ))
        fig3.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=280,
            xaxis=dict(tickfont=dict(color='#AAAACC', size=10), showline=False),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Avg Peluang Sukses', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#FFB347', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>PAYDAY EFFECT</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Pengaruh Tanggal Gajian</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 4: Korelasi Order vs Interaksi ──
    with col4:
        r = df['Jumlah_Order'].corr(df['total_interaction'])

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df['Jumlah_Order'],
            y=df['total_interaction'],
            mode='markers',
            marker=dict(color=PALETTE[3], size=5, opacity=0.4),
            hovertemplate='Order: <b>%{x:.0f}</b><br>Interaksi: <b>%{y:.0f}</b><extra></extra>'
        ))
        # Tambah garis regresi
        m, b = np.polyfit(df['Jumlah_Order'], df['total_interaction'], 1)
        x_line = np.linspace(df['Jumlah_Order'].min(), df['Jumlah_Order'].max(), 100)
        fig4.add_trace(go.Scatter(
            x=x_line, y=m * x_line + b,
            mode='lines',
            line=dict(color=PALETTE[1], width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        fig4.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=30, b=10), height=280,
            title=dict(text=f'Pearson r = {r:.3f}',
                       font=dict(color='#AAAACC', size=10), x=0.5),
            xaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Jumlah Order', font=dict(size=9, color='#AAAACC'))),
            yaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Total Interaksi', font=dict(size=9, color='#AAAACC'))),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#FF6584', font_size=12),
            showlegend=False
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>CORRELATION MATRIX</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Korelasi Penjualan vs Interaksi</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 5: Feature Importance ──
    if model is not None:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        fitur_cols = list(model.feature_names_in_)
        label_map = {
            'Bulan_Num':'Bulan','time_category':'Jam Posting',
            'is_weekend':'Weekend','is_payday':'Tanggal Gajian',
            'caption_length':'Panjang Caption','hashtag_count':'Jumlah Hashtag',
            'is_question':'Ada Pertanyaan','is_hard_selling':'Kata Jualan',
            'has_cta':'Ada CTA','type_encoded':'Tipe Konten'
        }
        imp = pd.Series(model.feature_importances_, index=fitur_cols).sort_values()
        imp.index = [label_map.get(i, i) for i in imp.index]

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            x=imp.values,
            y=imp.index,
            orientation='h',
            marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(imp))],
            marker_line_width=0,
            hovertemplate='<b>%{y}</b><br>Importance: <b>%{x:.4f}</b><br><i>Semakin tinggi semakin berpengaruh</i><extra></extra>'
        ))
        fig5.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10, r=10, t=10, b=10), height=320,
            xaxis=dict(tickfont=dict(color='#666680', size=9),
                       gridcolor='#1E1E35', showline=False,
                       title=dict(text='Importance Score', font=dict(size=9, color='#AAAACC'))),
            yaxis=dict(tickfont=dict(color='#AAAACC', size=9), showline=False),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF',
                            bordercolor='#6C63FF', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>FEATURE IMPORTANCE</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Faktor Penentu Kesuksesan Konten</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig5, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ── Grafik 6: Tipe Konten (Donut Chart) ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    
    df['Tipe_Konten'] = df['type'].apply(lambda x: 'Video/Reels' if any(k in str(x).lower() for k in ['reel','video']) else 'Foto/Carousel')
    df_sukses = df[df['is_success'] == 1]
    tipe_count  = df['Tipe_Konten'].value_counts()
    tipe_sukses = df_sukses['Tipe_Konten'].value_counts()
    colors_pie = PALETTE[:2]

    col_pie1, col_pie2 = st.columns(2)

    with col_pie1:
        fig_pie1 = go.Figure()
        fig_pie1.add_trace(go.Pie(
            labels=tipe_count.index,
            values=tipe_count.values,
            hole=0.6,
            marker=dict(colors=colors_pie, line=dict(color='#0A0A0F', width=2)),
            textfont=dict(size=10, color='#F0F0FF'),
            hovertemplate='<b>%{label}</b><br>Jumlah: <b>%{value}</b><br>Proporsi: <b>%{percent}</b><extra></extra>'
        ))
        fig_pie1.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10,r=10,t=10,b=10), height=250,
            annotations=[dict(text=f'Total<br>{len(df):,}', x=0.5, y=0.5,
                              font=dict(size=12, color='#F0F0FF', family='IBM Plex Mono'),
                              showarrow=False)],
            legend=dict(font=dict(color='#AAAACC', size=9), bgcolor='#111120',
                        bordercolor='#1E1E35'),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>CONTENT TYPE ANALYSIS</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Proporsi Tipe Konten Keseluruhan</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_pie1, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_pie2:
        fig_pie2 = go.Figure()
        fig_pie2.add_trace(go.Pie(
            labels=tipe_sukses.index,
            values=tipe_sukses.values,
            hole=0.6,
            marker=dict(colors=colors_pie, line=dict(color='#0A0A0F', width=2)),
            textfont=dict(size=10, color='#F0F0FF'),
            hovertemplate='<b>%{label}</b><br>Jumlah: <b>%{value}</b><br>Proporsi: <b>%{percent}</b><extra></extra>'
        ))
        fig_pie2.update_layout(
            paper_bgcolor='#111120', plot_bgcolor='#111120',
            margin=dict(l=10,r=10,t=10,b=10), height=250,
            annotations=[dict(text=f'Sukses<br>{len(df_sukses):,}', x=0.5, y=0.5,
                              font=dict(size=12, color='#F0F0FF', family='IBM Plex Mono'),
                              showarrow=False)],
            legend=dict(font=dict(color='#AAAACC', size=9), bgcolor='#111120',
                        bordercolor='#1E1E35'),
            hoverlabel=dict(bgcolor='#1E1E35', font_color='#F0F0FF', font_size=12),
        )
        st.markdown(
            "<div style='background:#111120;border:1px solid #1E1E35;"
            "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
            "<div class='section-tag'>CONTENT TYPE ANALYSIS</div>"
            "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
            "Proporsi Tipe Konten pada Postingan Sukses</div>",
            unsafe_allow_html=True
        )
        st.plotly_chart(fig_pie2, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 7: Tren Order per Bulan ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    df_tren = df.groupby('Bulan_Num')['Jumlah_Order'].sum().reset_index()
    df_tren['Nama_Bulan'] = df_tren['Bulan_Num'].map(NAMA_BULAN)
    df_tren = df_tren.sort_values('Bulan_Num')
    idx_max = df_tren['Jumlah_Order'].idxmax()
    bar_colors_tren = ['#FF6584' if i == idx_max else '#2EC4B6' for i in df_tren.index]

    fig_tren = go.Figure()
    fig_tren.add_trace(go.Bar(
        x=df_tren['Nama_Bulan'], y=df_tren['Jumlah_Order'],
        marker_color=bar_colors_tren, marker_line_width=0, opacity=0.4,
        hovertemplate='<b>%{x}</b><br>Order: <b>%{y:,}</b><extra></extra>',
        showlegend=False
    ))
    fig_tren.add_trace(go.Scatter(
        x=df_tren['Nama_Bulan'], y=df_tren['Jumlah_Order'],
        mode='lines+markers',
        line=dict(color='#2EC4B6', width=2.5),
        marker=dict(size=9, color='#2EC4B6', line=dict(color='white', width=2)),
        hovertemplate='<b>%{x}</b><br>Order: <b>%{y:,}</b><extra></extra>',
        showlegend=False
    ))
    fig_tren.update_layout(
        paper_bgcolor='#111120', plot_bgcolor='#111120',
        margin=dict(l=10,r=10,t=10,b=10), height=250,
        xaxis=dict(tickfont=dict(color='#AAAACC',size=9), showline=False, gridcolor='#1E1E35'),
        yaxis=dict(tickfont=dict(color='#666680',size=9), gridcolor='#1E1E35',
                   title=dict(text='Jumlah Order',font=dict(size=9,color='#AAAACC'))),
        hoverlabel=dict(bgcolor='#1E1E35',font_color='#F0F0FF',font_size=12),
    )
    st.markdown(
        "<div style='background:#111120;border:1px solid #1E1E35;"
        "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
        "<div class='section-tag'>TRANSACTION TREND</div>"
        "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
        "Tren Jumlah Order per Bulan</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(fig_tren, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Grafik 8: Order Tertinggi vs Terendah ──
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    df_tren_sorted = df_tren.sort_values('Jumlah_Order', ascending=False)
    top3 = df_tren_sorted.head(3)
    bot3 = df_tren_sorted.tail(3)
    df_cmp = pd.concat([top3, bot3])
    colors_cmp = ['#2EC4B6','#36CFC9','#5CDBD3','#FF6584','#FF85A1','#FFB3C1']

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        x=df_cmp['Nama_Bulan'], y=df_cmp['Jumlah_Order'],
        marker_color=colors_cmp, marker_line_width=0,
        text=[f'{int(v):,}' for v in df_cmp['Jumlah_Order']],
        textposition='outside',
        textfont=dict(color='#F0F0FF', size=11),
        hovertemplate='<b>%{x}</b><br>Order: <b>%{y:,}</b><extra></extra>'
    ))
    fig_cmp.update_layout(
        paper_bgcolor='#111120', plot_bgcolor='#111120',
        margin=dict(l=10,r=10,t=10,b=10), height=280,
        xaxis=dict(tickfont=dict(color='#AAAACC',size=10), showline=False, gridcolor='#1E1E35'),
        yaxis=dict(tickfont=dict(color='#666680',size=9), gridcolor='#1E1E35',
                   title=dict(text='Jumlah Order',font=dict(size=9,color='#AAAACC'))),
        hoverlabel=dict(bgcolor='#1E1E35',font_color='#F0F0FF',font_size=12),
        showlegend=False
    )
    st.markdown(
        "<div style='background:#111120;border:1px solid #1E1E35;"
        "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
        "<div class='section-tag'>TRANSACTION COMPARISON</div>"
        "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.5rem;'>"
        "Perbandingan Bulan Order Tertinggi vs Terendah</div>",
        unsafe_allow_html=True
    )
    st.plotly_chart(fig_cmp, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# [D] HISTORICAL DRILL-DOWN
# ══════════════════════════════════════════════════════════
elif "Drill-Down" in menu:
    render_header("DATA REPOSITORY", "HISTORICAL DRILL-DOWN")
    
    st.markdown("""
    <style>
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background-color: #1E1E45 !important;
        border: 1px solid #4A4A8A !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
        color: #A0A0FF !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] button {
        color: #7070CC !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] button:hover {
        color: #FF6584 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if df is None:
        st.error(f"❌ Gagal memuat data: {err}")
        st.stop()

    # Filter panel
    st.markdown("""
    <div class='content-card' style='padding: 0.5rem 0.8rem; margin-bottom: 0.2rem;'>
        <div class='section-tag' style='font-size: 0.7rem;'>CONTENT SCHEMA</div>
        <div style='font-size:0.9rem; font-weight:700; color:#F0F0FF;'>
            Interaksi per Tipe Konten
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='section-tag'>FILTER PARAMETERS</div>", unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        filter_label = st.multiselect("Label", ['✅ Sukses', '❌ Gagal'], default=['✅ Sukses', '❌ Gagal'])
    with fc2:
        semua_tipe  = df['type'].unique().tolist()
        filter_tipe = st.multiselect("Schema Type", semua_tipe, default=semua_tipe)
    with fc3:
        semua_bulan  = sorted(df['Bulan_Num'].unique().tolist())
        filter_bulan = st.multiselect("Periode", semua_bulan, default=semua_bulan, format_func=lambda x: NAMA_BULAN.get(x, x))
    st.markdown("<p style='color:#AAAACC;font-size:0.85rem;margin-bottom:0.2rem;'>🔍 Filter by KEYWORD_INDEX...</p>", unsafe_allow_html=True)
    keyword = st.text_input("", placeholder="Ketik kata kunci dalam caption...", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
    
    df_filter = df.copy()
    if filter_label:  df_filter = df_filter[df_filter['Label'].isin(filter_label)]
    if filter_tipe:   df_filter = df_filter[df_filter['type'].isin(filter_tipe)]
    if filter_bulan:  df_filter = df_filter[df_filter['Bulan_Num'].isin(filter_bulan)]
    if keyword:       df_filter = df_filter[df_filter['caption'].str.contains(keyword, case=False, na=False)]

    st.markdown(f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:0.72rem;color:#A0A0FF;letter-spacing:0.1em;margin-bottom:1rem;'>{len(df_filter):,} CONTENT_OBJECTS FOUND</div>", unsafe_allow_html=True)

    # Drill-down rows
    cols_header = st.columns([3, 1.5, 1, 1.5, 1.5, 1.5])
    for h, txt in zip(cols_header, ["CONTENT_OBJECT","TIMESTAMP","SCHEMA_TYPE","PERFORMANCE_INDEX","TELEMETRY","TRANSAKSI"]):
        h.markdown(f"""
            <div style='
                font-size:0.62rem;
                letter-spacing:0.15em;
                color:#FFFFFF; 
                font-family:"IBM Plex Mono",monospace;
                text-transform:uppercase;
                padding-bottom:0.5rem;
                border-bottom:1px solid #1E1E35;
                font-weight: 600;
            '>
                {txt}
            </div>
        """, unsafe_allow_html=True)
    
    for i, (_, row) in enumerate(df_filter.head(30).iterrows()):
        score_pct = int(row['Total_Score'] * 100)
        caption_preview = str(row.get('caption',''))[:40] + '...' if len(str(row.get('caption',''))) > 40 else str(row.get('caption',''))
        tipe_pill = f"<span class='pill {'pill-video' if 'video' in str(row['type']).lower() or 'reels' in str(row['type']).lower() else 'pill-photo'}'>{str(row['type']).upper()}</span>"
        c1, c2, c3, c4, c5, c6 = st.columns([3, 1.5, 1, 1.5, 1.5, 1.5])
        with c1:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;padding:0.7rem 0;border-bottom:1px solid #12121F;'>
                <div class='drill-number'>{i+1}</div>
                <div>
                    <div class='drill-title'>{caption_preview.upper() or f'POST #{i+1}'}</div>
                    <div class='drill-meta'>⊕ {int(row.get('total_interaction',0))} interactions</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with c2:
            dt_str = row['dt'].strftime('%Y / %m / %d') if hasattr(row['dt'], 'strftime') else str(row['dt'])[:10]
            st.markdown(f"<div style='padding:0.7rem 0;border-bottom:1px solid #12121F;font-family:\"IBM Plex Mono\",monospace;font-size:0.75rem;color:#888899;'>📅 {dt_str}</div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div style='padding:0.7rem 0;border-bottom:1px solid #12121F;'>{tipe_pill}</div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div style='padding:0.7rem 0;border-bottom:1px solid #12121F;'>
                <div class='perf-bar-wrap'>
                    <div class='perf-bar'><div class='perf-bar-fill' style='width:{score_pct}%'></div></div>
                    <div class='perf-score'>{score_pct}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with c5:
            st.markdown(f"""
            <div style='padding:0.7rem 0;border-bottom:1px solid #12121F;font-family:\"IBM Plex Mono\",monospace;font-size:0.75rem;color:#888899;'>
                {int(row.get('likes',0)):,} <span style='color:#444460'>LIKES</span><br>
                {int(row.get('comments',0)):,} <span style='color:#444460'>COMM</span>
            </div>""", unsafe_allow_html=True)

        with c6:
            order_val = int(row.get('Jumlah_Order', 0))
            bulan_nm  = row.get('Bulan_Nama', '')
            st.markdown(f"""
            <div style='padding:0.7rem 0;border-bottom:1px solid #12121F;
                        font-family:"IBM Plex Mono",monospace;font-size:0.75rem;color:#888899;'>
                <span style='color:#2ECC71;font-weight:700;'>{order_val:,}</span>
                <span style='color:#444460;'> ORDER</span><br>
                <span style='#FFFFFF;'>{bulan_nm}</span>
            </div>""", unsafe_allow_html=True)

    # Mini summary & download
    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='background:#111120;border:1px solid #1E1E35;"
        "border-radius:16px;padding:1.2rem 1.5rem 0.5rem 1.5rem;margin-bottom:1rem;'>"
        "<div class='section-tag'>SUMMARY TELEMETRY</div>"
        "<div style='font-size:0.9rem;font-weight:700;color:#F0F0FF;margin-bottom:0.8rem;'>"
        "Ringkasan Data Terfilter</div>",
        unsafe_allow_html=True
    )
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1: metric_card("TOTAL POST",   f"{len(df_filter)}", "", "📄")
    with sc2: metric_card("AVG INTERACT", f"{df_filter['total_interaction'].mean():.0f}", "", "📊")
    with sc3: metric_card("AVG ORDER",    f"{df_filter['Jumlah_Order'].mean():.0f}", "", "🛒")
    with sc4: metric_card("SUCCESS RATE", f"{df_filter['is_success'].mean()*100:.1f}%", "", "✅")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)
    kolom_tampil = ['dt','type','likes','comments','total_interaction','Jumlah_Order','Total_Score','Label']
    kolom_ada    = [c for c in kolom_tampil if c in df_filter.columns]
    df_dl = df_filter[kolom_ada].copy()
    df_dl['dt'] = df_dl['dt'].dt.strftime('%d %b %Y')
    csv = df_dl.to_csv(index=False).encode('utf-8')
    st.download_button("⬇  EXPORT DATA (.csv)", data=csv, file_name='socialinsight_export.csv', mime='text/csv')
    st.markdown("</div>", unsafe_allow_html=True)