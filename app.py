import streamlit as st

st.title("🚀 Aplikasi AI Pertama Saya")
st.write("Halo Dunia! Aplikasi Streamlit saya berhasil berjalan.")

# Input teks dari pengguna
nama = st.text_input("Masukkan nama Anda:")
if nama:
    st.write(f"Selamat datang di dunia AI, {nama}!")
