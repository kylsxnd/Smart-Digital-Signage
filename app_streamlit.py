import streamlit as st
import cv2
from ultralytics import YOLO
import time

# --- 1. KONFIGURASI IKLAN YOUTUBE ---
ADS_URL = {
    "male": "https://www.youtube.com/watch?v=PDlzQtLk05M",   # Iklan Nivea Men Cool Kick
    "female": "https://www.youtube.com/watch?v=ZV3K2gzTHC8"  # Iklan Azzura Jelly Liptint
}

# --- 1.5 KONFIGURASI LINK PEMBELIAN ---
BUY_LINKS = {
    "male": {
        "text": "🛒 Beli NIVEA MEN Cool Kick",
        "url": "https://shopee.co.id/search?keyword=nivea%20men%20cool%20kick"
    },
    "female": {
        "text": "🛒 Beli AZZURA Jelly Liptint",
        "url": "https://shopee.co.id/search?keyword=azzura%20jellycious%20liptint"
    }
}

# Fungsi Sakti untuk Embed YouTube
def get_youtube_iframe(url):
    if "watch?v=" in url:
        video_id = url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = url 
    
    return f'''
        <iframe width="100%" height="400" 
        src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1&controls=0&modestbranding=1" 
        frameborder="0" allow="autoplay; encrypted-media" allowfullscreen>
        </iframe>
    '''

# --- 2. LOAD MODEL YOLOv8 ---
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# --- 3. SETTING UI STREAMLIT ---
st.set_page_config(page_title="Smart Digital Signage - AI", layout="wide")
st.title(": Smart Digital Signage")
st.markdown("---")

# Inisialisasi Session State (Kosong di awal)
if 'active_ad' not in st.session_state:
    st.session_state.active_ad = None
if 'last_gender' not in st.session_state:
    st.session_state.last_gender = None

# Membuat Layout 2 Kolom
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🤖 AI Face Detection")
    frame_placeholder = st.empty() 

with col2:
    st.subheader("📺 Targeted Advertisement")
    video_container = st.empty()
    button_container = st.empty() # Wadah khusus buat tombol beli biar dinamis
    
    # Tampilan awal pas aplikasi baru di-run
    if st.session_state.active_ad is None:
        video_container.info("Menunggu audiens di depan kamera...")
    else:
        video_container.markdown(get_youtube_iframe(st.session_state.active_ad), unsafe_allow_html=True)
        if st.session_state.last_gender in BUY_LINKS:
            btn_data = BUY_LINKS[st.session_state.last_gender]
            button_container.link_button(btn_data["text"], btn_data["url"])

# --- 4. LOGIKA KAMERA & DETEKSI ---
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Kamera tidak terdeteksi.")
        break

    # Jalankan prediksi YOLOv8
    results = model.predict(frame, conf=0.5, verbose=False)
    
    current_gender = None
    
    # Ambil hasil deteksi (kalau ada muka)
    for r in results:
        if len(r.boxes) > 0:
            cls_id = int(r.boxes.cls[0])
            current_gender = model.names[cls_id].lower() 
            break 

    # --- LOGIKA GANTI IKLAN (UPDATE SESUAI KEHADIRAN) ---
    if current_gender:
        # JIKA ADA ORANG: Putar video & Munculkan link sesuai gender
        if current_gender in ADS_URL and current_gender != st.session_state.last_gender:
            st.session_state.active_ad = ADS_URL[current_gender]
            st.session_state.last_gender = current_gender
            
            # Tampilkan Video
            video_container.markdown(get_youtube_iframe(st.session_state.active_ad), unsafe_allow_html=True)
            
            # Tampilkan Tombol Beli
            button_container.empty() # Bersihin tombol sebelumnya (kalau ada)
            btn_data = BUY_LINKS[current_gender]
            button_container.link_button(btn_data["text"], btn_data["url"])
            
    else:
        # JIKA ORANGNYA PERGI (Gak ada muka yang kedetek)
        if st.session_state.last_gender is not None:
            # Reset state ke awal
            st.session_state.active_ad = None
            st.session_state.last_gender = None
            
            # Hapus video dan tombol, kembalikan tulisan standby
            video_container.info("Menunggu audiens di depan kamera...")
            button_container.empty()
            
    # --- VISUALISASI ---
    annotated_frame = results[0].plot()
    frame_placeholder.image(annotated_frame, channels="BGR", width="stretch")

    time.sleep(0.01)

cap.release()   