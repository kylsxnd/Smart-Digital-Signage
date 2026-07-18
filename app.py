import streamlit as st
import cv2
from ultralytics import YOLO
import time

# --- CONFIGURASI ---
# Ganti link di bawah ini dengan link YouTube iklan lu
ADS_URL = {
    "male": "https://www.youtube.com/watch?v=link_iklan_pria",
    "female": "https://www.youtube.com/watch?v=link_iklan_wanita",
    "default": "https://www.youtube.com/watch?v=link_iklan_umum"
}

# Load Model YOLOv8 lu (pastikan file best.pt ada di folder yang sama)
@st.cache_resource
def load_model():
    # Model akan otomatis pake GPU GTX 1650 Ti lu kalau CUDA udah terinstall
    return YOLO('best.pt')

model = load_model()

# --- UI SETTINGS ---
st.set_page_config(page_title="Smart Digital Signage - AI", layout="wide")
st.title("☕ Kopi-AI: Smart Digital Signage")
st.markdown("---")

# Inisialisasi Session State buat nahan link video
if 'active_ad' not in st.session_state:
    st.session_state.active_ad = ADS_URL["default"]
if 'last_gender' not in st.session_state:
    st.session_state.last_gender = None

# Membuat 2 Kolom
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🤖 AI Face Detection")
    frame_placeholder = st.empty() # Tempat render kamera

with col2:
    st.subheader("📺 Targeted Advertisement")
    video_placeholder = st.empty() # Tempat render video YouTube
    # Render video pertama kali
    video_placeholder.video(st.session_state.active_ad, autoplay=True, muted=True)

# --- LOGIKA KAMERA & AI ---
cap = cv2.VideoCapture(0) # 0 adalah index webcam laptop

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.error("Gagal akses kamera.")
        break

    # 1. Jalankan Deteksi YOLOv8
    results = model.predict(frame, conf=0.5, verbose=False)
    
    current_gender = None
    
    # 2. Ambil Label Hasil Deteksi
    for r in results:
        for c in r.boxes.cls:
            current_gender = model.names[int(c)].lower() # 'male' atau 'female'
            break # Ambil satu wajah utama saja

    # 3. Logika Ganti Iklan (Hanya update jika gender berubah)
    if current_gender and current_gender != st.session_state.last_gender:
        if current_gender in ADS_URL:
            st.session_state.active_ad = ADS_URL[current_gender]
            st.session_state.last_gender = current_gender
            # Update video di kolom kanan
            video_placeholder.video(st.session_state.active_ad, autoplay=True, muted=True)
            
    # 4. Gambar Bounding Box di Frame (Visualisasi)
    annotated_frame = results[0].plot()
    
    # 5. Tampilkan Kamera di Streamlit
    frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

    # Sedikit delay biar CPU nggak kerja terlalu keras
    time.sleep(0.01)

cap.release()