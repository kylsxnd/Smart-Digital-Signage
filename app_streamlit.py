import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# --- 1. KONFIGURASI IKLAN & LINK ---
ADS_URL = {
    "male": "https://www.youtube.com/watch?v=PDlzQtLk05M", 
    "female": "https://www.youtube.com/watch?v=ZV3K2gzTHC8"
}

BUY_LINKS = {
    "male": {"text": "🛒 Beli NIVEA MEN Cool Kick", "url": "https://shopee.co.id/search?keyword=nivea%20men%20cool%20kick"},
    "female": {"text": "🛒 Beli AZZURA Jelly Liptint", "url": "https://shopee.co.id/search?keyword=azzura%20jellycious%20liptint"}
}

# --- 2. FUNGSI SAKTI ---
@st.cache_resource
def load_model():
    return YOLO('best.pt') # Pastikan best.pt ada di folder root repo GitHub lu

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
        framebuffer="0" allow="autoplay; encrypted-media" allowfullscreen>
        </iframe>
    '''

# --- 3. UI STREAMLIT ---
st.set_page_config(page_title="Smart Digital Signage", layout="wide")
st.title("📺 Smart Digital Signage - AI Detector")

# Inisialisasi Session State
if 'active_ad' not in st.session_state: st.session_state.active_ad = None
if 'last_gender' not in st.session_state: st.session_state.last_gender = None

col1, col2 = st.columns([1, 1])

# --- 4. LOGIKA KAMERA (CLOUD FRIENDLY) ---
with col1:
    st.subheader("🤖 AI Face Detection")
    # Ganti cap.read() dengan st.camera_input
    img_file = st.camera_input("Tunjukkan wajah ke kamera...")

with col2:
    st.subheader("📺 Targeted Advertisement")
    video_container = st.empty()
    button_container = st.empty()

model = load_model()

if img_file is not None:
    # Ubah gambar dari Streamlit (bytes) ke format OpenCV
    bytes_data = img_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    # Jalankan prediksi
    results = model.predict(cv2_img, conf=0.5, verbose=False)
    
    # Ambil gender
    detected_gender = None
    for r in results:
        if len(r.boxes) > 0:
            cls_id = int(r.boxes.cls[0])
            detected_gender = model.names[cls_id].lower()
            break
    
    # Logika Iklan
    if detected_gender and detected_gender in ADS_URL:
        # Tampilkan hasil deteksi di col1
        annotated_frame = results[0].plot()
        col1.image(annotated_frame, channels="BGR", use_container_width=True)
        
        # Update Iklan (biar nggak reload terus)
        if detected_gender != st.session_state.last_gender:
            st.session_state.active_ad = ADS_URL[detected_gender]
            st.session_state.last_gender = detected_gender
            
        video_container.markdown(get_youtube_iframe(st.session_state.active_ad), unsafe_allow_html=True)
        button_container.link_button(BUY_LINKS[detected_gender]["text"], BUY_LINKS[detected_gender]["url"])
    else:
        video_container.warning("Gender tidak terdeteksi atau wajah tidak dikenali.")
else:
    video_container.info("Menunggu akses kamera...")
