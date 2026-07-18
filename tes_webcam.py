import cv2
from ultralytics import YOLO
import webbrowser
import time

# 1. LOAD MODEL (Pastiin path ini sesuai sama folder hasil training lu yang terakhir)
# Gue pake 'pi_gender_final_boss2' sesuai hasil training 90% lu tadi
model_path = 'runs/detect/pi_gender_final_boss2/weights/best.pt'
model = YOLO(model_path)

# 2. KONFIGURASI IKLAN
# Ganti link ini sesuai kebutuhan presentasi lu nanti
AD_LINKS = {
    'male': 'https://www.rolex.com',      # Contoh iklan jam tangan
    'female': 'https://www.sephora.com'   # Contoh iklan skincare
}

# 3. INISIALISASI WEBCAM
cap = cv2.VideoCapture(1)

# Variabel State / Kondisi Sistem
waiting_confirmation = False
detected_gender = None
last_action_time = 0

print("--- SISTEM SMART SIGNAGE AKTIF ---")
print("Menunggu pengunjung terdeteksi...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Balik frame biar kayak cermin (opsional, biar enak liatnya)
    frame = cv2.flip(frame, 1)

    # MODE A: MENCARI PENGUNJUNG
    if not waiting_confirmation:
        # Jalankan deteksi YOLOv8
        results = model(frame, conf=0.6, verbose=False)
        
        for r in results:
            for c in r.boxes.cls:
                detected_gender = r.names[int(c)]
                waiting_confirmation = True # Kunci sistem ke mode konfirmasi
                break # Ambil satu orang pertama saja
        
        # Tampilan Standby
        cv2.putText(frame, "STATUS: MENCARI PENGUNJUNG...", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # MODE B: KONFIRMASI (USER INTERACTION)
    else:
        # Bikin Box Overlay Gelap di bawah biar teks jelas
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 350), (640, 480), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Tampilkan Pesan Konfirmasi
        msg = f"TERDETEKSI: {detected_gender.upper()}"
        prompt = "MAU LIHAT PROMO? (Tekan 'Y' = YA / 'N' = TIDAK)"
        
        cv2.putText(frame, msg, (180, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, prompt, (70, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Logika Input Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y') or key == ord('Y'):
            print(f">>> MEMBUKA IKLAN {detected_gender.upper()}...")
            webbrowser.open(AD_LINKS[detected_gender])
            
            # Kasih jeda dikit biar sistem gak langsung deteksi lagi
            waiting_confirmation = False
            time.sleep(2) 
            
        elif key == ord('n') or key == ord('N'):
            print(">>> PENGUNJUNG MENOLAK.")
            waiting_confirmation = False
            time.sleep(1) # Jeda biar gak langsung kekunci lagi

    # Tampilkan Hasil ke Layar
    cv2.imshow("SMART DIGITAL SIGNAGE - INTERACTIVE", frame)

    # Keluar program tekan 'Q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()