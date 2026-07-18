from ultralytics import YOLO

if __name__ == '__main__':
    # Pakai model dasar biar dia belajar ulang secara total
    model = YOLO('yolov8n.pt') 

    model.train(
        data='data.yaml',
        epochs=50,       # 50 sudah cukup buat dataset ribuan
        imgsz=640,
        device=0,        # GTX 1650 lu harusnya udah terdeteksi CUDA-nya
        name='pi_gender_final_boss', # Nama folder hasil akhir
        batch=16         # Kalau laptop lu nge-lag atau error "Out of Memory", ganti ke 8
    )