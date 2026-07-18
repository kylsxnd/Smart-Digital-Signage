from ultralytics import YOLO

model = YOLO('best.pt') # Pake model terbaik lu
model.predict(source=0, show=True) # source=0 artinya webcam laptop