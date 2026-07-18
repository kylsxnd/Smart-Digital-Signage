from ultralytics import YOLO
model = YOLO('best.pt') # Pastiin ini file hasil training terakhir lu
print(model.names)