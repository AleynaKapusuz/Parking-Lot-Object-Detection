from ultralytics import YOLO

model = YOLO(r"C:\Users\Aleyna\Desktop\Aleyna\isGEDİK\ParkingLot\yolov8s.pt")

data = r"C:\Users\Aleyna\Desktop\Aleyna\isGEDİK\ParkingLot\data.yaml"

model.train(
    data=data,
    epochs=1,
    imgsz=720,
    workers=4,
    batch=32,
    fliplr=True,  # Yatay çevirme
    flipud=True,  # Dikey çevirme
    scale=0.2, # 80% - 120% arasında ölçekleme
    shear=10, # %10 eğme
    hsv_v=0.3,
    degrees=15,
    cache=True, # Verileri RAM’e önceden yükler → eğitim hızlanır
    cos_lr=True # Cosine learning rate decay → daha stabil sonuç
)
