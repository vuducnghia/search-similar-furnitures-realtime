from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='./furniture_detection/data.yaml', epochs=50, imgsz=640, batch=16)


