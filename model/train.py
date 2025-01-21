from ultralytics import YOLO

# Build a YOLOv9c model from pretrained weight
model = YOLO('yolov9c-seg.pt')

# Display model information (optional)
model.info()

# Train the model for 100 epochs
results = model.train(data='data.yaml', epochs=100, imgsz=640, verbose=True, batch=4)


