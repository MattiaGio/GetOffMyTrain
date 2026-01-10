from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # modello piccolo e veloce

model.train(
    data="models_training/ticket_to_ride_corners_dataset/data.yaml",
    epochs=50,
    imgsz=1024,
    batch=8,
)