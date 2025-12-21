from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # modello piccolo e veloce

model.train(
    data="ticket_to_ride_corners/data.yaml",
    epochs=50,
    imgsz=1024,
    batch=8,
)