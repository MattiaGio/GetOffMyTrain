from ultralytics import YOLO

# carica un modello pre-addestrato
model = YOLO("yolov8n.pt")  # oppure yolov8s.pt

# addestramento
model.train(
    data="models_training/ticket_to_ride_trains_dataset/data.yaml",
    epochs=80,
    imgsz=640,
    batch=8,
    device="cpu",   # se hai GPU usa device=0
    name="ticket2ride_trains_v2",
)

# esporta come trains.pt
model.export(format="pt")