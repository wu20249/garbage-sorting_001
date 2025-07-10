from ultralytics import YOLO

# Load a model
model = YOLO('./runs/detect/train255SimSPPF+MPDIoU/weights/best.pt')

# Export the model
model.export(format='rknn',opset=12)

