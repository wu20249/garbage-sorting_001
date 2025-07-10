from ultralytics import YOLO

model = YOLO("./runs/detect/train255SimSPPF+MPDIoU/weights/best.pt")  # load a custom model

# model = YOLO("./runs/detect/train33CBAM/weights/best.pt")
# Predict with the model

results = model("./my_datasetbig/images/ziji/yao.jpg",save=True)
