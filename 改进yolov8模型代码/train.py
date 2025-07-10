# #### 正常训练
from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('./ultralytics/cfg/models/v8/yolov8_C2f_iAFF.yaml').load("./yolov8n.pt")    # 创建模型并加载预训练权重
    model.train(
        epochs=100,          # 训练的轮数
        batch=128,
        # batch=26,            # 每次加载的图片数量
        data='./ultralytics/cfg/datasets/my_dataset.yaml',
        device='0,1,2,3'
        # device='1'  # 使用四张显卡 (设备ID，0,1,2,3代表4张GPU)
    )



# ##### 断点续训
# from ultralytics import YOLO
#
# # 加载检查点
# model = YOLO('/home/lzp/ultralytics_yolov8-main/runs/detect/train252/weights/last.pt')  # 替换为你的 last.pt 路径
#
# # 继续训练
# model.train(
#     data='/home/lzp/ultralytics_yolov8-main/ultralytics/cfg/datasets/my_dataset.yaml',
#     epochs=100,
#     imgsz=640,
#     device='0,1,2,3',
#     batch=128,
#     resume=True
# )



