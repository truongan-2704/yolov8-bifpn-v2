# import  warnings
# warnings.filterwarnings('ignore')

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'ultralytics/cfg/models/v8/yolov8-bifpn.yaml')

    model.train(
        data=r'data/coco_dataset/data.yaml',
        epochs=200,
        batch=32,  # Batch size
        amp=False,
        imgsz=640,  # Kích thước ảnh
        device='0',  # GPU id (0 nếu dùng GPU, "cpu" nếu dùng CPU)
        optimizer="SGD",  # Optimizer: SGD hoặc AdamW
        # lr0=0.01,                   # Learning rate ban đầu
        # weight_decay=0.0005,        # Hệ số weight decay
        # warmup_epochs=3.0,          # Số epoch warm-up
        # mosaic=1.0,                 # Mosaic augmentation
        # mixup=0.2,                  # Mixup augmentation
        # save=True,                  # Lưu kết quả train
        patience=20,
        cache=False,
        project='runs/train',
        name='exp'
    )