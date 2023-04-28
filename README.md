# yolov8_train
* Colab Demo: [<a href="https://colab.research.google.com/drive/1Zu2fnRvUE3VUj56BC3RwrEb86jUomWmm?usp=share_link"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>](https://colab.research.google.com/drive/1Zu2fnRvUE3VUj56BC3RwrEb86jUomWmm?usp=share_link)
## 訓練步驟
### 1.安裝 ultralytics
前置環境: Python>=3.7 和 PyTorch>=1.7
```
pip install ultralytics
```
### 2.準備資料集
資料集結構
```
└── datasets
    └── coco128  ← 資料集名稱
        ├── images ← 存放影像
            ├── train
            └── val
        └── labels ← 存放標記檔
            ├── train
            └── val
```
### 3.準備yaml檔
```
請參考yolov8/coco128.yaml
```
### 4.開始訓練
請參考yolov8/train.yaml
```
cd yolov8
python train.py
```
model.train(data="coco128.yaml", epochs=500, batch=32, workers=4)可用參數
```
model:  # path to model file, i.e. yolov8n.pt, yolov8n.yaml
data:  # path to data file, i.e. coco128.yaml
epochs: 100  # number of epochs to train for
patience: 50  # epochs to wait for no observable improvement for early stopping of training
batch: 16  # number of images per batch (-1 for AutoBatch)
imgsz: 640  # size of input images as integer or w,h
save: True  # save train checkpoints and predict results
save_period: -1 # Save checkpoint every x epochs (disabled if < 1)
cache: False  # True/ram, disk or False. Use cache for data loading
device:  # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
workers: 8  # number of worker threads for data loading (per RANK if DDP)
project:  # project name
name:  # experiment name, results saved to 'project/name' directory
exist_ok: False  # whether to overwrite existing experiment
pretrained: False  # whether to use a pretrained model
optimizer: SGD  # optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']
verbose: True  # whether to print verbose output
seed: 0  # random seed for reproducibility
deterministic: True  # whether to enable deterministic mode
single_cls: False  # train multi-class data as single-class
image_weights: False  # use weighted image selection for training
rect: False  # support rectangular training if mode='train', support rectangular evaluation if mode='val'
cos_lr: False  # use cosine learning rate scheduler
close_mosaic: 10  # disable mosaic augmentation for final 10 epochs
resume: False  # resume training from last checkpoint
```

## Inference
請參考yolov8/predict_single_yolov8.py
```
cd yolov8
python predict_single_yolov8.py
```
