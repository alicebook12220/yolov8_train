# yolov8_train

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
```
cd yolov8
python train.py
```
