from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolov8l.yaml")  # build a new model from scratch
    model = YOLO("yolov8l.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="coco128.yaml", epochs=500, batch=32, workers=4)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    #results = model("D:/harrylin/smart_image/datasets/Scratch/AAMORA00.B160UAN03_ET.B16UAN3_P4_ASI1229.5Q2A86N600.5Q2A86N63.202.b.1662857460.jpg")  # predict on an image
    #success = model.export(format="onnx")  # export the model to ONNX format
