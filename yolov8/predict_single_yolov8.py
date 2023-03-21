from ultralytics import YOLO
import os
import cv2
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/detect/train2/weights/best.pt")  # load a pretrained model (recommended for training)

    #metrics = model.val()  # evaluate model performance on the validation set
    #img = cv2.imread("D:/harrylin/smart_image/datasets/Scratch/AAMORA00.B160UAN03_ET.B16UAN3_P4_ASI1229.5Q2A86N600.5Q2A86N63.202.b.1662857460.jpg")
    img = cv2.imread("bus.jpg")
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model(img, save=False)  # predict on an image
    print(results)
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        #masks = result.masks  # Masks object for segmenation masks outputs
        #probs = result.probs  # Class probabilities for classification outputs
        boxes = boxes.cpu().numpy()
        if not boxes:
            break #沒東西就跳過
        for i in range(len(boxes)):
            box_data = boxes[i].data #[x_left, y_top ,x_right ,y_bottom ,conf, classes]
            x_left = int(box_data[0][0])
            y_top = int(box_data[0][1])
            x_right = int(box_data[0][2])
            y_bottom = int(box_data[0][3])
            conf = box_data[0][4]
            classes = int(box_data[0][5])
            boundingBox = [
				(x_left, y_top), #左上頂點
				(x_left, y_bottom), #左下頂點
				(x_right, y_bottom), #右下頂點
				(x_right, y_top) #右上頂點
			]
            print(boundingBox)
            print(conf)
            print(classes)
            rectColor = (255, 0, 0)
            textCoord = (x_left, y_top - 10) #文字位置
            pstring = str(int(100 * conf)) + "%" #信心度
            # 在影像中標出Box邊界和類別、信心度
            cv2.rectangle(img, boundingBox[0], boundingBox[2], rectColor, 2)
            cv2.putText(img, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)
        cv2.imwrite("predict/test.jpg", img)