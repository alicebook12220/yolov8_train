from ultralytics import YOLO
import os
import cv2
import numpy as np
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def img_segmentation(img, ymin, ymax, xmin, xmax):
    size = 96
    height = ymax -ymin
    width = xmax - xmin
    if height < size:
        diff = size - height
        ymin = int(ymin - diff/2)
        ymax = int(ymax + diff/2)
        if ymin < 0:
            ymin = 0
        if ymax > (img.shape[0] - 1):
            ymax = img.shape[0] - 1
    if width < size:
        diff = size - width
        xmin = int(xmin - diff/2)
        xmax = int(xmax + diff/2)
        if xmin < 0:
            xmin = 0
        if xmax > (img.shape[1] - 1):
            xmax = img.shape[1] - 1
    image_seg = img[ymin:ymax, xmin:xmax]
    return image_seg

if __name__ == '__main__':
    # Load a model
    model = YOLO("runs/detect/train6/weights/best.pt")  # load a pretrained model (recommended for training)

    #imagePath = "D:\\harrylin\\datasets\\Scratch_v1\\test\\deblur_clean_needreview_all\\*.jpg" 
    imagePath = "D:\\harrylin\\datasets\\Scratch_v1\\test\\I-Scratch\\*.jpg" 
    imagePath = "D:\\harrylin\\datasets\\Scratch_old\\*.jpg" 
    imagePath = "D:\\harrylin\\datasets\\defect_202301\\Scratch\\*\\*.jpg"
    #imagePath = "D:\\harrylin\\datasets\\other_seg_test\\*\\*\\*.jpg"
    imagePath = glob.glob(imagePath)
    imagePath.sort()
    print("len:", len(imagePath))
    for i, img_path in enumerate(imagePath):
        class_type = 1 #0:沒東西 1:other 2:刮傷
        img_name = img_path.split("\\")
        img_name = img_name[len(img_name) - 1]
        img_name = img_name.split(".jpg")
        img_name = img_name[0]
        img = cv2.imread(img_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img, save=False, verbose=False)  # predict on an image
        #print(results)
        for result in results:
            boxes = result.boxes  # Boxes object for bbox outputs
            #masks = result.masks  # Masks object for segmenation masks outputs
            #probs = result.probs  # Class probabilities for classification outputs
            boxes = boxes.cpu().numpy()
            if not boxes:
                class_type = 0
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
                if classes == 0: #and conf > 0.7:
                    class_type = 2
                    img_seg = img_segmentation(img, y_top, y_bottom, x_left, x_right)
                    cv2.imwrite("predict/img_seg/"+ img_name + "_" + str(i) + ".jpg", img_seg)
                #print(boundingBox)
                #print(conf)
                #print(classes)
                rectColor = (0, 255, 0)
                textCoord = (x_left, y_top - 10) #文字位置
                pstring = str(classes) + ":" + str(int(100 * conf)) + "%" #信心度
                # 在影像中標出Box邊界和類別、信心度
                #cv2.rectangle(img, boundingBox[0], boundingBox[2], rectColor, 2)
                #cv2.putText(img, pstring, textCoord, cv2.FONT_HERSHEY_DUPLEX, 1, rectColor, 2)
        if class_type == 1:
            cv2.imwrite("predict/other/"+ img_name + ".jpg", img)
        elif class_type == 2:
            cv2.imwrite("predict/scratch/"+ img_name + ".jpg", img)
        else:
            cv2.imwrite("predict/nothing/"+ img_name + ".jpg", img)