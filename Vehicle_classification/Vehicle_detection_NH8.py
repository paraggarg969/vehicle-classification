import cv2
from super_gradients.training import models
import numpy as np
import math
model= models.get('yolo_nas_s',pretrained_weights='coco')

cap= cv2.VideoCapture("./videos/Vehicles_NH-8.mp4")
count= 0
ClassNames = [
    "Person", "Bicycle", "Car", "Motorcycle", "Airplane", "Bus", "Train", "Truck",
    "Boat", "Traffic light", "Fire hydrant", "Stop sign", "Parking meter", "Bench",
    "Bird", "Cat", "Dog", "Horse", "Sheep", "Cow", "Elephant", "Bear", "Zebra",
    "Giraffe", "Backpack", "Umbrella", "Handbag", "Tie", "Suitcase", "Frisbee",
    "Skis", "Snowboard", "Sports ball", "Kite", "Baseball bat", "Baseball glove",
    "Skateboard", "Surfboard", "Tennis racket", "Bottle", "Wine glass", "Cup",
    "Fork", "Knife", "Spoon", "Bowl", "Banana", "Apple", "Sandwich", "Orange",
    "Broccoli", "Carrot", "Hot dog", "Pizza", "Donut", "Cake", "Chair", "Couch",
    "Potted plant", "Bed", "Dining table", "Toilet", "TV", "Laptop", "Mouse",
    "Remote", "Keyboard", "Cell phone", "Microwave", "Oven", "Toaster", "Sink",
    "Refrigerator", "Book", "Clock", "Vase", "Scissors", "Teddy bear", "Hair drier",
    "Toothbrush"
]
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out= cv2.VideoWriter('nh8.avi',cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))


while True:
    ret,frame= cap.read()
    count+=1
    if ret:
        result= (model.predict(frame,conf=0.35))
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences=result.prediction.confidence
        labels= result.prediction.labels.tolist()
        for (bbox_xyxy,confidence, cls) in zip(bbox_xyxys,confidences,labels):
            bbox=  np.array(bbox_xyxy)
            x1,y1,x2,y2= bbox[0],bbox[1],bbox[2],bbox[3]
            classname=int(cls)
            classname=ClassNames[classname]
            conf= math.ceil((confidence*100))/100
            label= f'{classname}'
            t_size=cv2.getTextSize(label,0,fontScale=1,thickness=1)[0]
            print(t_size)
            x1, y1, x2, y2 = map(int, bbox)
            c2= x1+t_size[0], y1-(t_size[1]+10)
            cv2.rectangle(frame,(x1,y1),c2,(0,165,255),-1,cv2.LINE_AA)
            cv2.putText(frame,label,(x1,y1-5),0,1,(255,0,0),1,lineType=cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,165,255), 2)
        out.write(frame)
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1000) & 0xFF== ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()


