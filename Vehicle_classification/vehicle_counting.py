import cv2
from super_gradients.training import models
import numpy as np
import math
from sort_b import *

model= models.get('yolo_nas_m',pretrained_weights='coco')

cap= cv2.VideoCapture("../Object_detection/videos/VehiclesEnteringandLeaving.mp4")

total_count_up=[]
total_count_down=[]


limit_down = [225, 850, 963, 850]
limit_up = [979, 850, 1667, 850]

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

tracker= Sort(max_age=20,min_hits=3,iou_threshold=0.3)

while True:
    ret,frame= cap.read()
    count+=1
    if ret:
        detections= np.empty((0,5))
        result= (model.predict(frame,conf=0.35))
        bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
        confidences=result.prediction.confidence
        labels= result.prediction.labels.tolist()
        for (bbox_xyxy,confidence, cls) in zip(bbox_xyxys,confidences,labels):
            bbox= np.array(bbox_xyxy)
            x1,y1,x2,y2= bbox[0],bbox[1],bbox[2],bbox[3]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            classname=int(cls)
            classname=ClassNames[classname]
            conf= math.ceil((confidence*100))/100
            currentArray= np.array([x1,y1,x2,y2,conf])
            detections= np.vstack((detections,currentArray))
        results_tracker= tracker.update(detections)
        cv2.line(frame,(limit_up[0],limit_up[1]), (limit_up[2],limit_up[3]), (255,0,0),5)
        cv2.line(frame,(limit_down[0],limit_down[1]), (limit_down[2],limit_down[3]),(255,0,0),5)
        for z in results_tracker:
            x1,y1,x2,y2,id= z
            x1,y1,x2,y2= int(x1), int(y1), int(x2),int(y2)
            cx,cy= int((x1+x2)/2), int((y1+y2)/2)
            cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(85,45,255),3)
            label= f'{int(id)}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - (t_size[1] + 10)
            cv2.rectangle(frame, (x1, y1), c2, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, label, (x1, y1 - 5), 0, 1, (255, 0, 0), 1, lineType=cv2.LINE_AA)

            if limit_up[0]< cx <limit_up[2] and limit_up[1] -15 < cy< limit_up[3] +15 :
                if total_count_up.count(id)== 0:
                    total_count_up.append(id)
                    cv2.line(frame, (limit_up[0], limit_up[1]), (limit_up[2], limit_up[3]), (0, 255, 0), 5)

            if limit_down[0]< cx <limit_down[2] and limit_down[1] -15 < cy< limit_down[3] +15 :
                if total_count_down.count(id)== 0:
                    total_count_down.append(id)
                    cv2.line(frame, (limit_down[0], limit_down[1]), (limit_down[2], limit_down[3]), (0, 255, 0), 5)

        t_size_up= cv2.getTextSize(("Vehicles Leaving"+ "abcd"),cv2.FONT_HERSHEY_PLAIN,3,5)[0]
        c1 = 1310 + t_size_up[0], 91 - (t_size_up[1]+13)
        cv2.rectangle(frame, (1290, 101), c1, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(frame,str("Vehicles Leaving : ") + str(len(total_count_up)),(1300,91),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)

        t_size_down = cv2.getTextSize(("Vehicles Entering" + "abcd"), cv2.FONT_HERSHEY_PLAIN, 3, 5)[0]
        c2 = 310 + t_size_down[0], 91 - (t_size_down[1] + 15)
        cv2.rectangle(frame, (290, 101), c2, (0, 165, 255), -1, cv2.LINE_AA)
        cv2.putText(frame,str("Vehicles Entering : ") +str(len(total_count_down)),(300,91),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),5)

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF== ord('q'):
            break
    else:
        break

cv2.destroyAllWindows()


