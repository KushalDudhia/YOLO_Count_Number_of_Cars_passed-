from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
# cap = cv2.VideoCapture(0)  # For Webcam
# cap.set(3, 1280)
# cap.set(4, 720)
cap = cv2.VideoCapture("../Videos/cars.mp4")  # For Video

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
               "diningtable", "toilet", "tv monitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

mask = cv2.imread("carbg.png")

#Tracking
tracker=Sort(max_age=25,min_hits=3,iou_threshold=0.3)

limits = [400, 250, 700, 250]
totalCount = []
while True:
    success, img = cap.read()
    VideoRegion = cv2.bitwise_and(img,mask)

    imgGraphcs = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    cvzone.overlayPNG(img,imgGraphcs,(0,0))

    results = model(VideoRegion, stream=True)  #stream is used generator which will be more efficient

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box

            # x1, y1, x2, y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # print(x1, y1, x2, y2)
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print(conf)

            # Class Name
            cls =int(box.cls[0])
            # cvzone.putTextRect(img, f'{conf}',(x1,y1-20))
            # cvzone.putTextRect(img, f'{cls} {conf}', (max(0, x1), max(35, y1)),scale=1, thickness=1)

            currentClass=classNames[cls]
            if currentClass =="car" or currentClass =="motorbike" and conf > 0.3:
         #       cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(0, y1)),scale=0.3, thickness=4, offset=5)
           #     cvzone.cornerRect(img, (x1, y1, w, h), l=1)

                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))



    requestsTracker=tracker.update(detections)
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),3)

    for result in requestsTracker:
        x1,y1,x2,y2,id=result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(result)
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)

        if limits[0]<cx<limits[2] and limits[1]-10 <cy< limits[1]+10:
            if totalCount.count(id) == 0:
                totalCount.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 3)
    #cvzone.putTextRect(img, f'Count: {len(totalCount)}',(50,50))
    cv2.putText(img,str(len(totalCount)),(100,100),cv2.FONT_HERSHEY_PLAIN,5,(50,250,255),9)

    cv2.imshow("Video",img)
    # cv2.imshow("VideoRegion",img)
    cv2.waitKey(1)






