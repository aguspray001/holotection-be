import json
import numpy as np
import cv2

confidenceThreshold = 0.7
NMSThreshold = 0.3

modelConfiguration = "D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/comvis-model/cfg/yolov3.cfg"
modelWeights = "D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/comvis-model/yolov3.weights"

labelsPath = "D:/AR/Thesis/backend/ObjectDetection/fastapi/assets/comvis-model/coco.names"
labels = open(labelsPath).read().strip().split('\n')

np.random.seed(10)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

def detectWithYoloV3(videoUrl):
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

    layerName = net.getLayerNames()
    layerName = [layerName[i - 1] for i in net.getUnconnectedOutLayers()]

    video = cv2.VideoCapture(videoUrl)
    (W, H) = (None, None)

    try:
        prop = cv2.CAP_PROP_FRAME_COUNT
        total = int(video.get(prop))
        print("[INFO] {} total frames in video".format(total))
    except:
        print("Could not determine no. of frames in video")

    count = 0
    while True:
        (ret, frame) = video.read()
        if not ret:
            break
        if W is None or H is None:
            (H,W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB = True, crop = False)
        net.setInput(blob)
        layersOutputs = net.forward(layerName)

        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if confidence > confidenceThreshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY,  width, height) = box.astype('int')
                    x = int(centerX - (width/2))
                    y = int(centerY - (height/2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        #Apply Non Maxima Suppression
        detectionNMS = cv2.dnn.NMSBoxes(boxes, confidences, confidenceThreshold, NMSThreshold)

        if(len(detectionNMS) > 0):
            for i in detectionNMS.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                mx = int((width+x)/2)
                my = int((y+height)/2)
                center_coordinates = (mx, my)
                print(center_coordinates)
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(labels[classIDs[i]], confidences[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # cv2.imshow('output', frame)
        if(cv2.waitKey(1) & 0XFF == ord('q')):
            break
        else:
            arrayResult = np.array(labels)[classIDs].tolist()
            return {
                "classes": arrayResult, 
                "confidences": confidences,
                'boxes': boxes
                }

    video.release()