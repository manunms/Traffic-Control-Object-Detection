import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg") # model training # .cfg ? .weights ?
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()] # storing names in classes array
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread("car4.jpg")
img = cv2.resize(img, None, fx=1, fy=1)
height, width, channels = img.shape #channels?- red, green, blue

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) #scalingfactor? function-extracts features of img(pre-processing)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
counter_car = 0
counter_bus = 0
counter_truck = 0
counter_bicycle = 0
counter_motorbike = 0

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
	    

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#print(indexes)
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        cv2.putText(img, label, (x, y + 30), font, 2, color, 2)
        if label == "car":
            counter_car=counter_car+1
        if label == "bus":
            counter_bus=counter_bus+1
        if label == "truck":
            counter_truck=counter_truck+1
        if label == "bicycle":
            counter_bicycle=counter_bicycle+1
        if label == "motorbike":
            counter_motorbike=counter_motorbike+1

print("Number of car:",counter_car)
print("Number of bus:",counter_bus)
print("Number of truck:",counter_truck)
print("Number of bicycle:",counter_bicycle)
print("Number of motorbike:",counter_motorbike)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()