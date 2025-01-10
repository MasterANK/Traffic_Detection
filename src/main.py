import cv2
import numpy as np

# Load default MobileNetSSD model
prototxt = r'MobileNet-SSD Model\MobileNetSSD_deploy.prototxt'
model = r'MobileNet-SSD Model\MobileNetSSD_deploy.caffemodel'
max_confidence = 0.1

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

class_names = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", 
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

interested_classes = ["car", "bus", "motorbike", "bicycle"]

path = r'D:\Python\Traffic_Detection\Test Data\road Traffic 2.mp4'

cap = cv2.VideoCapture(path)
target_width = 800
if not cap.isOpened():
    print("Error: Could not open the video.")
    exit()

colors = np.random.uniform(0, 255, size=(len(class_names), 3))

while cap.isOpened():
    ret , image = cap.read()
    # aspect_ratio = image.shape[1] / image.shape[0]
    # target_height = int(target_width / aspect_ratio)
    # image = cv2.resize(image, (target_width, target_height))
    height, width = image.shape[:2]

    # Preprocess the image as BLOB (Binary Large Object)
    blob_scale = 0.007843
    blob_mean = 127.5
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), blob_scale, (300, 300), blob_mean)

    # Set the input to the network
    net.setInput(blob)

    # Run the Detection
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > max_confidence:
            class_id = int(detections[0, 0, i, 1])
            if class_names[class_id] not in interested_classes:
                continue
            legend = f"{class_names[class_id]}: {confidence:.2f}%"
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), colors[class_id], 2)
            cv2.putText(image, legend, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

    cv2.imshow("Traffic Detection", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
