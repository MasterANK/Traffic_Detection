import torch
import cv2
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# from repo 'ultralytics/yolov5'
model_path = r'YOLO Model\yolov5n.pt'
# Load YOLOv5 model from the local path
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

model.eval()

video_path = r"..\Test Data\road Traffic 2.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

target_width = 800

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

prev_frame_time = 0
new_frame_time = 0

while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_height = int(target_width / aspect_ratio)
    frame = cv2.resize(frame, (target_width, target_height))
    results = model(frame)
    
    detection = results.pandas().xyxy[0]
    car_detections = detection[detection['name'] == 'car']

    new_frame_time = time.time()
    true_fps = int(1/(new_frame_time-prev_frame_time)) 
    prev_frame_time = new_frame_time 

    #write fps on the frame
    cv2.putText(frame, f"FPS: {true_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for _,row in car_detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        confidence = f"Car: {row['confidence']:.2f}"
        cv2.putText(frame, confidence, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
