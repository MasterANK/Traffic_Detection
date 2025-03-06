import torch
import cv2
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load general vehicle model (cars, trucks)
model_path = r'YOLO Model\yolov5n.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.eval()

video_path = r"..\Test Data\ambulance.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

target_width = 800
prev_frame_time = 0
new_frame_time = 0

# Tuned HSV ranges for white, blue, and red (based on your ambulance)
lower_white = np.array([0, 0, 150])  # White body
upper_white = np.array([180, 50, 255])
lower_red1 = np.array([0, 70, 50])  # Red lights
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
lower_blue = np.array([90, 70, 70])  # Blue stripe/lights (adjusted for your ambulance)
upper_blue = np.array([130, 255, 255])

# Thresholds for color detection
WHITE_THRESHOLD = 0.0
RED_THRESHOLD = 0.000
BLUE_THRESHOLD = 0.00

# Load template for light bar (create or use your ambulance photo)
template_path = r'..\Test Data\emergencytest.png'  # Replace with your light bar template image
template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
if template is None:
    raise ValueError("Template image not found. Please provide a valid path to the light bar template.")

# Preprocess template (resize or threshold if needed)
template = cv2.resize(template, (50, 20))  # Adjust size based on your light bar (tuned for your ambulance’s light bar)
_, template_threshold = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    aspect_ratio = frame.shape[1] / frame.shape[0]
    target_height = int(target_width / aspect_ratio)
    frame = cv2.resize(frame, (target_width, target_height))
    results = model(frame)
    
    detection = results.pandas().xyxy[0]
    vehicle_detections = detection[detection['name'].isin(['car', 'truck'])]  # All vehicles
    truck_detections = detection[detection['name'] == 'truck']  # Only trucks for ambulance check
    vehicle_count = len(vehicle_detections)
    ambulance_count = 0

    new_frame_time = time.time()
    true_fps = int(1 / (new_frame_time - prev_frame_time)) 
    prev_frame_time = new_frame_time 

    # Convert frame to HSV and grayscale for contour/template analysis
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for _, row in vehicle_detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        is_ambulance = False

        # Only check for ambulances in "truck" detections
        if row['name'] == 'truck':
            roi = frame[y1:y2, x1:x2]  # ROI in BGR for contour/template
            roi_hsv = hsv_frame[y1:y2, x1:x2]  # ROI in HSV for color
            roi_gray = gray_frame[y1:y2, x1:x2]  # ROI in grayscale for contour/template

            # Color Detection (White, Red, Blue)
            mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)
            mask_red1 = cv2.inRange(roi_hsv, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(roi_hsv, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_blue = cv2.inRange(roi_hsv, lower_blue, upper_blue)

            white_pixels = cv2.countNonZero(mask_white)
            red_pixels = cv2.countNonZero(mask_red)
            blue_pixels = cv2.countNonZero(mask_blue)
            total_pixels = roi_hsv.size // 3

            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            red_ratio = red_pixels / total_pixels if total_pixels > 0 else 0
            blue_ratio = blue_pixels / total_pixels if total_pixels > 0 else 0

            # Debug: Print color ratios
            print(f"Truck at ({x1},{y1})-({x2},{y2}): White={white_ratio:.3f}, Red={red_ratio:.3f}, Blue={blue_ratio:.3f}")

            # Contour Analysis (for rectangular shape or large features)
            mask_combined = cv2.bitwise_or(mask_white, cv2.bitwise_or(mask_red, mask_blue))  # Combine colors
            contours, _ = cv2.findContours(mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            is_rectangular = False
            for contour in contours:
                area = cv2.contourArea(contour)
                if 800 < area < 4000:  # Tuned for your ambulance size (smaller range)
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    # Check for rectangular shape (e.g., aspect ratio close to 2:1 for a truck/ambulance)
                    if 1.5 <= aspect_ratio <= 3.0 and w * h > 4000:  # Adjusted thresholds
                        is_rectangular = True
                        break

            # Template Matching (for light bar) - Fixed for size mismatch
            if roi_gray.shape[0] >= template_threshold.shape[0] and roi_gray.shape[1] >= template_threshold.shape[1]:
                roi_resized = cv2.resize(roi_gray, (roi_gray.shape[1] // 2, roi_gray.shape[0] // 2))  # Downscale but ensure larger than template
                if roi_resized.shape[0] >= template_threshold.shape[0] and roi_resized.shape[1] >= template_threshold.shape[1]:
                    result = cv2.matchTemplate(roi_resized, template_threshold, cv2.TM_CCOEFF_NORMED)
                    threshold = 0.6  # Lowered threshold for better matching (tuned for your light bar)
                    loc = np.where(result >= threshold)
                    has_light_bar = len(loc[0]) > 0  # If matches found, assume light bar present
                else:
                    has_light_bar = False  # Skip if ROI is too small
            else:
                has_light_bar = False  # Skip if original ROI is too small

            # Heuristic for ambulance detection: Color AND (Shape OR Light Bar)
            is_ambulance = (white_ratio > WHITE_THRESHOLD and (red_ratio > RED_THRESHOLD or blue_ratio > BLUE_THRESHOLD)) and (is_rectangular or has_light_bar)

            # # Optional: Visualize contours and template matches for debugging
            # if is_ambulance:
            #     cv2.drawContours(roi, contours, -1, (0, 255, 0), 2)  # Draw contours in green
            #     for pt in zip(*loc[::-1]):
            #         cv2.rectangle(roi, pt, (pt[0] + template_threshold.shape[1], pt[1] + template_threshold.shape[0]), (0, 0, 255), 2)  # Draw template matches in red
            #     cv2.imshow("ROI with Contours", roi)

        # Draw bounding boxes and labels
        if is_ambulance:
            ambulance_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for ambulance
            label = f"Ambulance: {row['confidence']:.2f}"
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for car/truck
            label = f"{row['name'].capitalize()}: {row['confidence']:.2f}"

        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display stats
    cv2.putText(frame, f"FPS: {true_fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Vehicles: {vehicle_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Ambulances: {ambulance_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Helper function to calculate ratios
def calculate_ratios(mask_white, mask_red, mask_blue, total_pixels):
    white_ratio = cv2.countNonZero(mask_white) / total_pixels if total_pixels > 0 else 0
    red_ratio = cv2.countNonZero(mask_red) / total_pixels if total_pixels > 0 else 0
    blue_ratio = cv2.countNonZero(mask_blue) / total_pixels if total_pixels > 0 else 0
    return white_ratio, red_ratio, blue_ratio