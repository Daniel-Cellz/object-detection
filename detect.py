from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load pretrained YOLOv8 model
model = YOLO("yolov8n.pt")

# Create folder for saved detections
os.makedirs("detections", exist_ok=True)

# Connect to phone camera stream (IP Webcam app)
cap = cv2.VideoCapture(0)

# Keep track of which IDs we've already logged (avoid spamming)
logged_ids = set()

# Open a log file
log_file = open("detection_log.txt", "a")

while True:
    cap.grab()  # discard old buffered frame to reduce lag
    ret, frame = cap.retrieve()
    if not ret:
        break

    # Run tracking (persist=True keeps IDs consistent across frames)
    results = model.track(frame, persist=True, verbose=False)[0]

    if results.boxes.id is not None:
        boxes = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().astype(int)
        class_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        for box, track_id, cls_id, conf in zip(boxes, track_ids, class_ids, confs):
            x1, y1, x2, y2 = map(int, box)
            label = model.names[cls_id]

            # Draw box + ID + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id} {conf:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Log new object the first time we see its ID
            unique_key = f"{label}_{track_id}"
            if unique_key not in logged_ids:
                logged_ids.add(unique_key)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

                # Save screenshot
                filename = f"detections/{label}_{track_id}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)

                # Write to log file
                log_line = f"[{timestamp}] New {label} detected (ID {track_id}), confidence {conf:.2f}\n"
                log_file.write(log_line)
                log_file.flush()
                print(log_line.strip())

    cv2.imshow("Object Detection + Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
log_file.close()