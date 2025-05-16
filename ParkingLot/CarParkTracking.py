import cv2
from ultralytics import YOLO

model_path = r"..\ParkingLot\runs\detect\train\weights\best.pt"
model = YOLO(model_path)

video_path = r"..\video.mp4"
cap = cv2.VideoCapture(video_path)

class_names = ["car", "free"]

threshold = 0.4

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, classes=[0, 1], persist=True)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        class_ids = result.boxes.cls
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(class_ids[i])
            conf = confs[i]

            if conf < threshold:
                continue

            if class_id == 1:
                color = (157, 0, 255)
                label = "free"
            else:
                color = (0, 255, 0)
                label = "car"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f" {label}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Arac ve Bos Park Yeri Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
