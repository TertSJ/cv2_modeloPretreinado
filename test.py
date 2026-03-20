import cv2
import torch


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap = cv2.VideoCapture(0)  
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    
    for box in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = box
        if conf > 0.5:
            label = model.names[int(cls)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()