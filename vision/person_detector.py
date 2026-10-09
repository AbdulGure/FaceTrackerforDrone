import cv2
from ultralytics import YOLO


class PersonDetector:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, frame):

        results = self.model.predict(
            frame,
            classes=[0],
            conf=0.5,
            verbose=False
        )

        people = []

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .astype(int)
                )

                people.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0])
                })

        return people


if __name__ == "__main__":

    detector = PersonDetector()

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            people = detector.detect(frame)

            for person in people:

                x1, y1, x2, y2 = person["bbox"]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "PERSON",
                    (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )

            cv2.imshow(
                "F450 Person Detection",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()