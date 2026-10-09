import cv2
import numpy as np
import time

from ultralytics import YOLO

from face_recognition import (
    app,
    OWNER_FILE,
    SIMILARITY_THRESHOLD,
    cosine_similarity
)

from target_association import associate_face_to_person


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "yolo11n.pt"

VERIFY_INTERVAL = 0.5
MAX_UNVERIFIED_TIME = 3.0

# ==========================================
# INITIALISATION
# ==========================================

def main():

    if not OWNER_FILE.exists():
        print("Owner embedding missing.")
        print("Run face enrollment first.")
        return

    owner_embedding = np.load(
        OWNER_FILE,
        allow_pickle=False
    )

    print("Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")

    target_id = None
    last_verified_time = 0.0
    last_check_time = 0.0

    status = "SEARCHING FOR OWNER"

    print("Multi-Person Tracking Active")
    print("Press Q to quit")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            height, width = frame.shape[:2]
            now = time.monotonic()

            # ==================================
            # YOLO + BYTETRACK
            # ==================================

            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                classes=[0],
                conf=0.35,
                verbose=False
            )

            people = []

            for result in results:

                if result.boxes is None:
                    continue

                if result.boxes.id is None:
                    continue

                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, ids):

                    x1, y1, x2, y2 = map(int, box)

                    people.append({
                        "bbox": (x1, y1, x2, y2),
                        "id": track_id
                    })

            # ==================================
            # FACE RECOGNITION
            # ==================================

            if now - last_check_time >= VERIFY_INTERVAL:

                last_check_time = now

                faces = app.get(frame)

                owner_candidates = []

                for face in faces:

                    similarity = cosine_similarity(
                        owner_embedding,
                        face.normed_embedding
                    )

                    if similarity < SIMILARITY_THRESHOLD:
                        continue

                    face_box = face.bbox.astype(int)

                    person_box = associate_face_to_person(
                        face_box,
                        people
                    )

                    if person_box is None:
                        continue

                    matching_people = [
                        person
                        for person in people
                        if person["bbox"] == person_box
                    ]

                    if len(matching_people) == 1:
                        owner_candidates.append(
                            matching_people[0]["id"]
                        )

                owner_candidates = set(owner_candidates)

                if len(owner_candidates) == 1:

                    verified_id = next(iter(owner_candidates))

                    if target_id is None:

                        target_id = verified_id
                        status = "OWNER ACQUIRED"

                        print(
                            f"Owner assigned tracking ID: {target_id}"
                        )

                    elif verified_id == target_id:

                        status = "OWNER VERIFIED"

                    else:

                        # A different ID may indicate
                        # an identity switch or reacquisition.
                        # Do not silently transfer the target.
                        target_id = None
                        status = "IDENTITY CONFLICT"

                    if target_id == verified_id:
                        last_verified_time = now

                elif len(owner_candidates) > 1:

                    target_id = None
                    status = "AMBIGUOUS OWNER"

            # ==================================
            # IDENTITY TIMEOUT
            # ==================================

            if target_id is not None:

                if now - last_verified_time > MAX_UNVERIFIED_TIME:

                    target_id = None
                    status = "IDENTITY EXPIRED"

            # ==================================
            # DRAW TRACKING RESULTS
            # ==================================

            target_visible = False

            for person in people:

                x1, y1, x2, y2 = person["bbox"]
                track_id = person["id"]

                is_target = track_id == target_id

                if is_target:

                    target_visible = True
                    color = (0, 255, 0)
                    label = f"OWNER ID {track_id}"

                else:

                    color = (255, 0, 0)
                    label = f"PERSON ID {track_id}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

                if is_target:

                    target_x = (x1 + x2) // 2
                    target_y = (y1 + y2) // 2

                    frame_x = width // 2
                    frame_y = height // 2

                    error_x = (
                        target_x - frame_x
                    ) / max(1, frame_x)

                    error_y = (
                        target_y - frame_y
                    ) / max(1, frame_y)

                    cv2.circle(
                        frame,
                        (target_x, target_y),
                        5,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        f"X: {error_x:.2f} Y: {error_y:.2f}",
                        (20, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

            if target_id is not None and not target_visible:

                status = "TARGET TEMPORARILY LOST"

                # Suspend following output while
                # the selected target is not visible.

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "F450 Multi-Person Tracking",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()