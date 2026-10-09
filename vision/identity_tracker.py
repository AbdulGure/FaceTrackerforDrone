import cv2
import numpy as np
import time
from person_detector import PersonDetector
from target_association import associate_face_to_person

from face_recognition import (
    app,
    OWNER_FILE,
    SIMILARITY_THRESHOLD,
    cosine_similarity
)

person_detector = PersonDetector()

# ==========================================
# CONFIGURATION
# ==========================================

VERIFY_INTERVAL = 0.5
MAX_UNVERIFIED_TIME = 3.0

# Minimum overlap between the recognised
# face and the current tracking region.
MIN_FACE_OVERLAP = 0.3

# ==========================================
# TRACKER FACTORY
# ==========================================

def create_tracker():

    # Modern OpenCV API
    if hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()

    # Legacy OpenCV API
    legacy = getattr(cv2, "legacy", None)

    if legacy is not None and hasattr(
        legacy, "TrackerCSRT_create"
    ):
        return legacy.TrackerCSRT_create()

    raise RuntimeError(
        "CSRT tracker unavailable. "
        "Install opencv-contrib-python."
    )


# ==========================================
# GEOMETRY
# ==========================================

def clamp_box(box, frame_width, frame_height):
    x, y, w, h = [int(v) for v in box]

    x1 = max(0, min(x, frame_width - 1))
    y1 = max(0, min(y, frame_height - 1))

    x2 = max(x1 + 1, min(x + w, frame_width))
    y2 = max(y1 + 1, min(y + h, frame_height))

    return x1, y1, x2 - x1, y2 - y1


def estimate_body_box(face_box, frame_width, frame_height):
    """
    Temporary approximation of a body region
    based on the position and size of a face.
    """

    x1, y1, x2, y2 = face_box

    face_width = x2 - x1
    face_height = y2 - y1

    center_x = (x1 + x2) / 2

    body_width = face_width * 3
    body_height = face_height * 6

    body_x = center_x - body_width / 2
    body_y = y1 - face_height * 0.5

    return clamp_box(
        (
            body_x,
            body_y,
            body_width,
            body_height
        ),
        frame_width,
        frame_height
    )


def intersection_over_face(face_box, body_box):
    """
    Returns the fraction of the face box
    contained within the tracking box.
    """

    fx1, fy1, fx2, fy2 = face_box

    bx, by, bw, bh = body_box

    bx2 = bx + bw
    by2 = by + bh

    ix1 = max(fx1, bx)
    iy1 = max(fy1, by)

    ix2 = min(fx2, bx2)
    iy2 = min(fy2, by2)

    intersection = (
        max(0, ix2 - ix1) *
        max(0, iy2 - iy1)
    )

    face_area = max(
        1,
        (fx2 - fx1) * (fy2 - fy1)
    )

    return intersection / face_area


# ==========================================
# MAIN
# ==========================================

def main():

    if not OWNER_FILE.exists():
        print("Owner embedding not found.")
        print("Run face enrollment first.")
        return

    owner_embedding = np.load(
        OWNER_FILE,
        allow_pickle=False
    )

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")

    tracker = None
    tracking = False

    last_verified_time = 0.0
    last_check_time = 0.0

    status = "SEARCHING FOR OWNER"

    print("F450 Identity-Aware Tracking")
    print("Press Q to quit")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            height, width = frame.shape[:2]
            now = time.monotonic()

            # ==================================
            # UPDATE TRACKER
            # ==================================

            tracked_box = None

            if tracking:

                success, box = tracker.update(frame)

                if success:
                    tracked_box = clamp_box(
                        box,
                        width,
                        height
                    )

                    status = "TRACKING OWNER"

                else:
                    tracking = False
                    tracker = None
                    status = "TARGET LOST"

            # ==================================
            # PERIODIC IDENTITY VERIFICATION
            # ==================================

            if now - last_check_time >= VERIFY_INTERVAL:

                last_check_time = now

                faces = app.get(frame)

                owner_matches = []
                unknown_faces = []

                for face in faces:

                    similarity = cosine_similarity(
                        owner_embedding,
                        face.normed_embedding
                    )

                    face_box = face.bbox.astype(int)

                    if similarity >= SIMILARITY_THRESHOLD:
                        owner_matches.append(
                            (face_box, similarity)
                        )
                    else:
                        unknown_faces.append(
                            (face_box, similarity)
                        )

                # ==================================
                # OWNER ACQUISITION
                # ==================================

                if not tracking:

                    if len(owner_matches) == 1:

                        face_box, similarity = owner_matches[0]

                        people = person_detector.detect(frame)

                        person_box = associate_face_to_person(
                            face_box,
                            people
                        )

                        if person_box is None:
                            status = "OWNER BODY NOT CONFIRMED"

                        else:
                            x1, y1, x2, y2 = person_box

                            body_box = (
                                x1,
                                y1,
                                x2 - x1,
                                y2 - y1
                            )

                            tracker = create_tracker()
                            tracker.init(frame, body_box)

                            tracking = True
                            tracked_box = body_box

                            last_verified_time = now

                            status = "OWNER ACQUIRED"

                        tracker = create_tracker()
                        tracker.init(frame, body_box)

                        tracking = True
                        tracked_box = body_box

                        last_verified_time = now

                        status = "OWNER ACQUIRED"

                        print(
                            f"Owner recognised: {similarity:.2f}"
                        )

                    elif len(owner_matches) > 1:
                        status = "AMBIGUOUS IDENTITY"

                    else:
                        status = "SEARCHING FOR OWNER"

                # ==================================
                # IDENTITY RE-VERIFICATION
                # ==================================

                else:

                    verified = False

                    if tracked_box is not None:

                        for face_box, similarity in owner_matches:

                            overlap = intersection_over_face(
                                face_box,
                                tracked_box
                            )

                            if overlap >= MIN_FACE_OVERLAP:
                                verified = True
                                last_verified_time = now
                                break

                    if verified:
                        status = "OWNER VERIFIED"

                    elif now - last_verified_time > MAX_UNVERIFIED_TIME:

                        tracking = False
                        tracker = None
                        tracked_box = None

                        status = "IDENTITY UNCERTAIN"

                        print(
                            "Identity verification expired. "
                            "Tracking suspended."
                        )

            # ==================================
            # DRAW TRACKING OUTPUT
            # ==================================

            if tracking and tracked_box is not None:

                x, y, w, h = tracked_box

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                target_x = x + w // 2
                target_y = y + h // 2

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

                cv2.line(
                    frame,
                    (frame_x, frame_y),
                    (target_x, target_y),
                    (255, 255, 0),
                    2
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

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if tracking else (0, 0, 255),
                2
            )

            cv2.imshow(
                "F450 Identity-Aware Tracker",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()