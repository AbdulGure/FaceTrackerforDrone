import cv2
import numpy as np
import argparse
from pathlib import Path
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OWNER_FILE = DATA_DIR / "owner_embedding.npy"

# Initial experimental threshold.
# Must be calibrated using genuine and
# impostor face comparisons.
SIMILARITY_THRESHOLD = 0.45

ENROLLMENT_SAMPLES = 20

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD FACE RECOGNITION MODEL
# ==========================================

print("Loading facial recognition model...")

app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(640, 640)
)

print("Model ready")


# ==========================================
# SIMILARITY CALCULATION
# ==========================================

def cosine_similarity(a, b):

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ==========================================
# CAMERA
# ==========================================

def open_camera():

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        raise RuntimeError("Camera unavailable")

    return cap


# ==========================================
# OWNER ENROLLMENT
# ==========================================

def enroll_owner():

    print("\nOWNER ENROLLMENT")
    print("Ensure only your face is visible.")
    print("Press SPACE to capture a sample.")
    print("Press Q to quit.")

    cap = open_camera()

    embeddings = []

    try:
        while len(embeddings) < ENROLLMENT_SAMPLES:

            ret, frame = cap.read()

            if not ret:
                break

            faces = app.get(frame)

            if len(faces) == 1:

                face = faces[0]

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                status = "Ready to capture"

            else:
                status = "Show exactly one face"

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Samples: {len(embeddings)}/{ENROLLMENT_SAMPLES}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow("Owner Enrollment", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == 32 and len(faces) == 1:

                embedding = faces[0].normed_embedding

                embeddings.append(
                    np.asarray(embedding, dtype=np.float32)
                )

                print(
                    f"Captured sample "
                    f"{len(embeddings)}/{ENROLLMENT_SAMPLES}"
                )

        if len(embeddings) == ENROLLMENT_SAMPLES:

            owner_embedding = np.mean(
                embeddings,
                axis=0
            )

            norm = np.linalg.norm(owner_embedding)

            if norm == 0:
                raise RuntimeError(
                    "Invalid owner embedding"
                )

            owner_embedding /= norm

            np.save(
                OWNER_FILE,
                owner_embedding
            )

            print("\nEnrollment complete")
            print("Owner profile saved locally")

        else:
            print("Enrollment incomplete")

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==========================================
# OWNER RECOGNITION
# ==========================================

def recognize_owner():

    if not OWNER_FILE.exists():

        print("No owner profile found.")
        print("Run enrollment first.")
        return

    owner_embedding = np.load(
        OWNER_FILE,
        allow_pickle=False
    )

    cap = open_camera()

    print("\nOWNER RECOGNITION ACTIVE")
    print("Press Q to quit")

    try:
        while True:

            ret, frame = cap.read()

            if not ret:
                break

            faces = app.get(frame)

            for face in faces:

                embedding = face.normed_embedding

                similarity = cosine_similarity(
                    owner_embedding,
                    embedding
                )

                x1, y1, x2, y2 = (
                    face.bbox.astype(int)
                )

                if similarity >= SIMILARITY_THRESHOLD:

                    label = "OWNER"
                    color = (0, 255, 0)

                else:

                    label = "UNKNOWN"
                    color = (0, 0, 255)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} | Similarity: {similarity:.2f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            cv2.imshow(
                "F450 Owner Recognition",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["enroll", "recognize"]
    )

    args = parser.parse_args()

    if args.mode == "enroll":
        enroll_owner()

    elif args.mode == "recognize":
        recognize_owner()