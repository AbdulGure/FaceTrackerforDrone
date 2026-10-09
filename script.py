import cv2
import speech_recognition as sr
import threading
import time
import re

# ==========================================
# CONFIGURATION
# ==========================================

WAKE_WORD = "drone"

VALID_STATES = {
    "IDLE",
    "FOLLOW",
    "HOVER",
    "LAND",
    "EMERGENCY"
}

# Shared system state
drone_state = "IDLE"
face_detected = False

# Thread synchronization
state_lock = threading.Lock()
stop_event = threading.Event()

# Face tracking settings
CENTER_TOLERANCE = 0.15
FACE_LOST_TIMEOUT = 2.0

# ==========================================
# FACE DETECTOR
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise RuntimeError("Could not load face detector")


# ==========================================
# STATE MANAGEMENT
# ==========================================

def get_state():
    with state_lock:
        return drone_state


def set_state(new_state):
    global drone_state

    if new_state not in VALID_STATES:
        print("Invalid state:", new_state)
        return

    with state_lock:

        # Emergency state requires manual reset
        if drone_state == "EMERGENCY":
            print("Emergency active. Restart required.")
            return

        if drone_state != new_state:
            drone_state = new_state
            print(f"\nSTATE CHANGED → {new_state}")


# ==========================================
# COMMAND PROCESSING
# ==========================================

def process_command(text):

    text = text.lower().strip()

    print("Recognized:", text)

    # Require wake word as a separate word
    if not re.search(r"\bdrone\b", text):
        print("Wake word not detected")
        return

    if "emergency" in text or "stop" in text:
        set_state("EMERGENCY")

    elif "land" in text:
        set_state("LAND")

    elif "follow me" in text:
        set_state("FOLLOW")

    elif "come here" in text:
        set_state("FOLLOW")

    elif "hover" in text:
        set_state("HOVER")

    elif "idle" in text:
        set_state("IDLE")

    else:
        print("Unknown command")


# ==========================================
# VOICE RECOGNITION THREAD
# ==========================================

def voice_worker():

    recognizer = sr.Recognizer()

    try:
        microphone = sr.Microphone()
    except Exception as e:
        print("Microphone initialization error:", e)
        return

    print("Voice recognition initialized")

    try:
        with microphone as source:
            print("Calibrating microphone...")

            recognizer.adjust_for_ambient_noise(
                source,
                duration=1
            )

            print("Microphone ready")

            while not stop_event.is_set():

                try:
                    print("Listening...")

                    audio = recognizer.listen(
                        source,
                        timeout=2,
                        phrase_time_limit=4
                    )

                    if stop_event.is_set():
                        break

                    text = recognizer.recognize_google(
                        audio
                    ).lower()

                    process_command(text)

                except sr.WaitTimeoutError:
                    continue

                except sr.UnknownValueError:
                    print("Speech not understood")

                except sr.RequestError as e:
                    print("Speech recognition service error:", e)
                    time.sleep(1)

                except Exception as e:
                    print("Voice error:", e)
                    time.sleep(1)

    except Exception as e:
        print("Microphone error:", e)

    print("Voice thread stopped")


# ==========================================
# FACE TRACKING
# ==========================================

def calculate_tracking(frame, face):

    x, y, w, h = face

    frame_height, frame_width = frame.shape[:2]

    face_center_x = x + w / 2
    face_center_y = y + h / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Normalized position error
    error_x = (
        face_center_x - frame_center_x
    ) / frame_center_x

    error_y = (
        face_center_y - frame_center_y
    ) / frame_center_y

    # Horizontal correction
    if error_x < -CENTER_TOLERANCE:
        horizontal = "TARGET LEFT"

    elif error_x > CENTER_TOLERANCE:
        horizontal = "TARGET RIGHT"

    else:
        horizontal = "HORIZONTAL CENTRED"

    # Vertical correction
    if error_y < -CENTER_TOLERANCE:
        vertical = "TARGET ABOVE"

    elif error_y > CENTER_TOLERANCE:
        vertical = "TARGET BELOW"

    else:
        vertical = "VERTICAL CENTRED"

    return horizontal, vertical, error_x, error_y


# ==========================================
# CAMERA THREAD / MAIN LOOP
# ==========================================

def camera_worker():

    global face_detected

    print("Starting camera...")

    cap = cv2.VideoCapture(
        0,
        cv2.CAP_AVFOUNDATION
    )

    if not cap.isOpened():
        print("ERROR: Camera unavailable")
        stop_event.set()
        return

    print("Camera ready")

    last_face_time = time.monotonic()
    last_action = None
    target_was_visible = False

    while not stop_event.is_set():

        ret, frame = cap.read()

        if not ret:
            print("Camera frame error")
            stop_event.set()
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        # Read current operating mode
        state = get_state()

        # ==================================
        # FACE DETECTION
        # ==================================

        face_detected = len(faces) > 0

        if face_detected:

            last_face_time = time.monotonic()
            target_was_visible = True

            # Select largest detected face
            face = max(
                faces,
                key=lambda f: f[2] * f[3]
            )

            x, y, w, h = face

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            horizontal, vertical, error_x, error_y = (
                calculate_tracking(frame, face)
            )

            # ==================================
            # FOLLOW MODE
            # ==================================

            if state == "FOLLOW":

                action = f"{horizontal} | {vertical}"

                if action != last_action:
                    print("TRACKING:", action)
                    last_action = action

                cv2.putText(
                    frame,
                    action,
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        # ==================================
        # TARGET LOST
        # ==================================

        else:

            if (
                state == "FOLLOW"
                and target_was_visible
                and time.monotonic() - last_face_time
                > FACE_LOST_TIMEOUT
            ):

                print("Target lost → HOVER")

                set_state("HOVER")

                last_action = None
                target_was_visible = False

        # ==================================
        # DISPLAY STATE
        # ==================================

        state = get_state()

        cv2.putText(
            frame,
            f"MODE: {state}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "F450 Drone AI Controller",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Camera stopped")


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print("==============================")
    print(" F450 AI DRONE CONTROLLER")
    print("==============================")

    print("Starting system...")

    # Start microphone in background
    voice_thread = threading.Thread(
        target=voice_worker,
        daemon=True
    )

    voice_thread.start()

    # Camera stays on main thread
    # for better macOS compatibility
    try:
        camera_worker()

    except KeyboardInterrupt:
        print("Keyboard interrupt received")

    finally:
        stop_event.set()

        voice_thread.join(timeout=3)

        print("System shutdown complete")


if __name__ == "__main__":
    main()